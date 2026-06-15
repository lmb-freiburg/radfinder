import math
import os
from functools import partial
from typing import Literal, Optional, Set, Tuple, Type, Union
from urllib.parse import urlparse

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, load_state_dict_from_file
from radfinder.models.layers import RotaryPositionEmbedding
from radfinder.models.modeling import global_pool_nlc, resample_abs_pos_embed
from radfinder.models.vision_transformer import Block
from radfinder.utils.logging_utils import log_info
from radfinder.utils.misc import to_3tuple
from timm.layers import AttentionPoolLatent, PatchDropout
from timm.models.vision_transformer import Mlp


class FeatureVisionTransformer(nn.Module):
    """Vision Transformer that accepts flattened patches as input."""

    def __init__(
        self,
        grid_size: Optional[Union[int, Tuple[int, int, int]]] = None,
        patch_dim: int = 768,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        attn_mode: str = "mha",
        q_proj_dim: Optional[int] = None,
        kv_proj_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = "learn",
        no_embed_class: bool = False,
        rope_kwargs: Optional[dict] = None,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        dynamic_grid_size: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        mlp_layer: Type[nn.Module] = Mlp,
        stored_init_kwargs: dict = None,  # hacky way to be able to recreate this module
    ) -> None:
        """
        Args:
            num_patches: Number of patches in the input.
            patch_dim: Dimension of each flattened input patch.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            attn_mode: Attention mode ('mha', 'mqa', 'mla').
            q_proj_dim: Query projection dimension for 'mla' mode.
            kv_proj_dim: Key, value projection dimension for 'mla' mode.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            pre_norm: Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
            final_norm: Enable norm after transformer blocks, before head (standard in most ViT).
            fc_norm: Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
        """
        super().__init__()
        self.stored_init_kwargs = stored_init_kwargs
        assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
        assert class_token or global_pool != "token"
        assert pos_embed in ("", "none", "learn", "rope")
        assert attn_mode in ("mha", "mqa", "mla")
        assert grid_size is not None or pos_embed in ("", "none", "rope")
        rope_kwargs = {} if rope_kwargs is None else dict(rope_kwargs)
        rope_kwargs.setdefault("dtype", torch.float32)  # robust with mixed-precision
        use_fc_norm = global_pool in ("avg", "avgmax", "max") if fc_norm is None else fc_norm
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.grid_size = None if grid_size is None else to_3tuple(grid_size)
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = (
            embed_dim  # for consistency with other models
        )
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_grid_size = dynamic_grid_size

        self.num_patches = None if grid_size is None else int(math.prod(grid_size))
        self.patch_proj = nn.Linear(patch_dim, embed_dim, proj_bias)
        self.axis_patch_proj = None  # created on demand via init_axis_patch_proj()
        self.local_cls_token = None  # created on demand via init_local_cls_token()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        assert pos_embed == "rope", f"{pos_embed=} - Only 'rope' was kept for simplicity."
        self.rope = RotaryPositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            **rope_kwargs,
        )
        # not sure if this is ever True for the global feature vision transformer, don't think so.
        self.requires_per_sample_rope = any(
            [
                self.rope.shift_coords is not None,
                self.rope.jitter_coords is not None,
                self.rope.rescale_coords is not None,
            ]
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    attn_mode=attn_mode,
                    q_proj_dim=q_proj_dim,
                    kv_proj_dim=kv_proj_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_bias=proj_bias,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.feature_info = [dict(module=f"blocks.{i}", num_chs=embed_dim) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == "map":
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def init_axis_patch_proj(self) -> None:
        """Create axis_patch_proj (embed_dim -> embed_dim) initialized from the avg-half
        of the pretrained patch_proj weights. Axis features are structurally similar to
        the avg-patch component of the [CLS; avg] input, so this initialization preserves
        pretrained knowledge for processing local detail."""
        self.stored_init_kwargs["init_axis_patch_proj"] = True
        self.axis_patch_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=self.patch_proj.bias is not None
        )
        with torch.no_grad():
            half = self.patch_proj.in_features // 2
            W_avg = self.patch_proj.weight.data[:, half:]
            self.axis_patch_proj.weight.data.copy_(W_avg)
            if self.patch_proj.bias is not None:
                self.axis_patch_proj.bias.data.copy_(self.patch_proj.bias.data)

    def init_local_cls_token(self) -> None:
        """Create a local CLS token initialized from the pretrained global CLS.
        Used for dual-CLS mode where global and local paths need separate CLS tokens."""
        self.stored_init_kwargs["init_local_cls_token"] = True
        assert self.cls_token is not None
        self.local_cls_token = nn.Parameter(self.cls_token.data.clone())
        self.num_prefix_tokens += 1

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "local_cls_token", "dist_token"}

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map" and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        grid_size: tuple[int, int, int] | torch.Tensor | None = None,
        attn_mask_3d: Optional[torch.Tensor] = None,
        coord_divisor: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        x = self.forward_features(x, grid_size, attn_mask_3d, coord_divisor)
        x = self.forward_head(x)
        return x

    def forward_features(
        self,
        x: torch.Tensor,
        grid_size: tuple[int, int, int] | torch.Tensor | None = None,
        attn_mask_3d: Optional[torch.Tensor] = None,
        coord_divisor: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        # # in case we ever need to reconstruct the grid_size from the attn_mask_3d
        # grid_size_0 = torch.sum(attn_mask_3d[:, :, 0, 0], dim=-1)
        # grid_size_1 = torch.sum(attn_mask_3d[:, 0, :, 0], dim=-1)
        # grid_size_2 = torch.sum(attn_mask_3d[:, 0, 0, :], dim=-1)
        # grid_size_new = torch.stack([grid_size_0, grid_size_1, grid_size_2], dim=-1)
        # assert (grid_size_new == grid_size).all()
        x = self.patch_proj(x)
        x = self.forward_features_projected(x, grid_size, attn_mask_3d, coord_divisor)
        return x

    def forward_features_projected(
        self,
        x: torch.Tensor,
        grid_size: tuple[int, int, int] | torch.Tensor | None = None,
        attn_mask_3d: Optional[torch.Tensor] = None,
        coord_divisor: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Like forward_features but skips patch_proj (input is already projected).
        Used for axis features that go through axis_patch_proj externally."""
        assert x.ndim == 3, f"Expected input with 3 dimensions (B, N, C), got {x.ndim}."
        x, rope = self._pos_embed(x, grid_size, coord_divisor=coord_divisor)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x, rope=rope, grid_size=grid_size, attn_mask_3d=attn_mask_3d)
        x = self.norm(x)
        return x

    def _pos_embed(
        self,
        x: torch.Tensor,  # (B=1, N=80, embed_dim=1080)
        grid_size: tuple[int, int, int] | torch.Tensor | None = None,  # (Hp=4, Wp=4, Dp=5)
        coord_divisor: tuple[int, int, int] | torch.Tensor | None = None,
    ):
        B = x.shape[0]
        if isinstance(grid_size, torch.Tensor) and grid_size.ndim == 2:
            if coord_divisor is not None:
                assert (
                    isinstance(coord_divisor, torch.Tensor)
                    and coord_divisor.ndim == 2
                    and coord_divisor.shape[0] == B
                )
            # (B, 3) grids (default case, i think it works now)
            rope = []
            for bi, (lH, lW, lD) in enumerate(grid_size.tolist()):
                H_div, W_div, D_div = (
                    coord_divisor[bi] if coord_divisor is not None else (None, None, None)
                )
                rope.append(self.rope(H=lH, W=lW, D=lD, H_div=H_div, W_div=W_div, D_div=D_div))

            # # one max rope for everything (TOO GOOD results)
            # max_grid = grid_size.max(dim=0).values
            # rope = self.rope(H=max_grid[0], W=max_grid[1], D=max_grid[2])
        else:
            H, W, D = to_3tuple(grid_size)
            H_div, W_div, D_div = None, None, None
            if coord_divisor is not None:
                H_div, W_div, D_div = to_3tuple(coord_divisor)
            if self.requires_per_sample_rope:
                rope = [
                    self.rope(H=H, W=W, D=D, H_div=H_div, W_div=W_div, D_div=D_div)
                    for _ in range(B)
                ]
            else:
                rope = self.rope(H=H, W=W, D=D, H_div=H_div, W_div=W_div, D_div=D_div)

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.local_cls_token is not None:
            to_cat.append(self.local_cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)

        return self.pos_drop(x), rope

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    @classmethod
    def from_pretrained(
        cls, checkpoint_path_or_url: Union[str, os.PathLike], verbose: bool = True, **kwargs
    ) -> "FeatureVisionTransformer":
        """Load pretrained model weights from a local path or a URL."""
        model = cls(**kwargs, stored_init_kwargs=kwargs)

        def _is_url(path: str) -> bool:
            try:
                parsed = urlparse(str(path))
                return parsed.scheme in ("http", "https")
            except Exception:
                return False

        def _is_hf_url(path: str) -> bool:
            try:
                parsed = urlparse(str(path))
                return "huggingface.co" in parsed.netloc
            except Exception:
                return False

        if _is_hf_url(checkpoint_path_or_url):
            if verbose:
                log_info(
                    f"Downloading pretrained weights from Hugging Face URL: {checkpoint_path_or_url}"
                )
            # Extract repo_id and filename from the URL
            parsed = urlparse(checkpoint_path_or_url)
            parts = parsed.path.strip("/").split("/")
            repo_id = "/".join(parts[:2])  # e.g., 'cclaess/SPECTRE'
            filename = parts[-1]  # e.g., 'spectre_backbone_vit_large_patch16_128.pt'

            local_path = hf_hub_download(repo_id=repo_id, filename=filename)
            state_dict = load_state_dict_from_file(local_path, map_location="cpu")
        elif _is_url(checkpoint_path_or_url):
            if verbose:
                log_info(f"Downloading pretrained weights from URL: {checkpoint_path_or_url}")
            state_dict = torch.hub.load_state_dict_from_url(
                checkpoint_path_or_url, map_location="cpu", weights_only=False, progress=verbose
            )
        else:
            local_path = os.fspath(checkpoint_path_or_url)
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Checkpoint file not found: {local_path}")
        if verbose:
            log_info(f"Loading checkpoint from local path: {local_path}")
            state_dict = torch.load(local_path, map_location="cpu", weights_only=False)

        msg = model.load_state_dict(state_dict, strict=False)
        if verbose:
            log_info(f"Loaded pretrained weights with msg: {msg}")
        return model


def feat_vit_tiny(
    patch_dim,
    checkpoint_path_or_url: Optional[str] = None,
    **kwargs,
) -> FeatureVisionTransformer:
    """Feature ViT-Tiny model."""
    kwargs = dict(
        patch_dim=patch_dim,
        embed_dim=192,
        depth=2,
        num_heads=2,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return FeatureVisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return FeatureVisionTransformer(**kwargs, stored_init_kwargs=kwargs)


def feat_vit_small(
    patch_dim,
    checkpoint_path_or_url: Optional[str] = None,
    **kwargs,
) -> FeatureVisionTransformer:
    """Feature ViT-Small model."""
    kwargs = dict(
        patch_dim=patch_dim,
        embed_dim=384,
        depth=2,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return FeatureVisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return FeatureVisionTransformer(**kwargs, stored_init_kwargs=kwargs)


def feat_vit_base(
    patch_dim,
    checkpoint_path_or_url: Optional[str] = None,
    **kwargs,
) -> FeatureVisionTransformer:
    """Feature ViT-Base model."""
    kwargs = dict(
        patch_dim=patch_dim,
        embed_dim=768,
        depth=2,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return FeatureVisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return FeatureVisionTransformer(**kwargs, stored_init_kwargs=kwargs)


def feat_vit_large(
    patch_dim,
    checkpoint_path_or_url: Optional[str] = None,
    **kwargs,
) -> FeatureVisionTransformer:
    """Feature ViT-Large model."""
    kwargs = dict(
        patch_dim=patch_dim,
        embed_dim=1080,
        depth=4,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return FeatureVisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return FeatureVisionTransformer(**kwargs, stored_init_kwargs=kwargs)
