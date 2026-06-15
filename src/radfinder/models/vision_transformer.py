"""
Reference hyperparameters for ViT models:

vit_tiny_patch16_128:
    img_size=(128, 128, 64), patch_size=(16, 16, 8), embed_dim=192, depth=12,
    num_heads=2, mlp_ratio=4, qkv_bias=True

vit_small_patch16_128:
    img_size=(128, 128, 64), patch_size=(16, 16, 8), embed_dim=384, depth=12,
    num_heads=4, mlp_ratio=4, qkv_bias=True

vit_base_patch16_128:
    img_size=(128, 128, 64), patch_size=(16, 16, 8), embed_dim=768, depth=12,
    num_heads=8, mlp_ratio=4, qkv_bias=True

vit_base_patch32_128:
    img_size=(128, 128, 64), patch_size=(32, 32, 16), embed_dim=768, depth=12,
    num_heads=8, mlp_ratio=4, qkv_bias=True

vit_large_patch16_128:
    img_size=(128, 128, 64), patch_size=(16, 16, 8), embed_dim=1080, depth=24,
    num_heads=12, mlp_ratio=4, qkv_bias=True

vit_large_patch32_128:
    img_size=(128, 128, 64), patch_size=(32, 32, 16), embed_dim=1080, depth=24,
    num_heads=12, mlp_ratio=4, qkv_bias=True
"""

import os
from functools import partial
from typing import List, Literal, Optional, Sequence, Set, Tuple, Type, Union
from urllib.parse import urlparse

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, load_state_dict_from_file
from radfinder.models.layers import PatchEmbed, RotaryPositionEmbedding
from radfinder.models.layers.faster_attention import FasterAttention
from radfinder.models.modeling import feature_take_indices, global_pool_nlc
from radfinder.utils.logging_utils import log_info
from radfinder.utils.misc import model_print
from timm.layers import AttentionPoolLatent, PatchDropout
from timm.models.vision_transformer import DropPath, LayerScale, Mlp
from torch.utils.checkpoint import checkpoint


def get_default_snippet_alignment():
    return {"enabled": False}


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_mode: str = "mha",
        q_proj_dim: Optional[int] = None,
        kv_proj_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FasterAttention(
            dim,
            num_heads=num_heads,
            mode=attn_mode,
            q_proj_dim=q_proj_dim,
            kv_proj_dim=kv_proj_dim,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, rope=None, grid_size=None, attn_mask_3d=None
    ) -> torch.Tensor:
        x = x + self.drop_path1(
            self.ls1(
                self.attn(self.norm1(x), rope=rope, grid_size=grid_size, attn_mask_3d=attn_mask_3d)
            )
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer with 3D Patch Embedding

    Commented values are for the spectre large checkpoint
    """

    def __init__(
        self,
        sliding_window_size: Union[int, Tuple[int, int, int]] = (
            128,
            128,
            64,
        ),  # same, but unused for rope
        patch_size: Union[int, Tuple[int, int, int]] = (16, 16, 8),  # same
        in_chans: int = 1,
        num_classes: int = 1000,  # 0
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map"] = "token",  # ""
        embed_dim: int = 768,  # 1080
        depth: int = 12,  # 24
        num_heads: int = 12,  # same
        attn_mode: str = "mha",
        q_proj_dim: Optional[int] = None,
        kv_proj_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,  # same
        qkv_bias: bool = True,  # same
        qk_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,  # 1.0, layer scale hyperparameter
        class_token: bool = True,
        pos_embed: str = "learn",  # "rope"
        no_embed_class: bool = False,
        rope_kwargs: Optional[dict] = None,  # {'base': 1000.0}
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        mlp_layer: Type[nn.Module] = Mlp,
        half_patch_embed: bool = False,
        pixdim: Tuple[float, float, float] = (0.75, 0.75, 1.5),  # unused, kept for compatibility
        grad_checkpoint_every: int = 0,
        min_area_for_padding: float = 0.0,  # unused, kept for compatibility
    ) -> None:
        """
        Args:
            sliding_window_size: Input window size (renamed from img_size for clarity)
            patch_size: Patch size.
            in_chans: Number of image input channels.
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
            pos_embed: Type of position embedding to use (default: 'learn').
            no_embed_class: Don't include position embeddings for class (or reg) tokens for learnable pos_embed.
            rope_kwargs: Additional arguments for rotary position embedding.
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
        assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
        assert class_token or global_pool != "token"
        assert pos_embed == "rope"  # in ("", "none", "learn", "rope")
        assert attn_mode in ("mha", "mqa", "mla")

        self.sliding_window_size = sliding_window_size
        self.patch_size = patch_size
        self.half_patch_embed = half_patch_embed
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
        self.dynamic_img_size = dynamic_img_size
        if grad_checkpoint_every is None or grad_checkpoint_every < 0:
            self.grad_checkpoint_every = 0
        else:
            self.grad_checkpoint_every = int(grad_checkpoint_every)

        embed_args = {}
        if self.dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWDC"))
        elif pos_embed == "rope":
            embed_args["output_fmt"] = "NHWDC"
        self.patch_embed = PatchEmbed(
            img_size=sliding_window_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        assert pos_embed == "rope", f"{pos_embed=} - Only 'rope' was kept for simplicity."
        rope_kwargs = {} if rope_kwargs is None else dict(rope_kwargs)
        rope_kwargs.setdefault("dtype", torch.float32)  # robust with mixed-precision
        self.rope = RotaryPositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            **rope_kwargs,
        )
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

        use_fc_norm = global_pool in ("avg", "avgmax", "max") if fc_norm is None else fc_norm
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
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

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token"}

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

    def set_input_size(
        self,
        img_size: Optional[Tuple[int, int, int]] = None,
        patch_size: Optional[Tuple[int, int, int]] = None,
    ):
        """
        Method updates the input image resolution, patch size

        Unused for Rope.

        Args:
            img_size: New input resolution, if None current resolution is used
            patch_size: New patch size, if None existing patch size is used
        """
        _prev_grid_size = self.patch_embed.grid_size
        self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)

    def _pos_embed(self, x: torch.Tensor):
        B, H, W, D, C = x.shape
        x = x.view(B, -1, C)

        if self.requires_per_sample_rope:
            rope = [self.rope(H=H, W=W, D=D) for _ in range(B)]
        else:
            rope = self.rope(
                H=H, W=W, D=D
            )  # 2-tuple sin, cos (H*W*D, head_dim=embed_dim/num_heads)

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
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

    def forward(self, x: torch.Tensor, attn_mask_3d: torch.Tensor | None = None) -> torch.Tensor:
        model_print(f"VisionTransformer input shape: {x.shape}")
        x = self.forward_features(x, attn_mask_3d=attn_mask_3d)
        model_print(f"VisionTransformer head input shape: {x.shape}")
        x = self.forward_head(x)
        return x

    def forward_features(
        self, x: torch.Tensor, attn_mask_3d: torch.Tensor | None = None
    ) -> torch.Tensor:
        # print(f"Input shape: {x.shape}")
        # # nearest is better than trilinear
        # x = F.interpolate(x, size=(128, 128, 64), mode="nearest")

        # # instead upsample by repeating in depth dimension so i know exactly what happens
        # BN, _, H, W, D = x.shape  # B * N, 1, H, W, D
        # x = x.unsqueeze(-1).repeat(1, 1, 1, 1, 1, 2).reshape(BN, 1, H, W, D * 2).contiguous()
        # print(f"Upsampled shape: {x.shape}")

        # x: (B * N, 1, H, W, D) e.g. 288, 1, 128, 128, 64
        model_print(f"Patch embedding input shape: {x.shape}")
        x = self.patch_embed(x)  # (B * N, Hp, Wp, Dp, embed_dim) e.g. 288, 8, 8, 8, 1080
        # print(f"Patch embedded shape: {x.shape}")
        model_print(f"Positional embedding input shape: {x.shape}")
        x, rope = self._pos_embed(x)
        # rope is tuple of (sin, cos) each of shape (512, 90) where 512=8x8x8 and 90=1080/12heads
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        grad_checkpoint_every = self.get_grad_checkpoint_every()
        for i, blk in enumerate(self.blocks):
            if i == 0:
                model_print(f"First block of {len(self.blocks)} input shape: {x.shape}")
            if grad_checkpoint_every > 0 and ((i + 1) % grad_checkpoint_every == 0):
                x = checkpoint(
                    lambda _x, _rope=rope, _attn_mask_3d=attn_mask_3d: blk(
                        _x, rope=_rope, attn_mask_3d=_attn_mask_3d
                    ),
                    x,
                    use_reentrant=False,
                )
            else:
                x = blk(x, rope=rope, attn_mask_3d=attn_mask_3d)
        model_print(f"Blocks output shape: {x.shape}")
        x = self.norm(x)
        return x

    def get_grad_checkpoint_every(self) -> int:
        grad_checkpoint_every = self.grad_checkpoint_every
        if not torch.is_grad_enabled():
            grad_checkpoint_every = 0
        if not self.training:
            grad_checkpoint_every = 0
        return int(grad_checkpoint_every)

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
    ) -> "VisionTransformer":
        """Load pretrained model weights from a local path or a URL."""
        model = cls(**kwargs)

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

        model.post_weight_loading()

        return model

    def post_weight_loading(self):
        if self.half_patch_embed:
            self.apply_half_patch_embed()

    def apply_half_patch_embed(self):
        """
        Half the size of the Conv3d patch embedding, to balance out half resolution input.

        Old input to the model:
            images at 0.75x0.75x1.5mm
            windows of size (128, 128, 64)
            patch size (16, 16, 8) -> 8x8x8 patches per window
        New input to the model:
            images at 0.75x0.75x3mm
            windows of size (128, 128, 32)
            patch size (16, 16, 4) -> 8x8x8 patches per window

        So here we have to reduce the patch embedding size by half in the D dimension.
        """
        # update the patch_size information in the model
        self.patch_size = (self.patch_size[0], self.patch_size[1], self.patch_size[2] // 2)
        log_info("Changing patch_embed from (16x16x8) to (16x16x4) by downsampling weights.")
        assert self.rope is not None, "Only supported for rope"
        assert isinstance(self.patch_embed, PatchEmbed), "Unexpected patch embed class"
        assert isinstance(self.patch_embed.norm, nn.Identity), "Unexpected patch embed norm"
        bias = self.patch_embed.proj.bias is not None

        # create new weights
        conv_weight = self.patch_embed.proj.weight  # [1080, 1, 16, 16, 8])
        _patch_dim = conv_weight.shape[0]
        device = conv_weight.device
        is_meta = conv_weight.is_meta
        # bias is (1080,) so it can stay the same
        # assuming the weights are already loaded, we now have to downsample them

        # option 1: use every 2nd slice -> model breaks.
        # option 2: average pool each 2 -> ok but not great
        # option 3: trilinear sampling, same result as average pooling
        # option 4: low pass filter + decimate, not very good
        # option 5: low pass filter + decimate + rms matching per out channel, same as avg pooling
        # option 6: fold kernel (summation)
        if not is_meta:
            conv_weight_new = (conv_weight[..., ::2] + conv_weight[..., 1::2]).contiguous()

            new_param_dict = {"proj.weight": conv_weight_new.detach().cpu().clone()}
            if bias:
                new_param_dict["proj.bias"] = self.patch_embed.proj.bias.detach().cpu().clone()

        # recreate the patch embedding at half the patch size in dim D, otherwise same params
        new_patch_embed = PatchEmbed(
            img_size=(128, 128, 32),
            patch_size=(16, 16, 4),
            in_chans=conv_weight.shape[1],
            embed_dim=conv_weight.shape[0],
            norm_layer=None,
            flatten=self.patch_embed.flatten,
            output_fmt=self.patch_embed.output_fmt,
            bias=bias,
            strict_img_size=self.patch_embed.strict_img_size,
            dynamic_img_pad=self.patch_embed.dynamic_img_pad,
        )

        if is_meta:
            # HF `from_pretrained` builds the model under `init_empty_weights`,
            # so parameters are on the meta device and have no data. Skip the
            # fold; the safetensors loader writes proj.weight/bias afterwards.
            new_patch_embed = new_patch_embed.to_empty(device="meta")
        else:
            # load new weights into the new patch_embed
            new_patch_embed.load_state_dict(new_param_dict, strict=True)
            new_patch_embed = new_patch_embed.to(device)

        # attach it to the model
        self.patch_embed = new_patch_embed
        self.patch_embed.eval()

        # num_patches stays the same (8x8x8) so rope, embed_dim etc. are unchanged
        # so this is all we need to do
