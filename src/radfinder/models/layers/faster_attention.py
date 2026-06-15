"""
Kernel comparison:

CUDNN: 36.23GB
FLASH: 36.18GB
EFFICIENT: 36.19GB
MATH: 48.75GB

Note that rope embeddings are recreated many times, that could be optimized.
"""

from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from radfinder.models.layers.rotary_pos_embed import rope_apply
from radfinder.utils.collate import pad_and_stack
from timm.layers import use_fused_attn
from torch.jit import Final
from torch.nn.attention import SDPBackend, sdpa_kernel


class FasterAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mode: str = "mha",
        q_proj_dim: Optional[int] = None,
        kv_proj_dim: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """
        {
            "self": Attention(),
            "dim": 1080,
            "num_heads": 12,
            "mode": "mha",
            "qproj_dim": None,
            "kv_proj_dim": None,
            "qkv_bias": True,
            "qk_norm": False,
            "proj_bias": True,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "norm_layer": "torch.nn.modules.normalization.LayerNorm",
        }
        """
        super().__init__()
        # print(f"Creating attention layer with args: {locals()}")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()
        assert self.fused_attn, "Fused attention is not available, without it, you will OOM"
        self.mode = mode.lower()
        assert self.mode in ["mha", "mqa", "mla"], "Attention mode must be 'mha', 'mqa', or 'mla'"
        assert not (
            self.mode == "mla" and kv_proj_dim is None
        ), "kv_proj_dim must be provided for 'mla' mode"
        assert not (
            self.mode == "mla" and q_proj_dim is None
        ), "q_proj_dim must be provided for 'mla' mode"

        if self.mode == "mha":
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)  # Key and value pair for every head
            self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        elif self.mode == "mqa":
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(
                dim, 2 * self.head_dim, bias=qkv_bias
            )  # Key and value pair shared across heads
            self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        elif self.mode == "mla":
            self.q_proj = nn.Linear(
                dim, q_proj_dim, bias=qkv_bias
            )  # Projected query for every head
            self.kv_proj = nn.Linear(
                dim, kv_proj_dim, bias=qkv_bias
            )  # Projected key and value pair for every head
            self.q_norm = norm_layer(q_proj_dim) if qk_norm else nn.Identity()
            self.kv_norm = norm_layer(kv_proj_dim) if qk_norm else nn.Identity()
            self.q = nn.Linear(q_proj_dim, dim, bias=qkv_bias)  # Query for every head
            self.kv = nn.Linear(
                kv_proj_dim, 2 * dim, bias=qkv_bias
            )  # Key and value pair for every head

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        grid_size: torch.Tensor | None = None,
        attn_mask_3d: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to the query and key tensors.
        Args:
            q: Query (B, num_heads, N, head_dim) e.g. [100, 12, 151, 90]
            k: Key (B, num_heads, N, head_dim) same
            rope (sin, cos) tensors for RoPE application. varying input.
            grid_size: Grid size of the image patches (B, 3)
            attn_mask_3d: Attention mask of shape (B, Hp_max, Wp_max, Dp_max)
        """
        q_dtype = q.dtype
        q_device = q.device
        if isinstance(rope, tuple):
            # case 1: simple tuple of sin, cos each shape (Hp*Wp*Dp, head_dim)
            sin, cos = rope
            sin = sin.unsqueeze(0).unsqueeze(0)  # unsqueeze num_heads and batch
            cos = cos.unsqueeze(0).unsqueeze(0)
            n_spatial = sin.shape[-2]  # number of spatial tokens covered by rope
        elif attn_mask_3d is None and isinstance(rope, list):
            # case 2: list of tuples of sin, cos with all same shape
            rope = tuple(torch.stack([r[i] for r in rope], dim=0) for i in range(2))
            sin, cos = rope
            sin, cos = sin.unsqueeze(1), cos.unsqueeze(1)  # pyright: ignore  # unsqueeze num_heads
            n_spatial = sin.shape[-2]  # number of spatial tokens covered by rope
        elif attn_mask_3d is not None and isinstance(rope, list):
            # case 3: list of tuples of sin, cos, each of different shape (Hp*Wp*Dp, head_dim)
            # construct the rope in 3d with padding
            B, Hp_max, Wp_max, Dp_max = attn_mask_3d.shape
            head_dim = q.shape[-1]
            sin3d = torch.zeros(B, Hp_max, Wp_max, Dp_max, head_dim, dtype=q_dtype, device=q_device)
            cos3d = torch.zeros(B, Hp_max, Wp_max, Dp_max, head_dim, dtype=q_dtype, device=q_device)
            for i in range(B):
                Hp, Wp, Dp = grid_size[i]
                sin3d[i, :Hp, :Wp, :Dp] = rope[i][0].view(Hp, Wp, Dp, head_dim)
                cos3d[i, :Hp, :Wp, :Dp] = rope[i][1].view(Hp, Wp, Dp, head_dim)
            # then reshape back to 1D: (B, n_spatial, head_dim)
            n_spatial = Hp_max * Wp_max * Dp_max
            sin = sin3d.view(B, n_spatial, head_dim).unsqueeze(1)  # (B, 1, n_spatial, head_dim)
            cos = cos3d.view(B, n_spatial, head_dim).unsqueeze(1)
        else:
            raise ValueError(f"{type(rope)=} and {type(attn_mask_3d)=} not understood.")

        n_total = q.shape[-2]  # total tokens per sample
        n_prefix = n_total - n_spatial  # e.g., [cls] or [reg] tokens at the front
        assert n_prefix >= 0, "RoPE sin/cos length exceeds sequence length"

        # cast sin, cos to q/k dtype to save memory, instead of the other way around.
        sin = sin.to(dtype=q_dtype)
        cos = cos.to(dtype=q_dtype)

        if n_prefix > 0:
            q_prefix = q[:, :, :n_prefix, :]
            k_prefix = k[:, :, :n_prefix, :]
            q_spatial = q[:, :, n_prefix:, :]
            k_spatial = k[:, :, n_prefix:, :]
        else:
            q_prefix = k_prefix = None
            q_spatial, k_spatial = q, k

        # Apply RoPE on the spatial tail
        q_spatial = rope_apply(q_spatial, sin, cos)
        k_spatial = rope_apply(k_spatial, sin, cos)

        # Stitch back
        if n_prefix > 0:
            q = torch.cat((q_prefix, q_spatial), dim=-2)
            k = torch.cat((k_prefix, k_spatial), dim=-2)
        else:
            q, k = q_spatial, k_spatial
        return q, k

    def compute_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        if self.mode == "mha":
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
        elif self.mode == "mqa":
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, N, 2, 1, self.head_dim).permute(2, 0, 3, 1, 4)
            kv = kv.expand(-1, -1, self.num_heads, -1, -1)  # Expand to match num_heads
            k, v = kv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
        elif self.mode == "mla":
            q = self.q_proj(x)
            kv = self.kv_proj(x)
            q, kv = self.q_norm(q), self.kv_norm(kv)  # Normalization on projections
            q = self.q(q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(kv).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
        return q, k, v

    def compute_attention_no_mask(
        self,
        q: torch.Tensor,  # (B, num_heads, num_tokens, head_dim)
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,  # (B, num_tokens)
    ) -> torch.Tensor:
        """The flash attention kernel supports head_dim of 90, but not attention masks"""
        assert attn_mask is None, "Attention masks are not supported for flash attention"
        B, _, N, _ = q.shape
        C = self.num_heads * self.head_dim
        assert self.fused_attn, "Non-fused attention disabled."
        # (B, num_heads broadcast, source_len broadcast, target_len N_tokens)
        mask = attn_mask[:, None, None, :] if attn_mask is not None else None
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        # # never use manu branch it's too memory intensive
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = attn @ v
        return x.transpose(1, 2).reshape(B, N, C)

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The cudnn and efficient attention kernels support masks, but not head_dim of 90.
        So we pad it to a multiple of 8 first.
        """
        B, H, N, D = q.shape
        C = H * D
        # Make layouts friendly for SDPA kernels
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        # Pad head_dim to multiple of 8 for EFFICIENT_ATTENTION or CUDNN_ATTENTION
        Dp = (D + 7) // 8 * 8
        pad = Dp - D
        if pad:
            # pad last dim (head_dim)
            q_ = F.pad(q, (0, pad))
            k_ = F.pad(k, (0, pad))
            v_ = F.pad(v, (0, pad))
        else:
            q_, k_, v_ = q, k, v
        drop = self.attn_drop.p if self.training else 0.0
        mask = attn_mask[:, None, None, :] if attn_mask is not None else None
        backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        if q_.device.type != "cuda":
            # The fused kernels are CUDA-only; add the math kernel so SDPA has a
            # viable backend on CPU (and other non-CUDA devices).
            backends.append(SDPBackend.MATH)
        with sdpa_kernel(backends):
            x = F.scaled_dot_product_attention(
                q_,
                k_,
                v_,
                dropout_p=drop,
                scale=self.scale,  # IMPORTANT: keep 1/sqrt(90), not 1/sqrt(96)
                is_causal=False,
                attn_mask=mask,
            )
        if pad:
            x = x[..., :D]  # back to head_dim=90
        return x.transpose(1, 2).reshape(B, N, C)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, embed_dim)
        rope=None,
        grid_size=None,
        attn_mask_3d: torch.Tensor | None = None,  # shape (B, Hp_max, Wp_max, Dp_max)
    ) -> torch.Tensor:

        q, k, v = self.compute_qkv(x)
        if rope is not None:
            q, k = self.apply_rotary_pos_emb(
                q, k, rope, grid_size=grid_size, attn_mask_3d=attn_mask_3d
            )
        attn_mask = None
        if attn_mask_3d is not None:
            attn_mask = convert_3d_mask_to_1d_mask(attn_mask_3d, q.shape[-2], q.device)
        x = self.compute_attention(q, k, v, attn_mask=attn_mask)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def convert_3d_mask_to_1d_mask(
    attn_mask_3d: torch.Tensor, N_total_max: int, device: torch.device
) -> torch.Tensor:
    """Convert 3d attn mask to 1d attn mask for attention layer."""
    B, Hp_max, Wp_max, Dp_max = attn_mask_3d.shape
    N_spatial_max = Hp_max * Wp_max * Dp_max
    N_prefix = N_total_max - N_spatial_max
    attn_mask = torch.zeros(B, N_total_max, dtype=torch.bool, device=device)
    attn_mask[:, :N_prefix] = True  # prefix tokens are always attended to
    attn_mask[:, N_prefix:] = attn_mask_3d.view(B, -1)
    return attn_mask
