"""
Gaussian-smoothed intra-scan localization loss.

Given snippet text and per-depth image features from the same scan, classify which
depth position the snippet belongs to. Soft Gaussian target handles noisy annotations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from radfinder.utils.logging_utils import log_debug

_VERBOSE = False


def _gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 1D Gaussian kernel with radius = ceil(3*sigma)."""
    radius = max(int(sigma * 3 + 0.5), 1)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


# NEG_INF = -50000.0
NEG_INF = float("-inf")


class GaussianLocalizationLoss(nn.Module):
    def __init__(
        self, sigma: float = 2.0, tau: float = 0.1, learnable_tau: bool = False, eps: float = 1e-8
    ):
        """
        Args:
            sigma: Gaussian spread in depth-position units. sigma=2 ~ 24mm at 12mm/pos.
                   Set to 0 to disable Gaussian smoothing (hard target).
            tau: Temperature for cosine similarity logits (init value if learnable).
            learnable_tau: If True, tau is learned as log(1/tau) parameter.
            eps: Epsilon for numerical stability in target normalization.
        """
        super().__init__()
        self.sigma = sigma
        self.eps = eps
        if learnable_tau:
            # Store log(1/tau) as learnable parameter, same convention as SigLIP's init_t
            self.log_inv_tau = nn.Parameter(torch.tensor(float(tau)).log().neg())
        else:
            self.log_inv_tau = None
            self.tau = tau

    def forward(
        self,
        slice_emb: torch.Tensor,
        snippet_emb: torch.Tensor,
        target_depth_mask: torch.Tensor,
        valid_depth_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Note: D2 = Dpmax * p2, entire scan is (B, Hpmax, Wpmax, Dpmax, E)
        Args:
            slice_emb: (S, D2, E) — per-slice image features from feature combiner
            snippet_emb:    (S, E)     — snippet text embeddings (projected to same dim)
            target_depth_mask:(S, D2) — boolean mask for label slice
            valid_depth_mask:(S, D2) — boolean mask for all input slices
        Returns:
            Scalar loss (mean over valid snippets).
        """
        if _VERBOSE:
            log_debug(
                f"[LocLoss] input: slice_emb {slice_emb.shape}, text {snippet_emb.shape}, "
                f"mask {target_depth_mask.shape}, avg True/slice: "
                f"{target_depth_mask.sum() / target_depth_mask.shape[0]:.1f}"
            )

        # Slices where filtered before so we can assume there is exactly one True slice per snippet
        assert (target_depth_mask.sum(dim=-1) == 1).all(), f"{target_depth_mask=}"
        # no targets are out of bounds
        assert (
            (target_depth_mask * valid_depth_mask).sum(dim=-1) == 1
        ).all(), f"{target_depth_mask=}, {valid_depth_mask=}"

        # Cosine similarity -> logits
        slice_emb = F.normalize(slice_emb, dim=-1)  # (S, D2, E)
        snippet_emb = F.normalize(snippet_emb, dim=-1)  # (S, E)
        inv_tau = self.log_inv_tau.exp() if self.log_inv_tau is not None else 1.0 / self.tau
        logits = torch.einsum("sde,se->sd", slice_emb, snippet_emb) * inv_tau  # (S, D2)
        logits = logits.masked_fill(~valid_depth_mask, NEG_INF)

        if _VERBOSE:
            tau_val = 1.0 / inv_tau.item() if isinstance(inv_tau, torch.Tensor) else self.tau
            log_debug(
                f"[LocLoss] logits: min={logits.min():.3f} max={logits.max():.3f} "
                f"mean={logits.mean():.3f} nan={logits.isnan().sum()} inf={logits.isinf().sum()} tau={tau_val:.4f}"
            )

        # Build soft target from depth mask distribution
        # No logit masking — CE gradient (softmax - target) naturally pushes probability away from
        # non-target positions
        target = target_depth_mask.float()  # (S, D2)
        target = target / (target.sum(dim=-1, keepdim=True) + self.eps)

        if _VERBOSE:
            log_debug(f"[LocLoss] target[0] nonzero at: {target[0].nonzero().squeeze(-1).tolist()}")

        # 1D Gaussian convolution along depth axis (skip if sigma == 0)
        if self.sigma > 0:
            kernel = _gaussian_kernel_1d(self.sigma, target.device, target.dtype)
            padding = len(kernel) // 2
            target = F.conv1d(
                target.unsqueeze(1), kernel.reshape(1, 1, -1), padding=padding
            ).squeeze(1)
            target = target / (target.sum(dim=-1, keepdim=True) + self.eps)
            # if gaussian blur creates labels outside the mask, set to 0 again
            target = target.masked_fill(~valid_depth_mask, 0.0)
            if _VERBOSE:
                log_debug(
                    f"[LocLoss] after blur (sigma={self.sigma}): top-5 pos={target[0].topk(5).indices.tolist()} vals={[f'{v:.4f}' for v in target[0].topk(5).values.tolist()]}"
                )

        # get log_probs. -inf values will stay -inf (softmax of -inf is 0, and log of 0 is -inf)
        log_probs = F.log_softmax(logits, dim=-1)
        # multiply by the target. masked values will be nan because -inf * 0 = nan
        per_logit_loss = -(target * log_probs)
        # finally we mask the nan values to 0 so they don't influence the sum here
        per_sample_loss = per_logit_loss.masked_fill(~valid_depth_mask, 0.0).sum(dim=-1)
        loss = per_sample_loss.mean()

        if _VERBOSE:
            pred_depth = logits.argmax(dim=-1)
            target_depth = target.argmax(dim=-1)
            correct = (pred_depth == target_depth).float().mean()
            log_debug(
                f"[LocLoss] loss={loss:.4f} (min={per_sample_loss.min():.4f} "
                f"max={per_sample_loss.max():.4f}), acc={correct:.3f}, "
                f"pred={pred_depth.tolist()[:10]}, tgt={target_depth.tolist()[:10]}"
            )
        return loss
