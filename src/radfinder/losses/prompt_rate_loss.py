"""
Prompt-supervised RaTE classification loss.

Computes BCE loss between image-prompt similarity logits and binary RaTE labels.
Used as auxiliary loss alongside SigLIP contrastive training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from packg.constclass import Const


class PromptSamplingC(Const):
    FIRST_ONLY = "first_only"
    RANDOM_1 = "random_1"
    RANDOM_1_TO_N = "random_1_to_n"
    MEAN_ALL = "mean_all"


class PromptRateLoss(nn.Module):
    """Prompt-supervised RaTE classification loss."""

    def __init__(
        self,
        pos_weight_per_q: torch.Tensor,
        label_smoothing: float = 0.0,
        question_weight_per_q: torch.Tensor | None = None,
    ):
        """
        Args:
            pos_weight_per_q: (num_questions,) per-question positive class weight for BCE.
            question_weight_per_q: (num_questions,) per-question importance weight for BCE.
                Scales entire loss element. Used to control relative contribution of
                question groups (e.g. upweight CT-RATE in mode=both).
        """
        super().__init__()
        self.register_buffer("pos_weight_per_q", pos_weight_per_q)
        self.register_buffer("question_weight_per_q", question_weight_per_q)
        self.label_smoothing = label_smoothing

    def forward(
        self,
        img_emb: torch.Tensor,  # (B, 512) L2-normalized
        prompt_emb: torch.Tensor,  # (Q, 3, 2, 512) L2-normalized
        labels: torch.Tensor,  # (B, Q) with -1 for unknown
        t: torch.Tensor,  # scalar (log temperature)
        b: torch.Tensor,  # scalar bias
        prompt_sampling_mode: str,
    ) -> torch.Tensor | None:
        """Compute masked BCE loss on logit difference (pos - neg).

        Returns None if no valid labels in batch.
        """
        B, _ = img_emb.shape
        pe = sample_prompts(prompt_emb, prompt_sampling_mode, B)  # (B, Q, 2, 512)
        logits = torch.einsum("bd,bqv d->bqv", img_emb, pe) * torch.exp(t) + b  # (B, Q, 2)
        logits_pos = logits[:, :, 0]  # (B, Q)
        logits_neg = logits[:, :, 1]  # (B, Q)
        logit_diff = logits_pos - logits_neg  # bias cancels but kept for generality
        mask = labels != -1
        if mask.sum() == 0:
            return None
        if self.label_smoothing > 0.0:
            labels = labels.float()
            labels[mask] = labels[mask] * (1 - self.label_smoothing * 2) + self.label_smoothing

        pw = self.pos_weight_per_q.unsqueeze(0).expand_as(logit_diff)
        kwargs = dict(pos_weight=pw[mask])
        if self.question_weight_per_q is not None:
            qw = self.question_weight_per_q.unsqueeze(0).expand_as(logit_diff)
            kwargs["weight"] = qw[mask]
        # default mean divides by number of unmasked elements.
        # so gradient for one step will be similar regardless of batch size and number of classes
        loss = F.binary_cross_entropy_with_logits(
            logit_diff[mask],  # (N_valid_labels,)
            labels[mask].float(),  # (N_valid_labels,)
            **kwargs,
        )
        return loss


def sample_prompts(prompt_emb, prompt_sampling_mode, B) -> list[int]:
    """Sample prompts

    Args:
        prompt_emb: (Q, 3, 2, 512) L2-normalized
        prompt_sampling_mode: one of PromptSamplingC options
        B: batch size (for random sampling modes)

    Returns:
        prompt_emb: (B or 1, Q, 2, 512) sampled prompt embeddings for pos and neg variants
    """
    Q, V, _, _ = prompt_emb.shape  # Q: num questions, V: num variants
    device = prompt_emb.device
    if prompt_sampling_mode == PromptSamplingC.FIRST_ONLY:
        return prompt_emb[:, 0, :, :].unsqueeze(0)  # (1, Q, 2, 512)
    if prompt_sampling_mode == PromptSamplingC.RANDOM_1:
        indices = torch.randint(0, V, size=(B, Q)).to(device)
        arange = torch.arange(Q).unsqueeze(0).to(device)
        return prompt_emb[arange, indices]  # (B, Q, 2, 512)
    if prompt_sampling_mode == PromptSamplingC.RANDOM_1_TO_N:
        indices = torch.randint(0, V, size=(B, Q, V)).to(device)  # (B, Q, V) in {0, ..., V-1}
        n_indices = torch.randint(1, V + 1, size=(B, Q)).to(device)  # (B, Q) in {1, ..., V}
        mask_t = torch.arange(V)[None, None].to(device) < n_indices.unsqueeze(-1)  # (B, Q, V) bool
        mask = mask_t[..., None, None]  # (B, Q, V, 1, 1)
        arange = torch.arange(Q)[None, :, None].to(device)
        prompt_emb_selected = prompt_emb[arange, indices]  # (B, Q, V, 2, 512)
        prompt_emb = (prompt_emb_selected * mask).sum(dim=2) / mask.sum(dim=2)  # (B, Q, 2, 512)
        return prompt_emb
    if prompt_sampling_mode == PromptSamplingC.MEAN_ALL:
        return prompt_emb.mean(dim=1).unsqueeze(0)  # (1, Q, 2, 512)
    raise ValueError(f"Unknown prompt_sampling mode: {prompt_sampling_mode=}")
