"""
Pool-based retrieval evaluation (Merlin protocol).

Evaluates text-to-image retrieval over non-overlapping pools of fixed sizes,
separately for Findings, Impressions, and Full Report text sections.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from radfinder.models.modeling import last_token_pool
from radfinder.models.vision_language import SigLIP
from radfinder.utils.logging_utils import log_info
from transformers import Qwen2TokenizerFast

from packg.tqdmext import tqdm_max_ncols
from visiontext.distutils import is_main_process


def run_pool_retrieval(
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    device: str = "cuda",
    model_config: dict | None = None,
    pool_sizes: list[int] | None = None,
    ks: list[int] | None = None,
    repeats: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    if pool_sizes is None:
        pool_sizes = [32, 64, 128]
    if ks is None:
        ks = [1, 8]
    if repeats is None:
        repeats = 100
    if seed is None:
        seed = 42
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    model = model.to(device)
    model.eval()

    # Phase 1: collect image embeddings via forward pass
    all_image_embeddings = []
    pbar = tqdm_max_ncols(
        total=len(dataloader),
        desc="Pool retrieval: images",
        smoothing=0,
        disable=not is_main_process(),
    )
    for batch in dataloader:
        pbar.update(1)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model(batch)
        all_image_embeddings.append(output.image_embeddings.cpu().float().numpy())
    pbar.close()
    image_emb = np.concatenate(all_image_embeddings, axis=0).astype(np.float32)
    N = image_emb.shape[0]

    # Phase 2: encode 3 text variants separately
    # Text format matches RandomReportTransformd (generate_report.py):
    #   findings stripped of "Impressions"/"impressions" to avoid header leakage
    findings_texts = []
    impressions_texts = []
    full_report_texts = []
    for item in dataset.data:
        f = item.get("findings", [""])[0] if item.get("findings") else ""
        imp = item.get("impressions", [""])[0] if item.get("impressions") else ""
        if f:
            f_str = f"Findings: {f}\n".replace("Impressions", "").replace("impressions", "")
        else:
            f_str = ""
        findings_texts.append(f_str)
        impressions_texts.append(f"Impressions: {imp}\n" if imp else "")
        parts = []
        if f:
            parts.append(f_str)
        if imp:
            parts.append(f"Impressions: {imp}\n")
        full_report_texts.append("".join(parts))

    assert len(findings_texts) == N

    # --- Text statistics ---
    n_with_findings = sum(1 for t in findings_texts if t.strip())
    n_with_impressions = sum(1 for t in impressions_texts if t.strip())
    n_findings_eq_full = sum(1 for f, fu in zip(findings_texts, full_report_texts) if f == fu)
    f_lens = [len(t) for t in findings_texts]
    i_lens = [len(t) for t in impressions_texts]
    fu_lens = [len(t) for t in full_report_texts]
    ratios = [f / fu if fu > 0 else 0.0 for f, fu in zip(f_lens, fu_lens)]
    log_info(
        f"Text stats: findings={n_with_findings}/{N}, impressions={n_with_impressions}/{N}, "
        f"findings==full={n_findings_eq_full}/{N}"
    )
    log_info(f"  Findings chars: mean={np.mean(f_lens):.0f}, median={np.median(f_lens):.0f}")
    log_info(f"  Impressions chars: mean={np.mean(i_lens):.0f}, median={np.median(i_lens):.0f}")
    log_info(f"  Full chars: mean={np.mean(fu_lens):.0f}, median={np.median(fu_lens):.0f}")
    log_info(f"  Ratio findings/full: mean={np.mean(ratios):.3f}")

    tokenizer = Qwen2TokenizerFast.from_pretrained(model_config["text_tokenizer"])
    normalize = model.criterion.normalize if model.criterion is not None else True

    findings_emb = encode_text_list(
        findings_texts,
        model.backbone_text,
        model.projection_text,
        tokenizer,
        device,
        normalize=normalize,
    )
    impressions_emb = encode_text_list(
        impressions_texts,
        model.backbone_text,
        model.projection_text,
        tokenizer,
        device,
        normalize=normalize,
    )
    full_report_emb = encode_text_list(
        full_report_texts,
        model.backbone_text,
        model.projection_text,
        tokenizer,
        device,
        normalize=normalize,
    )

    # Normalize image embeddings
    if normalize:
        image_emb = F.normalize(torch.from_numpy(image_emb), p=2, dim=-1).numpy()

    # --- Embedding similarity diagnostics ---
    # Cosine similarity between findings-only and full-report embeddings (per sample)
    find_full_cos = np.sum(findings_emb * full_report_emb, axis=1)
    impr_full_cos = np.sum(impressions_emb * full_report_emb, axis=1)
    find_img_cos = np.sum(findings_emb * image_emb, axis=1)
    full_img_cos = np.sum(full_report_emb * image_emb, axis=1)
    log_info(f"Pool retrieval: {N} samples, {image_emb.shape[1]}d embeddings")
    log_info(
        f"  cos(findings, full): mean={np.mean(find_full_cos):.4f}, "
        f"median={np.median(find_full_cos):.4f}, min={np.min(find_full_cos):.4f}"
    )
    log_info(
        f"  cos(impressions, full): mean={np.mean(impr_full_cos):.4f}, "
        f"median={np.median(impr_full_cos):.4f}"
    )
    log_info(
        f"  cos(findings, image): mean={np.mean(find_img_cos):.4f} "
        f"vs cos(full, image): mean={np.mean(full_img_cos):.4f}"
    )

    # Phase 3: compute pool-based recall for each text variant
    # Use all N samples for each variant (empty texts get dummy embeddings via encode_text_list)
    all_metrics: dict[str, float] = {"n": N}
    all_metrics["n_with_findings"] = n_with_findings
    all_metrics["n_with_impressions"] = n_with_impressions

    for prefix, emb in [
        ("find", findings_emb),
        ("impr", impressions_emb),
        ("full", full_report_emb),
    ]:
        metrics = compute_pool_recall(image_emb, emb, pool_sizes, ks, repeats, seed)
        for k, v in metrics.items():
            all_metrics[f"{prefix}_{k}"] = v
        if verbose:
            log_info(f"{prefix} ({N} samples): {metrics}")

    return all_metrics


@torch.inference_mode()
def encode_text_list(
    texts: list[str],
    text_backbone,
    text_projection,
    tokenizer,
    device: str,
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    all_emb = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        # Replace empty strings with a space to avoid tokenizer issues
        batch_texts = [t if t.strip() else " " for t in batch_texts]
        tok = tokenizer(
            batch_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=4096,
        )
        input_ids = torch.tensor(tok["input_ids"]).to(device)
        attention_mask = torch.tensor(tok["attention_mask"]).to(device)
        emb = text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        emb = last_token_pool(emb.last_hidden_state, attention_mask)
        emb = text_projection(emb)
        emb = emb.cpu().float()
        if normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        all_emb.append(emb.numpy())
    return np.concatenate(all_emb, axis=0).astype(np.float32)


def compute_pool_recall(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    pool_sizes: list[int],
    ks: list[int],
    repeats: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    N = len(image_emb)
    assert len(text_emb) == N

    # L2 normalize
    img_norms = np.linalg.norm(image_emb, axis=1, keepdims=True)
    txt_norms = np.linalg.norm(text_emb, axis=1, keepdims=True)
    eps = 1e-8
    image_emb = image_emb / np.clip(img_norms, eps, None)
    text_emb = text_emb / np.clip(txt_norms, eps, None)

    rng = np.random.default_rng(seed)
    results: dict[str, list[float]] = {}
    for ps in pool_sizes:
        for k in ks:
            results[f"pool{ps}_r{k}"] = []

    for _ in range(repeats):
        perm = np.arange(N)
        rng.shuffle(perm)

        for ps in pool_sizes:
            total_counts = {k: 0 for k in ks}
            total_queries = 0

            for start in range(0, N - ps + 1, ps):
                pool_idx = perm[start : start + ps]
                sim = image_emb[pool_idx] @ text_emb[pool_idx].T  # (ps, ps)
                ranks = np.argsort(-sim, axis=0)  # text→image: sort columns
                for k in ks:
                    for j in range(ps):
                        if j in ranks[:k, j]:
                            total_counts[k] += 1
                total_queries += ps

            if total_queries > 0:
                for k in ks:
                    results[f"pool{ps}_r{k}"].append(total_counts[k] / total_queries)

    metrics = {}
    for key, values in results.items():
        if values:
            metrics[key] = float(np.mean(values))
    return metrics
