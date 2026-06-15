from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from radfinder.models.vision_language import SigLIP
from radfinder.tasks.binary_zs_ctrate_task import (
    PATHOLOGIES,
    _encode_text_latents,
    _predict_rate,
    evaluate_chexzero,
    load_binary_labels,
)
from radfinder.tasks.localization_task import MM_PER_DEPTH_POSITION
from radfinder.tasks.localization_task import _compute_metrics as _loc_compute_metrics
from radfinder.tasks.localization_task import (
    _compute_perscan_metrics as _loc_compute_perscan_metrics,
)
from radfinder.tasks.localization_task import run_localization
from radfinder.tasks.pool_retrieval_task import encode_text_list
from radfinder.tasks.retrieval_task import compute_retrieval_cosine_r100
from radfinder.tasks.volume_retrieval_task import compute_map_at_k
from radfinder.utils.bootstrap import (
    DEFAULT_BOOTSTRAP_CI,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP,
    bootstrap_ci_multi,
    ci_from_repeats,
)
from radfinder.utils.logging_utils import log_info
from transformers import Qwen2TokenizerFast

from packg.tqdmext import tqdm_max_ncols
from visiontext.distutils import is_main_process


def bootstrap_standard_retrieval(
    ranks: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Bootstrap CIs for R@k, MedR, MeanR from per-sample ranks array."""
    ranks = np.asarray(ranks, dtype=np.float64)
    metric_fns: dict[str, callable] = {}
    for k in (1, 5, 10, 50, 100):
        metric_fns[f"r{k}"] = _recall_at_k_fn(k)
    metric_fns["medr"] = lambda x: float(np.floor(np.median(x)) + 1)
    metric_fns["meanr"] = lambda x: float(np.mean(x) + 1)

    results = bootstrap_ci_multi(ranks, metric_fns, n_bootstrap=n_bootstrap, ci=ci, seed=seed)

    out: dict[str, float] = {"n": len(ranks)}
    for name, (point, lo, hi) in results.items():
        out[name] = point
        out[f"{name}_ci_lo"] = lo
        out[f"{name}_ci_hi"] = hi
        out[f"{name}_ci_half"] = (hi - lo) / 2
    return out


def _recall_at_k_fn(k: int):
    def fn(ranks: np.ndarray) -> float:
        return float(np.mean(ranks < k))

    return fn


def pool_retrieval_ci(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    pool_sizes: list[int],
    ks: list[int],
    repeats: int = 100,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> dict[str, float]:
    """Pool retrieval with CIs computed directly from the per-repeat values."""
    N = len(image_emb)
    assert len(text_emb) == N

    img_norms = np.linalg.norm(image_emb, axis=1, keepdims=True)
    txt_norms = np.linalg.norm(text_emb, axis=1, keepdims=True)
    eps = 1e-8
    image_emb = image_emb / np.clip(img_norms, eps, None)
    text_emb = text_emb / np.clip(txt_norms, eps, None)

    rng = np.random.default_rng(seed)
    per_repeat: dict[str, list[float]] = {}
    for ps in pool_sizes:
        for k in ks:
            per_repeat[f"pool{ps}_r{k}"] = []

    for _ in range(repeats):
        perm = np.arange(N)
        rng.shuffle(perm)
        for ps in pool_sizes:
            total_counts = {k: 0 for k in ks}
            total_queries = 0
            for start in range(0, N - ps + 1, ps):
                pool_idx = perm[start : start + ps]
                sim = image_emb[pool_idx] @ text_emb[pool_idx].T
                pool_ranks = np.argsort(-sim, axis=0)
                for k in ks:
                    for j in range(ps):
                        if j in pool_ranks[:k, j]:
                            total_counts[k] += 1
                total_queries += ps
            if total_queries > 0:
                for k in ks:
                    per_repeat[f"pool{ps}_r{k}"].append(total_counts[k] / total_queries)

    out: dict[str, float] = {}
    for key, values in per_repeat.items():
        if not values:
            continue
        mean, lo, hi = ci_from_repeats(values, ci=ci)
        out[key] = mean
        out[f"{key}_ci_lo"] = lo
        out[f"{key}_ci_hi"] = hi
        out[f"{key}_ci_half"] = (hi - lo) / 2
    return out


def compute_map_at_k_per_sample(
    similarities: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """Like compute_map_at_k but returns per-query AP scores instead of the mean."""
    n = len(similarities)
    ap_scores = np.empty(n)
    for i in range(n):
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        query_labels = labels[i]
        num_relevant = 0
        precision_sum = 0.0
        for rank, j in enumerate(top_k_indices, 1):
            intersection = np.sum(query_labels & labels[j])
            union = np.sum(query_labels | labels[j])
            if intersection > 0 and union > 0:
                num_relevant += 1
                precision_sum += num_relevant / rank
        ap_scores[i] = precision_sum / k
    return ap_scores


def bootstrap_volume_retrieval(
    similarities: np.ndarray,
    labels: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 50),
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Bootstrap CIs for MAP@k from per-query AP scores."""
    out: dict[str, float] = {"n": len(similarities)}
    for k in ks:
        ap_scores = compute_map_at_k_per_sample(similarities, labels, k)
        results = bootstrap_ci_multi(
            ap_scores,
            {f"vol_map{k}": lambda x, _k=k: float(np.mean(x))},
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )
        for name, (point, lo, hi) in results.items():
            out[name] = point
            out[f"{name}_ci_lo"] = lo
            out[f"{name}_ci_hi"] = hi
            out[f"{name}_ci_half"] = (hi - lo) / 2
    return out


def run_retrieval_with_bootstrap(
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    device: str = "cuda",
    verbose: bool = False,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Like run_retrieval() but also returns bootstrap CIs.

    Returns: (standard_metrics, bootstrap_metrics)
    The standard_metrics dict is identical to what run_retrieval() would produce.
    """
    model = model.to(device)
    model.eval()
    log_info(f"Evaluating on {len(dataset)=}, {device=}, {dataloader=}")
    all_image_embeddings = []
    all_text_embeddings = []
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm_max_ncols(
        total=len(dataloader),
        desc="Evaluating retrieval",
        smoothing=0,
        disable=not is_main_process(),
    )
    for i, batch in enumerate(dataloader):
        pbar.update(1)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model(batch)
        image_embeddings = output.image_embeddings
        text_embeddings = output.text_embeddings
        assert image_embeddings is not None and text_embeddings is not None
        if model.criterion is not None:
            loss = model.criterion(image_embeddings, text_embeddings)
            total_loss += loss.detach().item()
            num_batches += 1
        all_image_embeddings.append(image_embeddings.float().cpu().numpy())
        all_text_embeddings.append(text_embeddings.float().cpu().numpy())
    pbar.close()

    all_image_embeddings = np.concatenate(all_image_embeddings, axis=0).astype(np.float32)
    all_text_embeddings = np.concatenate(all_text_embeddings, axis=0).astype(np.float32)

    img_norms = np.linalg.norm(all_image_embeddings, axis=1, keepdims=True)
    txt_norms = np.linalg.norm(all_text_embeddings, axis=1, keepdims=True)
    eps = 1e-8
    all_image_embeddings /= np.clip(img_norms, eps, None)
    all_text_embeddings /= np.clip(txt_norms, eps, None)

    similarities = all_image_embeddings @ all_text_embeddings.T
    dot = torch.from_numpy(similarities)

    metrics_t2i, other_t2i = compute_retrieval_cosine_r100(dot.T)

    loss_nonaccum = None
    if model.criterion is not None and num_batches > 0:
        loss_nonaccum = total_loss / num_batches

    standard_metrics = {(f"t2i_{k}" if k != "n" else "n"): v for k, v in metrics_t2i.items()}
    standard_metrics["loss_nonaccum"] = loss_nonaccum

    # Bootstrap t2i
    ranks_t2i = other_t2i["ranks"].numpy()
    bootstrap_t2i = bootstrap_standard_retrieval(
        ranks_t2i,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
    )
    bootstrap_metrics: dict[str, float] = {
        (f"t2i_{k}" if k != "n" else "n"): v for k, v in bootstrap_t2i.items()
    }
    bootstrap_metrics["n_bootstrap"] = n_bootstrap
    bootstrap_metrics["ci_level"] = ci

    return standard_metrics, bootstrap_metrics


def run_pool_retrieval_with_bootstrap(
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    device: str = "cuda",
    model_config: dict | None = None,
    pool_sizes: list[int] | None = None,
    ks: list[int] | None = None,
    repeats: int = 100,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    verbose: bool = False,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Like run_pool_retrieval() but also returns bootstrap CIs.

    Returns: (standard_metrics, bootstrap_metrics)
    """
    if pool_sizes is None:
        pool_sizes = [32, 64, 128]
    if ks is None:
        ks = [1, 8]

    model = model.to(device)
    model.eval()

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

    n_with_findings = sum(1 for t in findings_texts if t.strip())
    n_with_impressions = sum(1 for t in impressions_texts if t.strip())

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

    if normalize:
        image_emb = F.normalize(torch.from_numpy(image_emb), p=2, dim=-1).numpy()

    standard_metrics: dict[str, float] = {
        "n": N,
        "n_with_findings": n_with_findings,
        "n_with_impressions": n_with_impressions,
    }
    bootstrap_metrics: dict[str, float] = {
        "n": N,
        "n_bootstrap": 0,  # will be set per-variant
        "ci_level": ci,
    }

    for prefix, emb in [
        ("find", findings_emb),
        ("impr", impressions_emb),
        ("full", full_report_emb),
    ]:
        ci_metrics = pool_retrieval_ci(
            image_emb,
            emb,
            pool_sizes,
            ks,
            repeats=repeats,
            seed=seed,
            ci=ci,
        )
        for k, v in ci_metrics.items():
            full_key = f"{prefix}_{k}"
            if "_ci_" in k:
                bootstrap_metrics[full_key] = v
            else:
                standard_metrics[full_key] = v
                bootstrap_metrics[full_key] = v

    return standard_metrics, bootstrap_metrics


def bootstrap_binary_zs(
    predictions: np.ndarray,
    labels: np.ndarray,
    pathologies: list[str] | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Bootstrap CIs for binary ZS metrics (mean_auroc, mean_prec, mean_f1, mean_acc)."""
    point_main, _ = evaluate_chexzero(predictions, labels, pathologies=pathologies, verbose=False)

    rng = np.random.default_rng(seed)
    n = len(predictions)
    metric_names = ["mean_auroc", "mean_prec", "mean_f1", "mean_acc"]
    boot_values = {m: np.empty(n_bootstrap) for m in metric_names}

    pbar = tqdm_max_ncols(
        total=n_bootstrap,
        desc="Bootstrap binary ZS",
        smoothing=0,
        disable=not is_main_process(),
    )
    for b in range(n_bootstrap):
        pbar.update(1)
        idx = rng.integers(0, n, size=n)
        main_b, _ = evaluate_chexzero(
            predictions[idx],
            labels[idx],
            pathologies=pathologies,
            verbose=False,
        )
        for m in metric_names:
            boot_values[m][b] = main_b[m]
    pbar.close()

    alpha = (1 - ci) / 2
    out: dict[str, float] = {"n": n, "n_bootstrap": n_bootstrap, "ci_level": ci}
    for m in metric_names:
        out[m] = point_main[m]
        lo = float(np.percentile(boot_values[m], 100 * alpha))
        hi = float(np.percentile(boot_values[m], 100 * (1 - alpha)))
        out[f"{m}_ci_lo"] = lo
        out[f"{m}_ci_hi"] = hi
        out[f"{m}_ci_half"] = (hi - lo) / 2
    return out


def bootstrap_localization(
    pred: np.ndarray,
    target: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Bootstrap CIs for localization metrics from per-snippet pred/target arrays."""
    errors = np.abs(pred.astype(np.float64) - target.astype(np.float64))

    metric_fns: dict[str, callable] = {
        "loc_mae_mm": lambda e: float(np.mean(e) * MM_PER_DEPTH_POSITION),
        "loc_median_mm": lambda e: float(np.median(e) * MM_PER_DEPTH_POSITION),
        "loc_mae_positions": lambda e: float(np.mean(e)),
        "loc_acc_exact": lambda e: float(np.mean(e == 0)),
        "loc_acc_within_12mm": lambda e: float(np.mean(e <= 1)),
        "loc_acc_within_24mm": lambda e: float(np.mean(e <= 2)),
    }

    results = bootstrap_ci_multi(errors, metric_fns, n_bootstrap=n_bootstrap, ci=ci, seed=seed)

    out: dict[str, float] = {"n_snippets": len(pred), "n_bootstrap": n_bootstrap, "ci_level": ci}
    for name, (point, lo, hi) in results.items():
        out[name] = point
        out[f"{name}_ci_lo"] = lo
        out[f"{name}_ci_hi"] = hi
        out[f"{name}_ci_half"] = (hi - lo) / 2
    return out


def run_volume_retrieval_with_bootstrap(
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    device: str = "cuda",
    dataset_name: str = "ctrate",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[dict[str, float], dict[str, float]]:
    """Like run_volume_retrieval() but also returns bootstrap CIs."""
    model = model.to(device)
    model.eval()
    log_info(f"Volume retrieval: {len(dataset)=}, {device=}")

    all_image_embeddings = []
    pbar = tqdm_max_ncols(
        total=len(dataloader),
        desc="Volume retrieval embeddings",
        smoothing=0,
        disable=not is_main_process(),
    )
    for batch in dataloader:
        pbar.update(1)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model(batch)
        assert output.image_embeddings is not None
        all_image_embeddings.append(output.image_embeddings.cpu().numpy())
    pbar.close()

    all_image_embeddings = np.concatenate(all_image_embeddings, axis=0).astype(np.float32)
    norms = np.linalg.norm(all_image_embeddings, axis=1, keepdims=True)
    all_image_embeddings /= np.clip(norms, 1e-8, None)

    labels = load_binary_labels(dataset_name, dataset)
    assert len(labels) == len(all_image_embeddings)

    abnormal_mask = labels.sum(axis=1) > 0
    n_total = len(labels)
    embeddings = all_image_embeddings[abnormal_mask]
    labels_abnormal = labels[abnormal_mask]
    n_abnormal = len(embeddings)
    log_info(f"Volume retrieval: {n_abnormal}/{n_total} abnormal volumes")

    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, -np.inf)

    standard_metrics: dict[str, float] = {
        "vol_n_total": n_total,
        "vol_n_abnormal": n_abnormal,
    }
    for k in (5, 10, 50):
        standard_metrics[f"vol_map{k}"] = compute_map_at_k(similarities, labels_abnormal, k)

    bootstrap_metrics = bootstrap_volume_retrieval(
        similarities,
        labels_abnormal,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
    )
    bootstrap_metrics["n_bootstrap"] = n_bootstrap
    bootstrap_metrics["ci_level"] = ci

    return standard_metrics, bootstrap_metrics


def run_binary_zs_with_bootstrap(
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    device: str = "cuda",
    dataset_name: str = "ctrate",
    model_config: dict | None = None,
    prompt_mode: str = "t3",
    radchestct_label_mapping: str = "extended",
    eval_protocol: str = "default",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Like run_binary_zs() but also returns bootstrap CIs.

    Returns: (main_metrics, bootstrap_metrics, aux_metrics)
    """
    model = model.to(device)
    model.eval()

    labels = load_binary_labels(
        dataset_name, dataset, radchestct_label_mapping=radchestct_label_mapping
    )

    log_info(f"Evaluating binary ZS on {len(dataset)} samples, {device=}, {prompt_mode=}")
    all_image_embeddings = []
    pbar = tqdm_max_ncols(
        total=len(dataloader),
        desc="Extracting image embeddings",
        smoothing=0,
        disable=not is_main_process(),
    )
    for batch in dataloader:
        pbar.update(1)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model.forward_image_only(batch)
        all_image_embeddings.append(output.image_embeddings_secondary.cpu().float().numpy())
    pbar.close()

    all_image_embeddings = np.concatenate(all_image_embeddings, axis=0).astype(np.float32)

    tokenizer = Qwen2TokenizerFast.from_pretrained(model_config["text_tokenizer"])
    normalize = model.criterion.normalize

    image_emb = torch.from_numpy(all_image_embeddings).cpu()
    if normalize:
        image_emb = F.normalize(image_emb, p=2, dim=-1)
    image_emb_np = image_emb.numpy()

    if prompt_mode == "rate":
        predictedall = _predict_rate(model, tokenizer, image_emb_np, device, normalize)
    else:
        text_latents = _encode_text_latents(model, tokenizer, device, normalize, prompt_mode)
        n_path = len(PATHOLOGIES)
        sims = np.einsum("nd,pd->np", image_emb_np, text_latents)
        logits = sims.reshape(len(image_emb_np), n_path, 2)
        probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1)
        predictedall = probs[:, :, 0].numpy()

    pathologies_eval = None
    if eval_protocol == "radchestct_standard":
        predictedall[:, 1] = np.maximum(predictedall[:, 1], predictedall[:, 4])
        keep = [i for i in range(len(PATHOLOGIES)) if i not in (4, 13)]
        predictedall = predictedall[:, keep]
        labels = labels[:, keep]
        pathologies_eval = [PATHOLOGIES[i] for i in keep]
        log_info(f"[radchestct_standard] Merged calcification, evaluating {len(keep)} classes")

    main_metrics, aux_metrics = evaluate_chexzero(
        predictedall, labels, pathologies=pathologies_eval
    )

    bootstrap_metrics = bootstrap_binary_zs(
        predictedall,
        labels,
        pathologies=pathologies_eval,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
    )

    return main_metrics, bootstrap_metrics, aux_metrics


def run_localization_with_bootstrap(
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    device: str = "cuda",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Like run_localization() but also returns bootstrap CIs.

    Returns: (standard_metrics, bootstrap_metrics)
    """
    model = model.to(device)
    model.eval()
    log_info(f"Evaluating localization on {len(dataset)=}, {device=}")

    all_pred_depths: list[np.ndarray] = []
    all_target_depths: list[np.ndarray] = []
    all_valid_min: list[int] = []
    all_valid_max: list[int] = []
    all_scan_idx: list[np.ndarray] = []
    total_snippets = 0
    scan_offset = 0

    pbar = tqdm_max_ncols(
        total=len(dataloader),
        desc="Evaluating localization",
        smoothing=0,
        disable=not is_main_process(),
    )
    for batch in dataloader:
        pbar.update(1)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model.forward_localization(batch)

        if output.scan_slice_emb is None or output.snippet_emb is None:
            scan_offset += batch["image_grid_shape"].shape[0]
            continue

        B = output.scan_slice_emb.shape[0]
        scan_slice_emb = output.scan_slice_emb
        scan_valid_depth_mask = output.scan_valid_depth_mask
        snippet_emb = output.snippet_emb
        slice_target_depth_mask = output.slice_target_depth_mask
        slice_batch_idx_valid = output.slice_batch_idx_valid

        S = snippet_emb.shape[0]
        if S == 0:
            scan_offset += B
            continue

        slice_emb = scan_slice_emb[slice_batch_idx_valid]
        valid_mask = scan_valid_depth_mask[slice_batch_idx_valid]

        slice_emb_n = F.normalize(slice_emb, dim=-1)
        snippet_emb_n = F.normalize(snippet_emb, dim=-1)
        logits = torch.einsum("sde,se->sd", slice_emb_n, snippet_emb_n)
        logits = logits.masked_fill(~valid_mask, float("-inf"))

        pred_depth = logits.argmax(dim=-1)
        target_depth = slice_target_depth_mask.float().argmax(dim=-1)

        all_pred_depths.append(pred_depth.cpu().numpy())
        all_target_depths.append(target_depth.cpu().numpy())
        all_scan_idx.append(slice_batch_idx_valid.cpu().numpy() + scan_offset)

        valid_mask_np = valid_mask.cpu().numpy()
        for s in range(S):
            valid_idx = np.where(valid_mask_np[s])[0]
            all_valid_min.append(valid_idx[0])
            all_valid_max.append(valid_idx[-1])

        total_snippets += S
        scan_offset += B
    pbar.close()

    assert total_snippets > 0, "No valid snippets found for localization bootstrap"

    pred = np.concatenate(all_pred_depths)
    target = np.concatenate(all_target_depths)
    scan_idx = np.concatenate(all_scan_idx)
    valid_min = np.array(all_valid_min)
    valid_max = np.array(all_valid_max)
    n_valid = valid_max - valid_min + 1

    standard_metrics = _loc_compute_metrics(pred, target)
    scan_widths_mm = n_valid.astype(np.float64) * MM_PER_DEPTH_POSITION
    standard_metrics["scan_width_mean_mm"] = float(scan_widths_mm.mean())
    standard_metrics["scan_width_median_mm"] = float(np.median(scan_widths_mm))
    standard_metrics["scan_width_std_mm"] = float(scan_widths_mm.std())
    perscan_metrics = _loc_compute_perscan_metrics(pred, target, scan_idx, n_valid)
    standard_metrics.update(perscan_metrics)

    bootstrap_metrics = bootstrap_localization(
        pred, target, n_bootstrap=n_bootstrap, ci=ci, seed=seed
    )

    return standard_metrics, bootstrap_metrics
