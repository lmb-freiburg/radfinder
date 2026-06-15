"""
Localization evaluation task.

For each snippet (slice-level text) in the dataset, the model predicts which depth
position it belongs to.  The metric is L1 error in millimetres between the predicted
and ground-truth depth index (one depth position = 12 mm).
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from radfinder.models.vision_language import SigLIP
from radfinder.paths import get_medv_data_dir
from radfinder.utils.logging_utils import log_info

from packg.tqdmext import tqdm_max_ncols
from visiontext.distutils import is_main_process

# Each depth index in D2 space corresponds to 12 mm
# (pixdim_z=3.0 mm * half-patch-embed patch_size_z=4 voxels = 12 mm)
MM_PER_DEPTH_POSITION = 12.0


def filter_one_slice_per_scan(dataset) -> None:
    """
    Restrict each datapoint to at most one slice (the lexicographically first).

    Localization evaluates a single (scan, slice, snippet) triple per datapoint,
    so multi-slice records are reduced to a single deterministic slice and
    snippet-less records are dropped.
    """
    for d in dataset.data:
        slices = d.get("slices")
        if slices and len(slices) > 1:
            first_key = sorted(slices.keys())[0]
            d["slices"] = {first_key: slices[first_key]}
    dataset.data = [d for d in dataset.data if d.get("slices") and len(d["slices"]) > 0]
    log_info(f"Filtered dataset to <=1 slice per datapoint. New length: {len(dataset.data)}")


def run_localization(
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    device: str = "cuda",
    verbose: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Run localization evaluation: predict depth index for each snippet.

    For every valid snippet the model produces per-depth image features
    (scan_slice_emb) and a snippet text embedding (snippet_emb).  The predicted
    depth is the argmax of cosine similarity; the target depth comes from
    slice_target_depth_mask.

    Returns:
        Tuple of (metrics, per-snippet details).
    """
    collected = _collect_localization_outputs(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
    )
    pred = collected["pred"]
    target = collected["target"]
    scan_idx = collected["scan_idx"]
    valid_min = collected["valid_min"]
    valid_max = collected["valid_max"]
    n_valid = valid_max - valid_min + 1  # number of valid depth positions per snippet

    metrics = _compute_metrics(pred, target)

    # Scan width statistics (in mm)
    scan_widths_mm = n_valid.astype(np.float64) * MM_PER_DEPTH_POSITION
    metrics["scan_width_mean_mm"] = float(scan_widths_mm.mean())
    metrics["scan_width_median_mm"] = float(np.median(scan_widths_mm))
    metrics["scan_width_std_mm"] = float(scan_widths_mm.std())

    # Per-scan averaged metrics (removes scan-length weighting bias)
    perscan_metrics = _compute_perscan_metrics(pred, target, scan_idx, n_valid)
    metrics.update(perscan_metrics)

    # Baselines (only computed for verbose printing, not included in returned metrics)
    baseline_random = _baseline_random(target, valid_min, valid_max, seed=42)
    baseline_middle = _baseline_middle(target, valid_min, valid_max)

    if verbose:
        log_info("#################### localization ####################")
        log_info(f"  snippets evaluated: {metrics['n_snippets']}")
        log_info(
            f"  scan depth: mean {metrics['scan_width_mean_mm']:.1f} mm,"
            f" median {metrics['scan_width_median_mm']:.1f} mm,"
            f" std {metrics['scan_width_std_mm']:.1f} mm"
        )
        _print_metrics("model", metrics)
        _print_metrics("random slice", baseline_random)
        _print_metrics("middle slice", baseline_middle)
        _print_perscan_stats(pred, target, scan_idx, n_valid)

    details = {
        "schema_version": 1,
        "n_dataset_scans": len(dataset),
        "n_snippets": int(metrics["n_snippets"]),
        "n_scans_with_snippets": int(metrics["n_scans_with_snippets"]),
        "n_total_slices": collected["n_total_slices"],
        "n_invalid_slices": collected["n_invalid_slices"],
        "rows": collected["rows"],
    }
    log_info(
        f"Collected {details['n_snippets']} valid snippets from "
        f"{details['n_scans_with_snippets']} unique scans."
    )
    return metrics, details


def _collect_localization_outputs(
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    device: str,
) -> dict[str, Any]:
    model = model.to(device)
    model.eval()
    log_info(f"Evaluating localization on {len(dataset)=}, {device=}")

    all_pred_depths: list[np.ndarray] = []
    all_target_depths: list[np.ndarray] = []
    all_valid_min: list[int] = []
    all_valid_max: list[int] = []
    all_scan_idx: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    total_snippets = 0
    total_slices = 0
    total_invalid_slices = 0
    scan_offset = 0

    pbar = tqdm_max_ncols(
        total=len(dataloader),
        desc="Evaluating localization",
        smoothing=0,
        disable=not is_main_process(),
    )
    for batch in dataloader:
        pbar.update(1)
        valid_slices = batch.get("valid_slices")
        assert valid_slices is not None, "Localization task requires valid_slices in the batch"
        valid_slice_rows = _get_valid_batch_slice_rows(batch)
        total_slices += len(valid_slices)
        total_invalid_slices += len(valid_slices) - len(valid_slice_rows)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model.forward_localization(batch)

        if output.scan_slice_emb is None:
            raise RuntimeError(
                "Localization forward did not return scan_slice_emb. "
                "The dataloader must provide image_backbone_patch_axis2 and the model forward "
                "must compute per-depth axis2 embeddings for localization evaluation."
            )
        if output.snippet_emb is None:
            raise RuntimeError(
                "Localization forward did not return snippet_emb. "
                "Check that the batch contains slices, valid_slices, snippet_input_ids, "
                "and snippet_attention_mask."
            )

        B = output.scan_slice_emb.shape[0]
        scan_slice_emb = output.scan_slice_emb  # (B, D2, E)
        scan_valid_depth_mask = output.scan_valid_depth_mask  # (B, D2)
        snippet_emb = output.snippet_emb  # (S, E)
        slice_target_depth_mask = output.slice_target_depth_mask  # (S, D2)
        slice_batch_idx_valid = output.slice_batch_idx_valid  # (S,)

        S = snippet_emb.shape[0]
        if S == 0:
            scan_offset += B
            continue
        assert len(valid_slice_rows) == S, (
            f"Mismatch between valid slice metadata and model outputs: "
            f"{len(valid_slice_rows)=} vs {S=}"
        )
        assert (
            slice_target_depth_mask is not None
        ), "Localization task requires slice_target_depth_mask"

        slice_emb = scan_slice_emb[slice_batch_idx_valid]  # (S, D2, E)
        valid_mask = scan_valid_depth_mask[slice_batch_idx_valid]  # (S, D2)

        slice_emb_n = F.normalize(slice_emb, dim=-1)
        snippet_emb_n = F.normalize(snippet_emb, dim=-1)
        logits = torch.einsum("sde,se->sd", slice_emb_n, snippet_emb_n)  # (S, D2)
        masked_logits = logits.masked_fill(~valid_mask, float("-inf"))

        pred_depth = masked_logits.argmax(dim=-1)  # (S,)
        target_depth = slice_target_depth_mask.float().argmax(dim=-1)  # (S,)
        pred_score = masked_logits.gather(1, pred_depth[:, None]).squeeze(1)
        target_score = masked_logits.gather(1, target_depth[:, None]).squeeze(1)

        pred_np = pred_depth.cpu().numpy()
        target_np = target_depth.cpu().numpy()
        scan_idx_np = slice_batch_idx_valid.cpu().numpy() + scan_offset
        valid_mask_np = valid_mask.cpu().numpy()
        logits_np = logits.float().cpu().numpy()
        pred_score_np = pred_score.float().cpu().numpy()
        target_score_np = target_score.float().cpu().numpy()

        all_pred_depths.append(pred_np)
        all_target_depths.append(target_np)
        all_scan_idx.append(scan_idx_np)

        for s in range(S):
            valid_idx = np.where(valid_mask_np[s])[0]
            assert len(valid_idx) > 0, f"Snippet {s} has no valid depth positions"
            all_valid_min.append(int(valid_idx[0]))
            all_valid_max.append(int(valid_idx[-1]))

            row = dict(valid_slice_rows[s])
            row.update(
                {
                    "scan_index": int(scan_idx_np[s]),
                    "pred_depth_idx": int(pred_np[s]),
                    "target_depth_idx": int(target_np[s]),
                    "pred_score": float(pred_score_np[s]),
                    "target_score": float(target_score_np[s]),
                    "abs_error_positions": int(abs(pred_np[s] - target_np[s])),
                    "abs_error_mm": float(abs(pred_np[s] - target_np[s]) * MM_PER_DEPTH_POSITION),
                    "valid_depth_min": int(valid_idx[0]),
                    "valid_depth_max": int(valid_idx[-1]),
                    "n_valid_depth": int(len(valid_idx)),
                    "valid_depth_mask": valid_mask_np[s].tolist(),
                    "cosine_logits": logits_np[s].tolist(),
                }
            )
            rows.append(row)

        total_snippets += S
        scan_offset += B
    pbar.close()

    assert total_snippets > 0, "No valid snippets found for localization evaluation"

    return {
        "pred": np.concatenate(all_pred_depths),
        "target": np.concatenate(all_target_depths),
        "scan_idx": np.concatenate(all_scan_idx),
        "valid_min": np.array(all_valid_min, dtype=np.int64),
        "valid_max": np.array(all_valid_max, dtype=np.int64),
        "n_total_slices": total_slices,
        "n_invalid_slices": total_invalid_slices,
        "rows": rows,
    }


def _get_valid_batch_slice_rows(batch: dict[str, Any]) -> list[dict[str, Any]]:
    valid_slices = batch.get("valid_slices")
    if valid_slices is None:
        raise ValueError("Localization task requires valid_slices in the batch")
    if not isinstance(valid_slices, torch.Tensor):
        raise TypeError(f"Expected valid_slices to be torch.Tensor, got {type(valid_slices)}")
    if "slices" not in batch:
        raise ValueError("Localization task requires slices in the batch")

    filenames = batch.get("filename")
    scan_keys = batch.get("scan_key")
    if filenames is None or scan_keys is None:
        raise ValueError("Localization task requires filename and scan_key in the batch")

    valid_slice_rows = []
    flat_idx = 0
    valid_slices_list = valid_slices.tolist()
    for batch_idx, slices in enumerate(batch["slices"]):
        filename_rel = Path(filenames[batch_idx]).relative_to(get_medv_data_dir()).as_posix()
        for slice_key, slice_data in slices.items():
            assert flat_idx < len(valid_slices_list), f"{flat_idx=} out of bounds for valid_slices"
            if valid_slices_list[flat_idx]:
                valid_slice_rows.append(
                    {
                        "scan_key": scan_keys[batch_idx],
                        "filename_rel": filename_rel,
                        "slice_key": slice_key,
                        "snippet_text": slice_data["snippet"],
                    }
                )
            flat_idx += 1
    assert flat_idx == len(valid_slices_list), f"{flat_idx=} != {len(valid_slices_list)=}"
    return valid_slice_rows


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    errors_positions = np.abs(pred.astype(np.float64) - target.astype(np.float64))
    errors_mm = errors_positions * MM_PER_DEPTH_POSITION
    return {
        "loc_mae_mm": float(errors_mm.mean()),
        "loc_median_mm": float(np.median(errors_mm)),
        "loc_mae_positions": float(errors_positions.mean()),
        "loc_acc_exact": float((errors_positions == 0).mean()),
        "loc_acc_within_12mm": float((errors_positions <= 1).mean()),
        "loc_acc_within_24mm": float((errors_positions <= 2).mean()),
        "n_snippets": len(pred),
    }


def _compute_perscan_metrics(
    pred: np.ndarray, target: np.ndarray, scan_idx: np.ndarray, n_valid: np.ndarray
) -> dict[str, float]:
    """Compute per-scan averaged metrics to remove scan-length weighting bias."""
    errors_mm = np.abs(pred.astype(np.float64) - target.astype(np.float64)) * MM_PER_DEPTH_POSITION
    unique_scans = np.unique(scan_idx)
    scan_maes = []
    scan_acc_exact = []
    scan_acc_24mm = []
    for s in unique_scans:
        mask = scan_idx == s
        scan_maes.append(errors_mm[mask].mean())
        errors_pos = np.abs(pred[mask].astype(np.float64) - target[mask].astype(np.float64))
        scan_acc_exact.append((errors_pos == 0).mean())
        scan_acc_24mm.append((errors_pos <= 2).mean())
    return {
        "loc_perscan_mae_mm": float(np.mean(scan_maes)),
        "loc_perscan_acc_exact": float(np.mean(scan_acc_exact)),
        "loc_perscan_acc_within_24mm": float(np.mean(scan_acc_24mm)),
        "n_scans_with_snippets": len(unique_scans),
    }


def _print_perscan_stats(
    pred: np.ndarray, target: np.ndarray, scan_idx: np.ndarray, n_valid: np.ndarray
):
    """Print diagnostic stats about snippet distribution across scans."""
    unique_scans, counts = np.unique(scan_idx, return_counts=True)
    errors_mm = np.abs(pred.astype(np.float64) - target.astype(np.float64)) * MM_PER_DEPTH_POSITION
    scan_depths_mm = n_valid.astype(np.float64) * MM_PER_DEPTH_POSITION

    log_info("  --- per-scan diagnostics ---")
    log_info(f"  scans with snippets: {len(unique_scans)}")
    log_info(
        f"  snippets/scan: mean {counts.mean():.1f}, median {np.median(counts):.0f},"
        f" max {counts.max()}, distribution: "
        + ", ".join(f"{n}s:{(counts==n).sum()}" for n in sorted(set(counts))[:6])
    )

    # MAE stratified by scan depth (short vs long scans)
    # use per-snippet scan depth (all snippets from same scan have same depth)
    cutoff = 300
    short_mask = scan_depths_mm <= cutoff
    long_mask = ~short_mask
    if short_mask.any() and long_mask.any():
        short_mae = errors_mm[short_mask].mean()
        long_mae = errors_mm[long_mask].mean()
        short_depth = scan_depths_mm[short_mask].mean()
        long_depth = scan_depths_mm[long_mask].mean()
        log_info(
            f"  short scans (depth<={cutoff:.0f}mm): {short_mask.sum()} snippets,"
            f" mean depth {short_depth:.0f}mm, MAE {short_mae:.1f}mm"
        )
        log_info(
            f"  long scans  (depth>{cutoff:.0f}mm):  {long_mask.sum()} snippets,"
            f" mean depth {long_depth:.0f}mm, MAE {long_mae:.1f}mm"
        )

    # Per-scan averaged MAE
    scan_maes = []
    for s in unique_scans:
        mask = scan_idx == s
        scan_maes.append(errors_mm[mask].mean())
    scan_maes = np.array(scan_maes)
    log_info(f"  per-scan MAE: mean {scan_maes.mean():.1f}mm, median {np.median(scan_maes):.1f}mm")

    # First snippet vs subsequent snippets within each scan
    is_first = np.zeros(len(pred), dtype=bool)
    snippet_order = np.zeros(len(pred), dtype=int)  # 0-indexed order within scan
    scan_snippet_count = {}  # total snippets per scan
    seen_scans = {}
    for i, s in enumerate(scan_idx):
        if s not in seen_scans:
            is_first[i] = True
            seen_scans[s] = 0
        snippet_order[i] = seen_scans[s]
        seen_scans[s] += 1
    for i, s in enumerate(scan_idx):
        scan_snippet_count[s] = seen_scans[s]
    is_subsequent = ~is_first
    is_multi_scan = np.array([seen_scans[s] > 1 for s in scan_idx])
    n_first = is_first.sum()
    n_subseq = is_subsequent.sum()
    if n_first > 0:
        first_mae = errors_mm[is_first].mean()
        log_info(
            f"  1st snippet/scan ({n_first}): MAE {first_mae:.1f}mm,"
            f" mean depth {scan_depths_mm[is_first].mean():.0f}mm"
        )
    if n_subseq > 0:
        subseq_mae = errors_mm[is_subsequent].mean()
        log_info(
            f"  2nd+ snippet/scan ({n_subseq}): MAE {subseq_mae:.1f}mm,"
            f" mean depth {scan_depths_mm[is_subsequent].mean():.0f}mm"
        )
    # same comparison but restricted to multi-snippet scans only
    is_first_multi = is_first & is_multi_scan
    is_subseq_multi = is_subsequent  # already only from multi-snippet scans
    if is_first_multi.any() and is_subseq_multi.any():
        log_info("  (multi-snippet scans only)")
        log_info(
            f"    1st snippet ({is_first_multi.sum()}): MAE {errors_mm[is_first_multi].mean():.1f}mm,"
            f" mean depth {scan_depths_mm[is_first_multi].mean():.0f}mm"
        )
        log_info(
            f"    2nd+ snippet ({is_subseq_multi.sum()}): MAE {errors_mm[is_subseq_multi].mean():.1f}mm,"
            f" mean depth {scan_depths_mm[is_subseq_multi].mean():.0f}mm"
        )


def _baseline_random(
    target: np.ndarray, valid_min: np.ndarray, valid_max: np.ndarray, seed: int = 42
) -> dict[str, float]:
    """Random baseline: pick a uniformly random valid depth position per snippet."""
    rng = np.random.RandomState(seed)
    pred = np.array([rng.randint(lo, hi + 1) for lo, hi in zip(valid_min, valid_max)])
    return _compute_metrics(pred, target)


def _baseline_middle(
    target: np.ndarray, valid_min: np.ndarray, valid_max: np.ndarray
) -> dict[str, float]:
    """Middle baseline: pick the middle valid depth position per snippet."""
    pred = (valid_min + valid_max) // 2
    return _compute_metrics(pred, target)


def _print_metrics(name: str, m: dict[str, float]):
    log_info(
        f"  {name:>16s}:  MAE {m['loc_mae_mm']:6.1f} mm ({m['loc_mae_positions']:.2f} pos)"
        f"  median {m['loc_median_mm']:6.1f} mm"
        f"  exact {m['loc_acc_exact']*100:5.1f}%"
        f"  <=12mm {m['loc_acc_within_12mm']*100:5.1f}%"
        f"  <=24mm {m['loc_acc_within_24mm']*100:5.1f}%"
    )
