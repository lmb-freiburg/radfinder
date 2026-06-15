from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.data import DataLoader
from radfinder.models.vision_language import SigLIP
from radfinder.tasks.binary_zs_ctrate_task import load_binary_labels
from radfinder.utils.logging_utils import log_info

from packg.tqdmext import tqdm_max_ncols
from visiontext.distutils import is_main_process


def run_volume_retrieval(
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    device: str = "cuda",
    dataset_name: str = "ctrate",
    model_config: dict | None = None,
    verbose: bool = False,
) -> dict[str, float]:
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

    # Normalize
    norms = np.linalg.norm(all_image_embeddings, axis=1, keepdims=True)
    all_image_embeddings /= np.clip(norms, 1e-8, None)

    # Load labels
    labels = load_binary_labels(dataset_name, dataset)
    assert len(labels) == len(
        all_image_embeddings
    ), f"{len(labels)=} != {len(all_image_embeddings)=}"

    # Filter to abnormal volumes only
    abnormal_mask = labels.sum(axis=1) > 0
    n_total = len(labels)
    embeddings = all_image_embeddings[abnormal_mask]
    labels_abnormal = labels[abnormal_mask]
    n_abnormal = len(embeddings)
    log_info(f"Volume retrieval: {n_abnormal}/{n_total} abnormal volumes")

    # Cosine similarity, exclude self
    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, -np.inf)

    metrics = {
        "vol_n_total": n_total,
        "vol_n_abnormal": n_abnormal,
    }
    for k in (5, 10, 50):
        metrics[f"vol_map{k}"] = compute_map_at_k(similarities, labels_abnormal, k)

    if verbose:
        for key, val in metrics.items():
            log_info(f"  {key}: {val}")

    return metrics


def compute_map_at_k(similarities: np.ndarray, labels: np.ndarray, k: int) -> float:
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
    return float(np.mean(ap_scores))
