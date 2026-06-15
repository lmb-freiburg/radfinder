from copy import deepcopy
from timeit import default_timer
from typing import Any

import numpy as np
import torch
from monai.data import DataLoader
from radfinder.models.vision_language import SigLIP
from radfinder.utils.logging_utils import log_info

from packg.tqdmext import tqdm_max_ncols
from typedparser.objects import repr_value
from visiontext.distutils import is_main_process


def run_retrieval(
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    device: str = "cuda",
    verbose: bool = False,
) -> dict[str, float]:
    """
    Run retrieval evaluation and print metrics.

    Args:
        model: SigLIP model
        dataloader: DataLoader for evaluation
        dataset: Dataset being evaluated
        device: Device to run on

    Returns:
        Dictionary of text-to-image retrieval metrics (and optionally val_loss_nonaccum)
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
        if i < 2 and verbose:
            for k, v in sorted(batch.items(), key=lambda x: x[0]):
                pbar.write(f"Batch:  {k}: {repr_value(v, depth=1, key=k)}")
        pbar.update(1)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if i < 2 and verbose:
            for k, v in sorted(batch.items(), key=lambda x: x[0]):
                pbar.write(f"Model input:  {k}: {repr_value(v, depth=1, key=k)}")
        output = model(batch)
        if i < 2 and verbose:
            for k, v in sorted(output.items(), key=lambda x: x[0]):
                pbar.write(f"Model output:  {k}: {repr_value(v, depth=1, key=k)}")
        image_embeddings = output.image_embeddings
        text_embeddings = output.text_embeddings
        assert image_embeddings is not None and text_embeddings is not None

        # Optionally compute loss
        if model.criterion is not None:
            loss = model.criterion(image_embeddings, text_embeddings)
            total_loss += loss.detach().item()
            num_batches += 1

        all_image_embeddings.append(image_embeddings.cpu().numpy())
        all_text_embeddings.append(text_embeddings.cpu().numpy())
    pbar.close()

    # Concatenate all embeddings
    all_image_embeddings = np.concatenate(all_image_embeddings, axis=0).astype(np.float32)
    all_text_embeddings = np.concatenate(all_text_embeddings, axis=0).astype(np.float32)
    if verbose:
        log_info(f"Final embeddings: {all_image_embeddings.shape=}, {all_text_embeddings.shape=}")

    # Normalize embeddings
    img_norms = np.linalg.norm(all_image_embeddings, axis=1, keepdims=True)
    txt_norms = np.linalg.norm(all_text_embeddings, axis=1, keepdims=True)
    eps = 1e-8
    num_img_clipped = np.sum(img_norms < eps)
    num_txt_clipped = np.sum(txt_norms < eps)
    all_image_embeddings /= np.clip(img_norms, eps, None)
    all_text_embeddings /= np.clip(txt_norms, eps, None)

    # Compute similarities
    similarities = all_image_embeddings @ all_text_embeddings.T
    dot = torch.from_numpy(similarities)

    # Compute metrics for both directions
    metrics_i2t = compute_retrieval_cosine_r100(dot)[0]
    metrics_t2i = compute_retrieval_cosine_r100(dot.T)[0]
    metrics = deepcopy(metrics_t2i)
    if verbose:
        log_info("#################### text to image ####################")
        for r in (1, 5, 10, 50, 100):
            log_info(f"r{r:>3}: {metrics.pop(f'r{r}') * 100:6.2f}")
        for k, v in metrics.items():
            log_info(f"{k:>6}: {v:6.2f}" if isinstance(v, float) else f"{k:>6}: {v}")

    # Add validation loss if computed
    loss_nonaccum = None
    if model.criterion is not None and num_batches > 0:
        loss_nonaccum = total_loss / num_batches
        if verbose:
            log_info(f"val_loss_nonaccum: {loss_nonaccum:.4f}")

    # rename all metrics except n (number of datapoints) to t2i_{metric}
    all_metrics = {(f"t2i_{k}" if k != "n" else "n"): v for k, v in metrics_t2i.items()}
    all_metrics["loss_nonaccum"] = loss_nonaccum
    return all_metrics


def compute_retrieval_cosine_r100(dot: torch.Tensor) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Args:
        dot: cosine similarity computed as image @ text.T with shape (N, N)

    Returns:
        dictionary of metrics,
        dictionary of other values  (top1 pred index for each row, rank for each row)
    """
    n = len(dot)
    ranks = torch.empty(n)
    top1 = torch.empty(n)

    t1 = default_timer()

    # loop rows
    for index in range(n):
        # sort columns by highest similarity descending
        inds = torch.argsort(dot[index], descending=True)
        # the label (correct pair) is also "index". get rank of this correct embedding
        where = torch.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank

        # to save the top1 result:
        top1[index] = inds[0]

    # compute retrieval metrics
    r1 = len(torch.where(ranks < 1)[0]) / len(ranks)
    r5 = len(torch.where(ranks < 5)[0]) / len(ranks)
    r10 = len(torch.where(ranks < 10)[0]) / len(ranks)
    r50 = len(torch.where(ranks < 50)[0]) / len(ranks)
    r100 = len(torch.where(ranks < 100)[0]) / len(ranks)
    medr = (torch.floor(torch.median(ranks)) + 1).item()
    meanr = (ranks.mean() + 1).item()
    report_dict = {
        "r1": r1,
        "r5": r5,
        "r10": r10,
        "r50": r50,
        "r100": r100,
        "medr": medr,
        "meanr": meanr,
        "n": n,
    }
    other = {
        "top1": top1,
        "ranks": ranks,
    }
    return report_dict, other
