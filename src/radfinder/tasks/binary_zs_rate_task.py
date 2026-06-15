"""
Binary zero-shot classification task for RaTE 319 questions.

Evaluates binary classification on RaTE findings (319 questions) using CLIP-style
positive/negative text prompts loaded from YAML files, with per-question AUROC metric.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from monai.data import DataLoader
from radfinder.data.ct_rate import extract_report_key
from radfinder.data.inspect import InspectDataset
from radfinder.data.merlin import MerlinDataset
from radfinder.data.rad_chestct import RadChestCTDataset
from radfinder.models.modeling import last_token_pool
from radfinder.models.vision_language import SigLIP
from radfinder.paths import PROMPTS_DIR, RATE_CONFIG_DIR, get_medv_data_dir
from radfinder.tasks.question_vector_loader import load_question_vectors
from radfinder.utils.logging_utils import log_error, log_info, log_warning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from transformers import Qwen2TokenizerFast

from packg.tqdmext import tqdm_max_ncols
from visiontext.distutils import is_main_process


def run_binary_zs_rate(
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    device: str = "cuda",
    dataset_name: str = "ctrate",
    model_config: dict | None = None,
    modality: str = "abdomen_chest",
    split: str = "val",
    verbose: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Run binary zero-shot classification on RaTE 319 questions using CLIP prompts.

    Args:
        model: SigLIP model (unwrapped).
        dataloader: DataLoader for evaluation.
        dataset: Dataset being evaluated.
        device: Device to run on.
        dataset_name: Name of dataset (for label loading).
        model_config: Model config dict (needed for tokenizer).
        modality: Modality string for question vector selection.
        split: Dataset split.
        verbose: Whether to print verbose output.

    Returns:
        Tuple of (main_metrics, aux_metrics) where main_metrics has mean_auroc/precision/f1/acc
        and aux_metrics has per-question and per-category details.
    """
    model = model.to(device)
    model.eval()

    # Resolve question map path
    language = "en"
    question_map_csv = (
        RATE_CONFIG_DIR
        / f"modalities_{language}"
        / "question_maps"
        / f"question_map_{modality}.csv"
    )

    # Load prompts
    question_ids, positive_prompts, negative_prompts, categories = load_prompts_from_yaml(
        question_map_csv
    )
    num_questions = len(question_ids)
    log_info(f"Loaded {num_questions} questions with CLIP prompts from {PROMPTS_DIR}")

    # Extract image embeddings
    log_info(f"Evaluating binary ZS RaTE on {len(dataset)} samples, {device=}")
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
        output = model(batch)
        all_image_embeddings.append(output.image_embeddings_secondary.cpu().float().numpy())
    pbar.close()

    all_image_embeddings = np.concatenate(all_image_embeddings, axis=0).astype(np.float32)
    if verbose:
        log_info(f"Image embeddings shape: {all_image_embeddings.shape}")

    # Encode text prompts
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_config["text_tokenizer"])
    pos_embeddings, neg_embeddings = encode_rate_prompts(
        model.backbone_text,
        model.projection_text,
        tokenizer,
        positive_prompts,
        negative_prompts,
        device,
        normalize=False,  # normalization handled below with image embeddings
    )

    # Interleave pos/neg: [pos_q1, neg_q1, pos_q2, neg_q2, ...]
    text_latents = np.empty((num_questions * 2, pos_embeddings.shape[1]), dtype=np.float32)
    text_latents[0::2] = pos_embeddings
    text_latents[1::2] = neg_embeddings

    # Compute logits with temperature and bias (same as binary_zs_ctrate_task)
    image_emb = torch.from_numpy(all_image_embeddings).cpu()
    text_lat = torch.from_numpy(text_latents).cpu()
    b = len(image_emb)

    if model.criterion.normalize:
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        text_lat = F.normalize(text_lat, p=2, dim=-1)
    raw_logits = image_emb @ text_lat.t()
    temp = model.criterion.t.to(image_emb.device)
    bias = model.criterion.b.to(image_emb.device)
    logits = raw_logits * torch.exp(-temp) + bias  # (B, num_q*2)
    logits = logits.reshape((b, num_questions, 2))  # (B, num_q, 2)
    sm_logits = logits.softmax(dim=-1)  # (B, num_q, 2)
    predicted = sm_logits[:, :, 0].detach().cpu().numpy()  # (B, num_q) P(positive)

    if verbose:
        pos_logits = logits[:, :, 0].detach().cpu().numpy()
        log_info("[Binary ZS RaTE] Positive class logits before softmax:")
        log_info(f"  mean: {pos_logits.mean():.5f}, std: {pos_logits.std():.5f}")
        sm_pos = sm_logits[:, :, 0].detach().cpu().numpy()
        log_info("[Binary ZS RaTE] Positive class logits after softmax:")
        log_info(f"  mean: {sm_pos.mean():.5f}, std: {sm_pos.std():.5f}")

    # Load RaTE labels and filter to samples with labels
    labels_all, label_qids, valid_mask = load_rate_labels(
        dataset_name,
        dataset,
        split,
        modality,
        language,
    )
    assert (
        labels_all.shape[1] == num_questions
    ), f"Label columns {labels_all.shape[1]} != prompt questions {num_questions}"

    # Filter to only samples that have labels
    labels = labels_all[valid_mask]
    predicted = predicted[valid_mask]
    n_labeled = len(labels)
    log_info(f"Evaluating on {n_labeled} labeled samples (out of {len(valid_mask)} total)")

    if n_labeled == 0:
        log_error("No labeled samples found. Cannot compute metrics.")
        return {
            "mean_auroc": 0.0,
            "mean_precision": 0.0,
            "mean_precision_w": 0.0,
            "mean_f1": 0.0,
            "mean_f1_w": 0.0,
            "mean_acc": 0.0,
            "mean_acc_raw": 0.0,
            "n_samples": 0,
            "n_questions_evaluated": 0,
            "n_questions_skipped": 0,
        }, {}

    # Label space analysis
    pos_counts = labels.sum(axis=0)  # (num_q,) number of positives per question
    neg_counts = n_labeled - pos_counts
    prevalence = pos_counts / n_labeled

    n_all_neg = int((pos_counts == 0).sum())
    n_all_pos = int((neg_counts == 0).sum())
    n_rare = int((pos_counts < 5).sum())  # fewer than 5 positives
    log_info(f"Label space analysis ({n_labeled} samples, {num_questions} questions):")
    log_info(f"  All-negative (no positives):  {n_all_neg} questions")
    log_info(f"  All-positive (no negatives):  {n_all_pos} questions")
    log_info(f"  Rare (<5 positives):          {n_rare} questions")
    log_info(f"  Evaluable (both classes):     {num_questions - n_all_neg - n_all_pos} questions")
    log_info(f"  Mean prevalence:              {prevalence.mean():.4f}")
    log_info(f"  Median prevalence:            {np.median(prevalence):.4f}")
    log_info(
        f"  Total positive labels:        {int(pos_counts.sum())} / {n_labeled * num_questions}"
    )

    # Compute per-question metrics
    METRIC_NAMES = ["auroc", "prec", "precw", "f1", "f1w", "acc", "accr"]
    aux_metrics = {}
    per_q: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    cat_vals: dict[str, dict[str, list[float]]] = {}
    n_skipped = 0

    for i, (qid, cat) in enumerate(zip(question_ids, categories)):
        y_true = labels[:, i]
        y_pred = predicted[:, i]

        aux_metrics[f"{qid}_{cat}_pos_count"] = int(pos_counts[i])
        aux_metrics[f"{qid}_{cat}_prevalence"] = float(prevalence[i])

        # Skip questions with only one class present
        if len(np.unique(y_true)) < 2:
            n_skipped += 1
            continue

        auroc = roc_auc_score(y_true, y_pred)
        aux_metrics[f"{qid}_{cat}_auroc"] = auroc

        # Threshold-based metrics
        y_pred_bin = (y_pred >= 0.5).astype(int)
        prec = precision_score(y_true, y_pred_bin, zero_division=0)
        prec_w = precision_score(y_true, y_pred_bin, average="weighted", zero_division=0)
        f1_bin = f1_score(y_true, y_pred_bin, zero_division=0)
        f1_w = f1_score(y_true, y_pred_bin, average="weighted", zero_division=0)
        acc = balanced_accuracy_score(y_true, y_pred_bin)
        acc_r = accuracy_score(y_true, y_pred_bin)

        vals = [auroc, prec, prec_w, f1_bin, f1_w, acc, acc_r]
        aux_metrics[f"{qid}_{cat}_precision"] = prec
        aux_metrics[f"{qid}_{cat}_precision_weighted"] = prec_w
        aux_metrics[f"{qid}_{cat}_f1"] = f1_bin
        aux_metrics[f"{qid}_{cat}_f1_weighted"] = f1_w
        aux_metrics[f"{qid}_{cat}_bal_acc"] = acc
        aux_metrics[f"{qid}_{cat}_acc"] = acc_r

        for m, v in zip(METRIC_NAMES, vals):
            per_q[m].append(v)

        # Collect for per-category average
        if cat not in cat_vals:
            cat_vals[cat] = {m: [] for m in METRIC_NAMES}
        for m, v in zip(METRIC_NAMES, vals):
            cat_vals[cat][m].append(v)

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    main_metrics = {
        "mean_auroc": _mean(per_q["auroc"]),
        "mean_precision": _mean(per_q["prec"]),
        "mean_precision_w": _mean(per_q["precw"]),
        "mean_f1": _mean(per_q["f1"]),
        "mean_f1_w": _mean(per_q["f1w"]),
        "mean_acc": _mean(per_q["acc"]),
        "mean_acc_raw": _mean(per_q["accr"]),
        "n_samples": n_labeled,
        "n_questions_evaluated": len(per_q["auroc"]),
        "n_questions_skipped": n_skipped,
    }

    # Per-category means (into aux)
    for cat in sorted(cat_vals.keys()):
        for m in METRIC_NAMES:
            aux_metrics[f"cat_{cat}_mean_{m}"] = _mean(cat_vals[cat][m])

    if verbose:
        log_info("Per-category AUROC:")
        for cat in sorted(cat_vals.keys()):
            n_q = len(cat_vals[cat]["auroc"])
            log_info(f"  {cat:<40s}: {_mean(cat_vals[cat]['auroc']):.4f} ({n_q} questions)")
        log_info(
            f"  {'Macro metrics':<40s}: ({len(per_q['auroc'])} questions, {n_skipped} skipped)"
        )
        for k, v in main_metrics.items():
            if isinstance(v, float):
                log_info(f"  {k:<40s}: {v:.4f}")

    return main_metrics, aux_metrics


def load_prompts_from_yaml(
    question_map_csv: str | Path,
) -> tuple[list[str], list[list[str]], list[list[str]], list[str]]:
    """
    Load CLIP prompts from YAML files and align to canonical question ordering.

    Args:
        question_map_csv: Path to question_map_abdomen_chest.csv defining q-ID ordering.

    Returns:
        Tuple of (question_ids, positive_prompts, negative_prompts, categories) where:
        - question_ids: list of qids (q0001..q0319)
        - positive_prompts: list of list[str], 3 positive prompts per question
        - negative_prompts: list of list[str], 3 negative prompts per question
        - categories: list of category names per question
    """
    question_map_csv = Path(question_map_csv)

    # Load question map for canonical ordering
    qmap = pd.read_csv(question_map_csv)
    assert {"qid", "category", "question"}.issubset(qmap.columns)

    # Build question text -> prompts lookup from all YAML files
    question_to_prompts: dict[str, dict] = {}
    for yaml_file in sorted(PROMPTS_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        for entry in data["prompts"]:
            q_text = entry["question"]
            question_to_prompts[q_text] = {
                "positive": entry["positive"],
                "negative": entry["negative"],
            }

    # Explicit corrections for typos/spelling mismatches between question_map CSV and YAMLs
    _CSV_TO_YAML_FIXES = {
        "debice": "device",  # q0244 typo
        "dilation": "dilatation",  # q0029, q0030 spelling
        "lymphomas": "lymphoma",  # q0098 plural
    }

    def _normalize_question(q_text: str) -> str:
        """Apply known CSV→YAML text corrections."""
        result = q_text
        for old, new in _CSV_TO_YAML_FIXES.items():
            result = result.replace(old, new)
        # Capitalize first letter (q0030 has lowercase "is")
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        return result

    # Align prompts to canonical q-ID ordering
    question_ids = []
    positive_prompts = []
    negative_prompts = []
    categories = []
    missing = []

    for _, row in qmap.iterrows():
        qid = row["qid"]
        q_text: str = row["question"]
        cat = row["category"]

        prompts = question_to_prompts.get(q_text) or question_to_prompts.get(
            _normalize_question(q_text)
        )
        if prompts is None:
            missing.append((qid, q_text))
            # Use fallback prompts based on question text
            positive_prompts.append(
                [
                    f"Yes. {q_text.replace('?', '.')}",
                    f"The answer is yes. {q_text.replace('?', '.')}",
                    f"Findings consistent with: {q_text.replace('?', '.').replace('Is there ', '').replace('Are there ', '')}",
                ]
            )
            negative_prompts.append(
                [
                    f"No. {q_text.replace('Is there', 'There is no').replace('Are there', 'There are no').replace('?', '.')}",
                    f"The answer is no. {q_text.replace('?', '.')}",
                    f"No findings of: {q_text.replace('?', '.').replace('Is there ', '').replace('Are there ', '')}",
                ]
            )
        else:
            positive_prompts.append(prompts["positive"])
            negative_prompts.append(prompts["negative"])

        question_ids.append(qid)
        categories.append(cat)

    if missing:
        log_warning(f"{len(missing)} questions missing from YAML prompts, using fallback:")
        for qid, q_text in missing[:5]:
            log_warning(f"  {qid}: {q_text}")
        if len(missing) > 5:
            log_warning(f"  ... and {len(missing) - 5} more")

    assert len(question_ids) == len(
        qmap
    ), f"Expected {len(qmap)} questions, got {len(question_ids)}"
    return question_ids, positive_prompts, negative_prompts, categories


@torch.inference_mode()
def encode_rate_prompts(
    text_backbone,
    text_projection,
    tokenizer,
    positive_prompts: list[list[str]],
    negative_prompts: list[list[str]],
    device: str,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode CLIP prompts into text embeddings, averaging variants per question.

    For each question, encodes 3 positive and 3 negative prompt variants,
    averages the embeddings within each group to get 1 pos + 1 neg vector.

    Args:
        text_backbone: Text encoder model.
        text_projection: Text projection head.
        tokenizer: Tokenizer for the text model.
        positive_prompts: List of 3 positive prompts per question.
        negative_prompts: List of 3 negative prompts per question.
        device: Device to use.
        normalize: Whether to L2-normalize before averaging.

    Returns:
        (pos_embeddings, neg_embeddings) each shape (num_questions, embed_dim)
    """
    num_questions = len(positive_prompts)

    # Collect all texts: [pos1_v1, pos1_v2, pos1_v3, pos2_v1, ..., neg1_v1, ...]
    all_texts = []
    for pos_list in positive_prompts:
        all_texts.extend(pos_list)
    for neg_list in negative_prompts:
        all_texts.extend(neg_list)

    # Encode all texts at once
    tokenizer_output = tokenizer(
        all_texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=4096,
    )
    input_ids = torch.tensor(tokenizer_output["input_ids"]).to(device)
    attention_mask = torch.tensor(tokenizer_output["attention_mask"]).to(device)

    text_embeddings = text_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    text_embeddings = last_token_pool(text_embeddings.last_hidden_state, attention_mask)
    text_embeddings = text_projection(text_embeddings)
    all_embs = text_embeddings.detach().cpu().float()  # (total_texts, embed_dim)

    # Split into positive and negative groups
    n_variants_pos = sum(len(p) for p in positive_prompts)
    pos_embs_flat = all_embs[:n_variants_pos]
    neg_embs_flat = all_embs[n_variants_pos:]

    # Average variants per question
    pos_averaged = []
    offset = 0
    for pos_list in positive_prompts:
        n = len(pos_list)
        chunk = pos_embs_flat[offset : offset + n]
        if normalize:
            chunk = F.normalize(chunk, p=2, dim=-1)
        # pos_averaged.append(chunk.mean(dim=0))
        pos_averaged.append(chunk[0])
        offset += n

    neg_averaged = []
    offset = 0
    for neg_list in negative_prompts:
        n = len(neg_list)
        chunk = neg_embs_flat[offset : offset + n]
        if normalize:
            chunk = F.normalize(chunk, p=2, dim=-1)
        neg_averaged.append(chunk.mean(dim=0))
        offset += n

    pos_embeddings = torch.stack(pos_averaged).numpy()  # (num_q, embed_dim)
    neg_embeddings = torch.stack(neg_averaged).numpy()  # (num_q, embed_dim)

    return pos_embeddings, neg_embeddings


def get_labeled_scan_keys(
    dataset_name: str,
    split: str,
    modality: str,
    language: str = "en",
) -> list[str] | None:
    """
    Return scan keys that have RaTE labels for the given modality.

    If the dataset does not require filtering (e.g. all scans have labels), returns None.
    """
    pass


def _map_scans_to_reports(
    dataset_name: str,
    dataset: Any,
    split: str,
) -> list[str]:
    """Map dataset sample order to report IDs."""
    if dataset_name == "ctrate":
        return [extract_report_key(item["image"]) for item in dataset.data]
    elif dataset_name == "merlin":
        return [
            MerlinDataset.get_datapoint_key_from_scan_path(item["image"]) for item in dataset.data
        ]
    elif dataset_name == "inspect":
        return [
            InspectDataset.get_datapoint_key_from_scan_path(item["image"]) for item in dataset.data
        ]
    elif dataset_name == "radchestct":
        return [
            RadChestCTDataset.get_datapoint_key_from_scan_path(item["image"])
            for item in dataset.data
        ]
    else:
        raise NotImplementedError(f"Dataset {dataset_name}")


def get_ctrate_labels_dir() -> Path:
    return get_medv_data_dir() / "public/CT-RATE/report_structuring/p0rate_en"


def load_rate_labels(
    dataset_name: str,
    dataset: Any,
    split: str,
    modality: str = "abdomen_chest",
    language: str = "en",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Load RaTE binary labels aligned to the dataloader sample order.

    Returns:
        (labels, question_ids, valid_mask) where:
        - labels: (N, num_questions) binary array (zeros for unmatched samples)
        - question_ids: list of qid strings
        - valid_mask: (N,) boolean array, True where labels exist
    """
    report_ids = _map_scans_to_reports(dataset_name, dataset, split)

    # Determine label source per dataset
    labels_dir = None
    file_prefix = None
    if dataset_name == "ctrate":
        labels_dir = get_ctrate_labels_dir()
        file_prefix = "questions_chest"
    elif dataset_name == "merlin":
        labels_dir = Path(get_medv_data_dir()) / "public/Merlin/report_structuring/p0rate_en"
        file_prefix = "questions_abdomen"
    elif dataset_name == "inspect":
        labels_dir = Path(get_medv_data_dir()) / "public/Inspect/report_structuring/p0rate_en"
        file_prefix = "questions_chest"

    qv_data = load_question_vectors(
        language=language,
        modality=modality,
        split=split,
        labels_dir=labels_dir,
        file_prefix=file_prefix,
    )
    report_id_to_idx = {str(rid): i for i, rid in enumerate(qv_data.report_ids)}
    qids = qv_data.qids
    n_questions = len(qids)
    labels = np.zeros((len(report_ids), n_questions), dtype=np.int64)
    valid_mask = np.zeros(len(report_ids), dtype=bool)
    for i, rid in enumerate(report_ids):
        if rid in report_id_to_idx:
            labels[i] = qv_data.labels[report_id_to_idx[rid]]
            valid_mask[i] = True

    n_valid = int(valid_mask.sum())
    log_info(f"Label alignment: {n_valid}/{len(report_ids)} samples have RaTE labels")

    return labels, qids, valid_mask
