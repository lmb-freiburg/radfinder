"""
Binary zero-shot classification task.

Evaluates binary classification on CT pathologies (18 classes) using zero-shot
text prompts and per-pathology AUROC metric. Metrics use Youden's J per-class
threshold (CheXzero protocol).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from radfinder.data.dataloader_train import DatasetNameC
from radfinder.data.prompt_rate_labels import (
    CTRATE18_TO_RATE_QIDS_FILE,
    PATHOLOGIES,
    TEMPLATES,
    load_ctrate18_prompts,
)
from radfinder.data.rad_chestct import (
    aggregate_radchestct_to_ctrate18,
    load_radchestct_labels,
    load_radchestct_to_ctrate18_mapping,
)
from radfinder.loader_utils import load_csv
from radfinder.models.modeling import last_token_pool
from radfinder.models.vision_language import SigLIP
from radfinder.paths import RADFINDER_REPO_DIR, RATE_CONFIG_DIR, get_medv_data_dir
from radfinder.tasks.binary_zs_rate_task import encode_rate_prompts, load_prompts_from_yaml
from radfinder.utils.logging_utils import log_info
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from transformers import Qwen2TokenizerFast

from packg.constclass import Const
from packg.tqdmext import tqdm_max_ncols
from visiontext.distutils import is_main_process


class PromptModeC(Const):
    T3 = "t3"
    MEAN7 = "mean7"
    RATE = "rate"


def run_binary_zs(
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    device: str = "cuda",
    dataset_name: str = "ctrate",
    model_config: dict | None = None,
    prompt_mode: str = PromptModeC.T3,
    verbose: bool = False,
    radchestct_label_mapping: str = "extended",
    eval_protocol: str = "default",
) -> tuple[dict[str, float], dict[str, float]]:
    """Run binary zero-shot classification and return metrics."""
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

    if verbose:
        log_info(
            f"[Binary ZS] predictions: mean={predictedall.mean():.5f}, std={predictedall.std():.5f}"
        )

    if eval_protocol == "radchestct_standard":
        # CT-CLIP protocol: merge Arterial (idx 1) + Coronary (idx 4) predictions,
        # then drop Coronary (idx 4) and Mosaic attenuation (idx 13)
        predictedall[:, 1] = np.maximum(predictedall[:, 1], predictedall[:, 4])
        keep = [i for i in range(len(PATHOLOGIES)) if i not in (4, 13)]
        predictedall = predictedall[:, keep]
        labels = labels[:, keep]
        pathologies_eval = [PATHOLOGIES[i] for i in keep]
        log_info(f"[radchestct_standard] Merged calcification, evaluating {len(keep)} classes")
        return evaluate_chexzero(predictedall, labels, pathologies=pathologies_eval)

    return evaluate_chexzero(predictedall, labels)


def _encode_text_latents(model, tokenizer, device, normalize, prompt_mode):
    """Encode text latents for t3 or mean7 prompt modes. Returns (n_path*2, dim) array."""
    if prompt_mode == PromptModeC.T3:
        template = ["There is {a}.", "There is no {a}."]
        latents = get_text_latents(
            model.backbone_text, model.projection_text, tokenizer, template, device
        )
        if normalize:
            latents = F.normalize(torch.from_numpy(latents), p=2, dim=-1).numpy()
        return latents
    elif prompt_mode == PromptModeC.MEAN7:
        all_latents = []
        for _name, (pos_tmpl, neg_tmpl) in TEMPLATES.items():
            template = [pos_tmpl, neg_tmpl]
            latents = get_text_latents(
                model.backbone_text, model.projection_text, tokenizer, template, device
            )
            if normalize:
                latents = F.normalize(torch.from_numpy(latents), p=2, dim=-1).numpy()
            all_latents.append(latents)
        stacked = np.stack(all_latents, axis=0)  # (7, n_path*2, dim)
        mean_lats = stacked.mean(axis=0)
        return F.normalize(torch.from_numpy(mean_lats), p=2, dim=-1).numpy()  # (n_path*2, dim)
    elif prompt_mode == PromptModeC.RATE:
        raise NotImplementedError(f"Prompt mode {prompt_mode} is already handled by _predict_rate")
    else:
        PromptModeC.verify_value(prompt_mode)


def _predict_rate(model, tokenizer, image_emb_np, device, normalize):
    """Compute (N, 18) predictions using RaTE chest prompts aggregated to CT-RATE pathologies."""
    question_map_csv = RATE_CONFIG_DIR / "modalities_en/question_maps/question_map_chest.csv"
    chest_qids, pos_prompts, neg_prompts, _ = load_prompts_from_yaml(question_map_csv)

    pos_emb, neg_emb = encode_rate_prompts(
        model.backbone_text,
        model.projection_text,
        tokenizer,
        pos_prompts,
        neg_prompts,
        device,
        normalize=True,
    )
    if normalize:
        pos_emb = F.normalize(torch.from_numpy(pos_emb), p=2, dim=-1).numpy()
        neg_emb = F.normalize(torch.from_numpy(neg_emb), p=2, dim=-1).numpy()

    rate_text_latents = np.empty((len(chest_qids) * 2, pos_emb.shape[1]), dtype=np.float32)
    rate_text_latents[0::2] = pos_emb
    rate_text_latents[1::2] = neg_emb

    mapping_file = RADFINDER_REPO_DIR / CTRATE18_TO_RATE_QIDS_FILE
    with open(mapping_file) as f:
        ctrate18_to_rate = json.load(f)
    qid_to_col = {qid: i for i, qid in enumerate(chest_qids)}

    n_q = len(chest_qids)
    n_samples = len(image_emb_np)
    sims = np.einsum("nd,pd->np", image_emb_np, rate_text_latents)
    logits = sims.reshape(n_samples, n_q, 2)
    probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1)
    preds_93 = probs[:, :, 0].numpy()  # (N, 93)

    pred_18 = np.zeros((n_samples, len(PATHOLOGIES)))
    for i, pathology in enumerate(PATHOLOGIES):
        cols = [qid_to_col[q] for q in ctrate18_to_rate[pathology]]
        pred_18[:, i] = preds_93[:, cols].max(axis=1)

    return pred_18


def load_binary_labels(
    dataset_name: str, dataset, radchestct_label_mapping: str = "extended"
) -> np.ndarray:
    """Load binary pathology labels for the given dataset."""
    all_filenames = [d["image"] for d in dataset.data]
    if dataset_name == DatasetNameC.CTRATE:
        labels_raw = pd.read_csv(
            get_medv_data_dir()
            / "public/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
        )
        labels_indexed = labels_raw.set_index("VolumeName")
        volume_names = [Path(f).name for f in all_filenames]
        return labels_indexed.loc[volume_names].to_numpy()
    elif dataset_name == DatasetNameC.RADCHESTCT:
        # Load all splits and merge (scan_keys are unique across splits: trn*/val*/tst*)
        all_native: dict[str, np.ndarray] = {}
        col_names = None
        for s in ("train", "valid", "test"):
            try:
                s_labels, s_cols = load_radchestct_labels(None, s)
                all_native.update(s_labels)
                col_names = s_cols
            except FileNotFoundError:
                pass
        assert col_names is not None, "No Rad-ChestCT label CSVs found"
        mapping = load_radchestct_to_ctrate18_mapping(variant=radchestct_label_mapping)
        scan_keys = [d["scan_key"] for d in dataset.data]
        labels_list = []
        matched = []
        for i, sk in enumerate(scan_keys):
            if sk in all_native:
                row = aggregate_radchestct_to_ctrate18(all_native[sk], col_names, mapping)
                labels_list.append(row)
                matched.append(i)
        assert len(labels_list) > 0, (
            f"Rad-ChestCT: 0/{len(scan_keys)} scans matched labels. "
            f"Sample scan_key: {scan_keys[0]}, sample label key: {next(iter(all_native))}"
        )
        dataset.data = [dataset.data[i] for i in matched]
        log_info(f"Rad-ChestCT binary ZS: {len(labels_list)}/{len(scan_keys)} scans with labels")
        return np.stack(labels_list)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported for binary zs eval")


@torch.inference_mode()
def get_text_latents(text_backbone, text_projection, tokenizer, template, device):
    """Encode pathology prompts into text embeddings."""
    texts = []
    for pathology in PATHOLOGIES:
        texts.append(template[0].format(a=pathology))
        texts.append(template[1].format(a=pathology))
    tokenizer_output = tokenizer(
        texts,
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
    return text_embeddings.detach().cpu().float().numpy()


def evaluate_chexzero(
    predictions, labels, pathologies: list[str] | None = None, verbose: bool = True
):
    """
    Evaluate using CheXzero protocol: AUROC + per-class optimal threshold via Youden's J.

    Returns:
        Tuple of (main_metrics, aux_metrics) where main_metrics has macro-averaged metrics
        and aux_metrics has per-pathology details.
    """
    if pathologies is None:
        pathologies = PATHOLOGIES
    n_path = len(pathologies)
    aurocs, precisions, sensitivities, specificities, f1s, accs, opt_thresholds = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    aux_metrics = {}

    for i in range(n_path):
        y_true = labels[:, i]
        y_score = predictions[:, i]

        try:
            auroc = roc_auc_score(y_true, y_score)
        except ValueError:
            continue

        aurocs.append(auroc)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        opt_thresh = thresholds[best_idx]
        if not np.isfinite(opt_thresh):
            opt_thresh = 0.5
        opt_thresholds.append(opt_thresh)
        sens = tpr[best_idx]
        spec = 1.0 - fpr[best_idx]
        sensitivities.append(sens)
        specificities.append(spec)

        y_pred = (y_score >= opt_thresh).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0.0)
        f1_val = f1_score(y_true, y_pred, average="weighted", zero_division=0.0)
        acc = balanced_accuracy_score(y_true, y_pred)
        precisions.append(prec)
        f1s.append(f1_val)
        accs.append(acc)

        pathology = pathologies[i]
        aux_metrics[f"{pathology}_auc"] = float(auroc)
        aux_metrics[f"{pathology}_threshold"] = float(opt_thresh)
        aux_metrics[f"{pathology}_sens"] = float(sens)
        aux_metrics[f"{pathology}_spec"] = float(spec)
        aux_metrics[f"{pathology}_prec"] = float(prec)
        aux_metrics[f"{pathology}_f1"] = float(f1_val)
        aux_metrics[f"{pathology}_acc"] = float(acc)

    if verbose:
        for i, (path, thresh) in enumerate(zip(pathologies[: len(opt_thresholds)], opt_thresholds)):
            log_info(f"    {path:<40s}: thresh={thresh:.4f} auc={aurocs[i]:.4f}")

    def _mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    main_metrics = {
        "mean_auroc": _mean(aurocs),
        "mean_prec": _mean(precisions),
        "mean_sens": _mean(sensitivities),
        "mean_spec": _mean(specificities),
        "mean_f1": _mean(f1s),
        "mean_acc": _mean(accs),
    }
    return main_metrics, aux_metrics


def evaluate_binary_auroc(y_pred, y_true, labels):
    """Compute per-class AUROC and return as a DataFrame."""
    assert len(y_pred) == len(y_true), f"{len(y_pred)=} {len(y_true)=}"
    results = {}
    for i, label in enumerate(labels):
        try:
            auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auroc = float("nan")
        results[f"{label}_auc"] = [auroc]
    return pd.DataFrame(results)
