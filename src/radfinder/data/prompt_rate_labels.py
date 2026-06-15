"""
Prompt-rate label loading + prompt selection for the auxiliary disease loss.

The training pipeline (`train_siglip.py`) uses these helpers to:
  - Pick which positive/negative text prompts to embed for the prompt-rate loss
    (`load_prompts_for_mode`).
  - Build per-scan binary disease-label tensors keyed by `scan_key`
    (`build_prompt_rate_labels`).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from radfinder.data.ct_rate import CTRateDataset, extract_report_key, get_ctrate_image_paths
from radfinder.data.inspect import InspectDataset, get_inspect_image_paths
from radfinder.data.merlin import MerlinDataset, get_merlin_image_paths
from radfinder.data.rad_chestct import (
    RadChestCTDataset,
    aggregate_radchestct_to_ctrate18,
    get_radchestct_image_paths,
    load_radchestct_labels,
    load_radchestct_to_ctrate18_mapping,
)
from radfinder.loader_utils import load_csv
from radfinder.paths import RADFINDER_REPO_DIR, RATE_CONFIG_DIR, get_medv_data_dir
from radfinder.tasks.binary_zs_rate_task import get_ctrate_labels_dir, load_prompts_from_yaml
from radfinder.tasks.question_vector_loader import load_question_vectors
from radfinder.utils.logging_utils import log_debug

from packg.constclass import Const


class PromptRateModeC(Const):
    RATE = "rate"
    CTRATE = "ctrate"
    BOTH = "both"


PATHOLOGIES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]


TEMPLATES = {
    "T1": ("{a}.", "No {a}."),
    "T2": ("Findings are compatible with {a}.", "Findings are not compatible with {a}."),
    "T3": ("There is {a}.", "There is no {a}."),
    "T4": ("{a} is seen.", "{a} is not seen."),
    "T5": ("{a}.", "Not {a}."),
    "T6": ("{a} is observed.", "{a} is not observed."),
    "T7": ("{a} is present.", "{a} is not present."),
}

CTRATE18_TO_RATE_QIDS_FILE = f"configs/tasks/binary_zs/ctrate18_to_rate_qids.json"

CTRATE18_PROMPTS: dict[str, dict[str, list[str]]] = {
    "Medical material": {
        "positive": [
            "Medical material is present in the thorax.",
            "Implanted medical devices or surgical hardware are identified in the chest.",
            "Radiopaque foreign material consistent with medical devices such as stents, catheters, pacemaker leads, or surgical hardware is seen within the thorax.",
        ],
        "negative": [
            "There is no medical material in the thorax.",
            "No implanted medical devices or surgical hardware are identified.",
            "The thorax is free of radiopaque foreign material, medical devices, or surgical hardware.",
        ],
    },
    "Arterial wall calcification": {
        "positive": [
            "Arterial wall calcification is present.",
            "Atherosclerotic calcifications are identified along the arterial walls.",
            "Dense mural calcified plaques are seen along the walls of the aorta, coronary, or other thoracic arteries, consistent with atherosclerotic disease.",
        ],
        "negative": [
            "There is no arterial wall calcification.",
            "No atherosclerotic arterial wall calcifications are identified.",
            "The thoracic arterial walls are smooth without mural calcified plaques or atherosclerotic changes.",
        ],
    },
    "Cardiomegaly": {
        "positive": [
            "Cardiomegaly is present.",
            "The heart is enlarged, consistent with cardiomegaly.",
            "The heart is significantly increased in size on cross-sectional imaging, indicating cardiac enlargement.",
        ],
        "negative": [
            "There is no cardiomegaly.",
            "No cardiac enlargement is identified.",
            "The heart is normal in size without evidence of cardiomegaly.",
        ],
    },
    "Pericardial effusion": {
        "positive": [
            "Pericardial effusion is present.",
            "Fluid is identified within the pericardial space, consistent with pericardial effusion.",
            "A layer of low-attenuation fluid surrounds the heart within the pericardial sac.",
        ],
        "negative": [
            "There is no pericardial effusion.",
            "No pericardial effusion is identified.",
            "The pericardial space is free of fluid.",
        ],
    },
    "Coronary artery wall calcification": {
        "positive": [
            "Coronary artery wall calcification is present.",
            "Atherosclerotic calcifications are identified within the coronary arteries.",
            "The coronary arteries demonstrate mural calcified plaques consistent with coronary atherosclerotic disease.",
        ],
        "negative": [
            "There is no coronary artery wall calcification.",
            "No coronary artery calcifications are identified.",
            "The coronary arteries are free of atherosclerotic calcifications.",
        ],
    },
    "Hiatal hernia": {
        "positive": [
            "A hiatal hernia is present.",
            "Herniation of the stomach through the esophageal hiatus is identified.",
            "There is superior displacement of the gastric fundus through the diaphragmatic hiatus into the thorax, consistent with a hiatal hernia.",
        ],
        "negative": [
            "There is no hiatal hernia.",
            "No hiatal hernia is identified.",
            "The gastroesophageal junction is in normal position without herniation of abdominal contents through the esophageal hiatus.",
        ],
    },
    "Lymphadenopathy": {
        "positive": [
            "Lymphadenopathy is present in the thorax.",
            "Pathologically enlarged lymph nodes are identified in the mediastinum, hila, or axillae.",
            "There are enlarged lymph nodes exceeding 1 cm in short axis in the mediastinal, hilar, axillary, or cervical regions.",
        ],
        "negative": [
            "There is no lymphadenopathy.",
            "No pathologically enlarged lymph nodes are identified.",
            "The mediastinal, hilar, axillary, and cervical lymph nodes are within normal limits in size.",
        ],
    },
    "Emphysema": {
        "positive": [
            "Emphysema is present in the lungs.",
            "Emphysematous changes are identified in the lung parenchyma.",
            "Areas of abnormally low attenuation without visible walls are seen in the lungs, consistent with parenchymal destruction from emphysema.",
        ],
        "negative": [
            "There is no emphysema.",
            "No emphysematous changes are identified in the lungs.",
            "The lung parenchyma does not demonstrate areas of abnormal lucency or parenchymal destruction suggestive of emphysema.",
        ],
    },
    "Atelectasis": {
        "positive": [
            "Atelectasis is present.",
            "Areas of atelectasis are identified in the lung.",
            "Band-like or wedge-shaped opacities with volume loss are seen in the lung parenchyma, consistent with atelectasis.",
        ],
        "negative": [
            "There is no atelectasis.",
            "No atelectasis is identified.",
            "The lungs are fully expanded without areas of volume loss or atelectatic change.",
        ],
    },
    "Lung nodule": {
        "positive": [
            "A lung nodule is present.",
            "One or more pulmonary nodules are identified in the lungs.",
            "Discrete focal opacities are seen within the lung parenchyma, representing solid, part-solid, or ground-glass pulmonary nodules.",
        ],
        "negative": [
            "There are no lung nodules.",
            "No pulmonary nodules are identified.",
            "The lung parenchyma is free of discrete nodular opacities.",
        ],
    },
    "Lung opacity": {
        "positive": [
            "Lung opacity is present.",
            "Parenchymal opacities are identified in the lungs.",
            "Areas of increased attenuation are seen in the lung parenchyma, consistent with ground-glass opacity, consolidation, or other airspace opacification.",
        ],
        "negative": [
            "There is no lung opacity.",
            "No parenchymal opacities are identified in the lungs.",
            "The lung parenchyma is clear without ground-glass opacity, consolidation, or airspace opacification.",
        ],
    },
    "Pulmonary fibrotic sequela": {
        "positive": [
            "Pulmonary fibrotic sequela is present.",
            "Findings of pulmonary fibrosis are identified in the lungs.",
            "Reticulation, traction bronchiectasis, or honeycombing is seen in the lung parenchyma, indicating fibrotic interstitial lung disease.",
        ],
        "negative": [
            "There is no pulmonary fibrotic sequela.",
            "No pulmonary fibrosis is identified.",
            "The lungs do not demonstrate reticulation, traction bronchiectasis, or honeycombing to suggest fibrotic disease.",
        ],
    },
    "Pleural effusion": {
        "positive": [
            "Pleural effusion is present.",
            "Fluid is identified within the pleural space.",
            "There is dependent layering fluid within the pleural space, consistent with pleural effusion.",
        ],
        "negative": [
            "There is no pleural effusion.",
            "No pleural effusion is identified.",
            "The pleural spaces are clear without fluid collection.",
        ],
    },
    "Mosaic attenuation pattern": {
        "positive": [
            "A mosaic attenuation pattern is present in the lungs.",
            "Geographic areas of differing lung density are identified, consistent with mosaic attenuation.",
            "There are sharply demarcated regions of heterogeneous lung attenuation with alternating areas of hyper- and hypo-attenuation, consistent with a mosaic pattern.",
        ],
        "negative": [
            "There is no mosaic attenuation pattern.",
            "No mosaic attenuation is identified in the lungs.",
            "The lung parenchyma demonstrates homogeneous attenuation without geographic variation in density to suggest mosaic attenuation.",
        ],
    },
    "Peribronchial thickening": {
        "positive": [
            "Peribronchial thickening is present.",
            "Thickening of the bronchial walls is identified.",
            "There is circumferential soft tissue thickening surrounding the bronchi, consistent with peribronchial thickening.",
        ],
        "negative": [
            "There is no peribronchial thickening.",
            "No peribronchial thickening is identified.",
            "The bronchial walls are normal in thickness without peribronchial thickening.",
        ],
    },
    "Consolidation": {
        "positive": [
            "Pulmonary consolidation is present.",
            "An area of consolidation is identified in the lung parenchyma.",
            "A region of increased attenuation with air bronchograms obscuring the underlying pulmonary vasculature is observed, consistent with consolidation.",
        ],
        "negative": [
            "There is no pulmonary consolidation.",
            "No consolidation is identified in the lungs.",
            "The lung parenchyma is clear without areas of consolidation or airspace opacification.",
        ],
    },
    "Bronchiectasis": {
        "positive": [
            "Bronchiectasis is present.",
            "Irreversible airway dilatation consistent with bronchiectasis is identified.",
            "There are dilated thick-walled bronchi exceeding the diameter of their accompanying pulmonary arteries, consistent with bronchiectasis.",
        ],
        "negative": [
            "There is no bronchiectasis.",
            "No bronchiectasis is identified.",
            "The airways are normal in caliber without evidence of abnormal bronchial dilatation.",
        ],
    },
    "Interlobular septal thickening": {
        "positive": [
            "Interlobular septal thickening is present.",
            "Thickened interlobular septa are identified in the lungs.",
            "There are thickened interlobular septa visible as fine linear opacities outlining the secondary pulmonary lobules.",
        ],
        "negative": [
            "There is no interlobular septal thickening.",
            "No interlobular septal thickening is identified.",
            "The interlobular septa are normal in thickness without evidence of thickening.",
        ],
    },
}


def load_ctrate18_prompts() -> tuple[list[str], list[list[str]], list[list[str]]]:
    """Load 18 CT-RATE pathology prompts from ctrate18_prompts.yaml.

    Returns (pathology_names, positive_prompts, negative_prompts).
    """
    pos_prompts = []
    neg_prompts = []
    for pathology in PATHOLOGIES:
        entry = CTRATE18_PROMPTS[pathology]
        assert len(entry["positive"]) == 3 and len(entry["negative"]) == 3, (
            f"Expected 3 pos/neg prompts for {pathology}, got "
            f"{len(entry['positive'])}/{len(entry['negative'])}"
        )
        pos_prompts.append(entry["positive"])
        neg_prompts.append(entry["negative"])
    return list(PATHOLOGIES), pos_prompts, neg_prompts


def load_prompts_for_mode(
    mode: str,
) -> tuple[list[list[str]], list[list[str]]]:
    """Load positive and negative prompts based on prompt-rate mode.

    Returns (pos_prompts, neg_prompts) where each is list of list[str]
    (3 variants per question).
    """
    mode = PromptRateModeC.verify_value(mode)
    if mode == PromptRateModeC.RATE:
        question_map_csv = (
            RATE_CONFIG_DIR / "modalities_en/question_maps/question_map_abdomen_chest.csv"
        )
        _, pos_prompts, neg_prompts, _ = load_prompts_from_yaml(question_map_csv)
        return pos_prompts, neg_prompts
    elif mode == PromptRateModeC.CTRATE:
        _, pos_prompts, neg_prompts = load_ctrate18_prompts()
        return pos_prompts, neg_prompts
    elif mode == PromptRateModeC.BOTH:
        question_map_csv = (
            RATE_CONFIG_DIR / "modalities_en/question_maps/question_map_abdomen_chest.csv"
        )
        _, rate_pos, rate_neg, _ = load_prompts_from_yaml(question_map_csv)
        _, ctrate_pos, ctrate_neg = load_ctrate18_prompts()
        return rate_pos + ctrate_pos, rate_neg + ctrate_neg
    raise ValueError(f"Unknown mode: {mode}")


# ──────────────────────────── label loading ────────────────────────────


def load_merged_rate_labels(
    split: str,
    language: str = "en",
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Merge abdomen_chest + abdomen + chest QVs into unified 319-dim labels.

    Returns (report_id → 319-dim label array with -1 for unknown, canonical qids).
    """
    qv_ac = load_question_vectors(language, "abdomen_chest", split)
    qv_ab = load_question_vectors(language, "abdomen", split)
    qv_ch = load_question_vectors(language, "chest", split)
    canonical_qids = qv_ac.qids  # q0001..q0319
    assert len(canonical_qids) == 319

    # Build per-modality lookups (report_id → row index)
    ac_lookup = {str(r): i for i, r in enumerate(qv_ac.report_ids)}
    ab_lookup = {str(r): i for i, r in enumerate(qv_ab.report_ids)}
    ch_lookup = {str(r): i for i, r in enumerate(qv_ch.report_ids)}

    # Merge all report IDs from all 3 disjoint modality CSVs
    all_report_ids = set(str(r) for r in qv_ac.report_ids)
    all_report_ids.update(str(r) for r in qv_ab.report_ids)
    all_report_ids.update(str(r) for r in qv_ch.report_ids)

    labels_dict: dict[str, np.ndarray] = {}
    for rid in all_report_ids:
        row = np.full(319, -1, dtype=np.int64)
        if rid in ac_lookup:
            # abdomen_chest reports have all 319 questions
            row[:] = qv_ac.labels[ac_lookup[rid]]
        else:
            # abdomen-only: q0001-q0226 (indices 0-225)
            if rid in ab_lookup:
                row[:226] = qv_ab.labels[ab_lookup[rid]]
            # chest-only: q0227-q0319 (indices 226-318)
            if rid in ch_lookup:
                row[226:] = qv_ch.labels[ch_lookup[rid]]
        labels_dict[rid] = row

    n_full = sum(1 for v in labels_dict.values() if (v != -1).all())
    n_partial = sum(1 for v in labels_dict.values() if (v == -1).any())
    log_debug(
        f"[prompt_rate split={split} mode=rate] Merged labels: {len(labels_dict)} reports "
        f"({n_full} full 319-dim, {n_partial} partial)"
    )
    return labels_dict, canonical_qids


def build_prompt_rate_labels(
    dataset_names: list[str],
    split: str = "train",
    language: str = "en",
    mode: str = PromptRateModeC.RATE,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Build scan_key -> label tensor for all given datasets.

    Supports 3 modes:
    - rate: 319-dim RaTE labels (original)
    - ctrate: 18-dim CT-RATE pathology labels
    - both: 337-dim (319 rate + 18 ctrate concatenated)

    Returns (pr_labels: scan_key -> Tensor, labels_dict: key -> ndarray for pos_weight).
    """
    mode = PromptRateModeC.verify_value(mode)
    if mode == PromptRateModeC.RATE:
        return _build_rate_labels(dataset_names, split, language)
    elif mode == PromptRateModeC.CTRATE:
        return _build_ctrate18_labels(dataset_names, split)
    elif mode == PromptRateModeC.BOTH:
        rate_pr, rate_dict = _build_rate_labels(dataset_names, split, language)
        ctrate_pr, _ = _build_ctrate18_labels(dataset_names, split)
        all_keys = set(rate_pr.keys()) | set(ctrate_pr.keys())
        combined_pr: dict[str, torch.Tensor] = {}
        combined_dict: dict[str, np.ndarray] = {}
        rate_dim = 319
        ctrate_dim = len(PATHOLOGIES)
        for sk in all_keys:
            rate_lbl = rate_pr.get(sk, torch.full((rate_dim,), -1, dtype=torch.long))
            ctrate_lbl = ctrate_pr.get(sk, torch.full((ctrate_dim,), -1, dtype=torch.long))
            combined = torch.cat([rate_lbl, ctrate_lbl])
            combined_pr[sk] = combined
            combined_dict[sk] = combined.numpy()
        log_debug(
            f"[prompt_rate split={split} mode=both] "
            f"Combined labels: {len(combined_pr)} scans, {rate_dim + ctrate_dim} dims"
        )
        return combined_pr, combined_dict
    raise ValueError(f"Unknown mode: {mode}")


def _build_rate_labels(
    dataset_names: list[str],
    split: str,
    language: str,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Build 319-dim RaTE labels (original behavior)."""
    labels_dict: dict[str, np.ndarray] = {}
    pr_labels: dict[str, torch.Tensor] = {}

    if "ctrate" in dataset_names:
        qv_ctrate = load_question_vectors(
            "en",
            "chest",
            split,
            labels_dir=get_ctrate_labels_dir(),
            file_prefix="questions_chest",
        )
        ctrate_labels = {rid: qv_ctrate.labels[i] for i, rid in enumerate(qv_ctrate.report_ids)}
        ctrate_qids = qv_ctrate.qids
        for rid, chest_labels in ctrate_labels.items():
            row = np.full(319, -1, dtype=np.int64)
            row[226:] = chest_labels
            labels_dict[rid] = row
        ctrate_data_dir = Path(get_medv_data_dir()) / "public/CT-RATE"
        ctrate_split = "valid" if split == "val" else split
        ctrate_image_paths = get_ctrate_image_paths(ctrate_data_dir, ctrate_split)
        n_ctrate = 0
        for p in ctrate_image_paths:
            sk = CTRateDataset.get_datapoint_key_from_scan_path(str(p))
            rk = extract_report_key(str(p))
            if rk in ctrate_labels:
                pr_labels[sk] = torch.from_numpy(labels_dict[rk])
                n_ctrate += 1
        log_debug(
            f"[prompt_rate split={split} dataset=ctrate mode=rate] {n_ctrate} labeled scans, "
            f"{len(ctrate_labels)} reports, {len(ctrate_qids)} chest questions"
        )

    if "merlin" in dataset_names:
        merlin_labels_dir = Path(get_medv_data_dir()) / "public/Merlin/report_structuring/p0rate_en"
        qv_merlin = load_question_vectors(
            "en",
            "abdomen",
            split,
            labels_dir=merlin_labels_dir,
            file_prefix="questions_abdomen",
        )
        merlin_labels = {rid: qv_merlin.labels[i] for i, rid in enumerate(qv_merlin.report_ids)}
        merlin_qids = qv_merlin.qids
        for rid, abdomen_labels in merlin_labels.items():
            row = np.full(319, -1, dtype=np.int64)
            row[:226] = abdomen_labels
            labels_dict[rid] = row

        merlin_image_paths = get_merlin_image_paths(
            Path(get_medv_data_dir()) / "public/Merlin", split
        )
        n_merlin = 0
        for p in merlin_image_paths:
            sk = MerlinDataset.get_datapoint_key_from_scan_path(p)
            if sk in merlin_labels:
                pr_labels[sk] = torch.from_numpy(labels_dict[sk])
                n_merlin += 1
        log_debug(
            f"[prompt_rate split={split} dataset=merlin mode=rate] {n_merlin} labeled scans, "
            f"{len(merlin_labels)} reports, {len(merlin_qids)} abdomen questions"
        )
        assert n_merlin > 0, (
            f"Merlin: 0/{len(merlin_image_paths)} scans matched labels. "
            f"Sample scan_key: {MerlinDataset.get_datapoint_key_from_scan_path(merlin_image_paths[0])}, "
            f"sample label key: {next(iter(merlin_labels))}"
        )

    if "inspect" in dataset_names:
        inspect_labels_dir = (
            Path(get_medv_data_dir()) / "public/Inspect/report_structuring/p0rate_en"
        )
        inspect_split = "valid" if split == "val" else split
        qv_inspect = load_question_vectors(
            "en",
            "chest",
            inspect_split,
            labels_dir=inspect_labels_dir,
            file_prefix="questions_chest",
        )
        inspect_labels = {rid: qv_inspect.labels[i] for i, rid in enumerate(qv_inspect.report_ids)}
        inspect_qids = qv_inspect.qids
        for rid, chest_labels in inspect_labels.items():
            row = np.full(319, -1, dtype=np.int64)
            row[226:] = chest_labels
            labels_dict[rid] = row

        inspect_image_paths = get_inspect_image_paths(
            Path(get_medv_data_dir()) / "public/Inspect", inspect_split
        )
        n_inspect = 0
        for p in inspect_image_paths:
            sk = InspectDataset.get_datapoint_key_from_scan_path(p)
            if sk in inspect_labels:
                pr_labels[sk] = torch.from_numpy(labels_dict[sk])
                n_inspect += 1
        log_debug(
            f"[prompt_rate split={split} dataset=inspect mode=rate] {n_inspect} labeled scans, "
            f"{len(inspect_labels)} reports, {len(inspect_qids)} chest questions"
        )
        assert n_inspect > 0, (
            f"Inspect: 0/{len(inspect_image_paths)} scans matched labels. "
            f"Sample scan_key: {InspectDataset.get_datapoint_key_from_scan_path(inspect_image_paths[0])}, "
            f"sample label key: {next(iter(inspect_labels))}"
        )

    return pr_labels, labels_dict


def _build_ctrate18_labels(
    dataset_names: list[str],
    split: str,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Build 18-dim CT-RATE pathology labels.

    CT-RATE: from {split}_predicted_labels.csv.
    """
    num_classes = len(PATHOLOGIES)
    pathology_list = list(PATHOLOGIES)
    pr_labels: dict[str, torch.Tensor] = {}
    labels_dict: dict[str, np.ndarray] = {}

    # CT-RATE labels from multi_abnormality_labels CSV
    if "ctrate" in dataset_names:
        ctrate_data_dir = Path(get_medv_data_dir()) / "public/CT-RATE"
        ctrate_split = "valid" if split == "val" else split
        label_file = (
            ctrate_data_dir
            / "dataset/multi_abnormality_labels"
            / f"{ctrate_split}_predicted_labels.csv"
        )
        assert label_file.is_file(), f"CT-RATE label file not found: {label_file}"
        ctrate_df = pd.read_csv(label_file).set_index("VolumeName")

        ctrate_image_paths = get_ctrate_image_paths(ctrate_data_dir, ctrate_split)
        sk_list = [
            CTRateDataset.get_datapoint_key_from_scan_path(str(p)) for p in ctrate_image_paths
        ]
        vn_list = [Path(p).name for p in ctrate_image_paths]
        common_vns = [vn for vn in vn_list if vn in ctrate_df.index]
        if common_vns:
            batch = ctrate_df.loc[common_vns, pathology_list].to_numpy().astype(np.int64)
            vn_to_sk = dict(zip(vn_list, sk_list))
            for i, vn in enumerate(common_vns):
                sk = vn_to_sk[vn]
                row = batch[i]
                pr_labels[sk] = torch.from_numpy(row)
                labels_dict[sk] = row
        log_debug(
            f"[prompt_rate split={split} dataset=ctrate mode=ctrate] "
            f"{len(common_vns)} labeled scans"
        )

    if "inspect" in dataset_names:
        inspect_labels_dir = (
            Path(get_medv_data_dir()) / "public/Inspect/report_structuring/p0rate_en"
        )
        inspect_split = "valid" if split == "val" else split
        qv_inspect = load_question_vectors(
            "en",
            "chest",
            inspect_split,
            labels_dir=inspect_labels_dir,
            file_prefix="questions_chest",
        )
        inspect_labels = {rid: qv_inspect.labels[i] for i, rid in enumerate(qv_inspect.report_ids)}
        inspect_qids = qv_inspect.qids

        mapping_file = RADFINDER_REPO_DIR / CTRATE18_TO_RATE_QIDS_FILE
        with open(mapping_file) as f:
            ctrate18_to_rate = json.load(f)
        qid_to_col = {qid: i for i, qid in enumerate(inspect_qids)}

        inspect_image_paths = get_inspect_image_paths(
            Path(get_medv_data_dir()) / "public/Inspect", inspect_split
        )
        n_inspect = 0
        for p in inspect_image_paths:
            sk = InspectDataset.get_datapoint_key_from_scan_path(p)
            if sk not in inspect_labels:
                continue
            chest_labels = inspect_labels[sk]
            row = np.zeros(num_classes, dtype=np.int64)
            for i, pathology in enumerate(pathology_list):
                rate_qids = ctrate18_to_rate[pathology]
                cols = [qid_to_col[q] for q in rate_qids if q in qid_to_col]
                if cols:
                    row[i] = int(chest_labels[cols].max())
            pr_labels[sk] = torch.from_numpy(row)
            labels_dict[sk] = row
            n_inspect += 1
        log_debug(
            f"[prompt_rate split={split} dataset=inspect mode=ctrate] " f"{n_inspect} labeled scans"
        )
        assert n_inspect > 0, (
            f"Inspect ctrate18: 0/{len(inspect_image_paths)} scans matched labels. "
            f"Sample scan_key: {InspectDataset.get_datapoint_key_from_scan_path(inspect_image_paths[0])}, "
            f"sample label key: {next(iter(inspect_labels))}"
        )

    if "radchestct" in dataset_names:
        radchestct_native, col_names = load_radchestct_labels(None, split)
        mapping = load_radchestct_to_ctrate18_mapping()

        radchestct_data_dir = Path(get_medv_data_dir()) / "public/Rad-ChestCT"
        radchestct_split = "valid" if split == "val" else split
        radchestct_image_paths = get_radchestct_image_paths(radchestct_data_dir, radchestct_split)
        n_radchestct = 0
        for p in radchestct_image_paths:
            sk = RadChestCTDataset.get_datapoint_key_from_scan_path(p)
            if sk in radchestct_native:
                row = aggregate_radchestct_to_ctrate18(radchestct_native[sk], col_names, mapping)
                pr_labels[sk] = torch.from_numpy(row)
                labels_dict[sk] = row
                n_radchestct += 1
        log_debug(
            f"[prompt_rate split={split} dataset=radchestct mode=ctrate] "
            f"{n_radchestct} labeled scans"
        )
        assert n_radchestct > 0, (
            f"Rad-ChestCT ctrate18: 0/{len(radchestct_image_paths)} scans matched labels. "
            f"Sample scan_key: {RadChestCTDataset.get_datapoint_key_from_scan_path(radchestct_image_paths[0])}, "
            f"sample label key: {next(iter(radchestct_native))}"
        )

    return pr_labels, labels_dict
