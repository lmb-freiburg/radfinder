from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from radfinder.loader_utils import load_csv
from radfinder.paths import RATE_CONFIG_DIR, get_medv_data_dir
from radfinder.utils.logging_utils import log_debug

VALID_LANGUAGES = {"en", "de"}
VALID_MODALITIES = {"abdomen", "chest", "abdomen_chest"}
VALID_SPLITS = {"train", "val", "test"}


@dataclass
class QuestionVectorData:
    report_ids: list[str]
    qids: list[str]
    labels: np.ndarray
    split: str
    language: str
    modality: str
    source_file: Path


def normalize_split_name(split: str) -> str:
    split_norm = split.lower()
    if split_norm in {"valid", "validation"}:
        return "val"
    return split_norm


def _validate_inputs(language: str, modality: str, split: str) -> tuple[str, str, str]:
    language_norm = language.lower()
    modality_norm = modality.lower()
    split_norm = normalize_split_name(split)

    assert (
        language_norm in VALID_LANGUAGES
    ), f"Unsupported language={language}. Expected one of {sorted(VALID_LANGUAGES)}"
    assert (
        modality_norm in VALID_MODALITIES
    ), f"Unsupported modality={modality}. Expected one of {sorted(VALID_MODALITIES)}"
    assert (
        split_norm in VALID_SPLITS
    ), f"Unsupported split={split}. Expected one of {sorted(VALID_SPLITS)}"
    return language_norm, modality_norm, split_norm


def load_labels_from_raw_csv(
    labels_dir: Path,
    question_map_csv: Path,
    file_prefix: str,
    split: str,
) -> tuple[dict[str, np.ndarray], list[str], Path]:
    """
    Load long-format questions CSV and pivot to wide-format keyed by report_id.

    Args:
        labels_dir: directory containing raw CSV files (e.g. questions_chest_train.csv)
        question_map_csv: CSV mapping question text → qid
        file_prefix: e.g. "questions_chest", "questions_abdomen"
        split: "train", "val", or "test"

    Returns:
        (report_labels, qids, csv_path) where report_labels maps report_id → ndarray(num_q,)
    """
    qmap = pd.read_csv(question_map_csv)
    qids = qmap["qid"].tolist()
    qtext_to_idx = {row["question"]: i for i, (_, row) in enumerate(qmap.iterrows())}
    n_questions = len(qids)

    split_map = {"train": "train", "val": "valid", "test": "test"}
    possible_paths = (
        labels_dir / f"{file_prefix}_{split_map.get(split, split)}.csv",
        labels_dir / f"{file_prefix}_{split}.csv",
    )
    possible_paths = sorted(set(possible_paths))
    for csv_path in possible_paths:
        try:
            df = load_csv(csv_path)
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError(f"Label file not found: tried {possible_paths}")
    assert len(df) > 0, f"Label CSV is empty: {csv_path}"

    csv_questions = set(df["question"].unique())
    mapped_questions = csv_questions & set(qtext_to_idx.keys())
    unmapped_questions = csv_questions - set(qtext_to_idx.keys())
    assert len(mapped_questions) > 0, (
        f"No questions in {csv_path} match the question map {question_map_csv}. "
        f"Sample CSV questions: {list(csv_questions)[:3]}"
    )
    if unmapped_questions:
        raise ValueError(
            f"{len(unmapped_questions)} questions in CSV not in question map: "
            f"{list(unmapped_questions)[:3]}"
        )

    # Vectorized pivot: map questions to column indices, answers to 0/1
    df["_qidx"] = df["question"].map(qtext_to_idx)
    unmapped_rows = df["_qidx"].isna()
    if unmapped_rows.any():
        bad_q = df.loc[unmapped_rows, "question"].iloc[0]
        raise ValueError(f"Question in CSV not found in question map: {bad_q}")
    df["_qidx"] = df["_qidx"].astype(int)
    df["_label"] = (
        df["answer"].astype(str).str.strip().str.rstrip(".").str.lower() == "yes"
    ).astype(np.int64)
    df["_rid"] = df["report_id"].astype(str)

    # Pivot to wide format (aggfunc="last" matches old iterrows overwrite behavior for dupes)
    wide = df.pivot_table(index="_rid", columns="_qidx", values="_label", aggfunc="last")
    wide = wide.reindex(columns=range(n_questions), fill_value=0).fillna(0).astype(np.int64)

    report_labels: dict[str, np.ndarray] = {}
    labels_array = wide.to_numpy()
    for i, rid in enumerate(wide.index):
        report_labels[rid] = labels_array[i]

    assert len(report_labels) > 0, f"No reports loaded from {csv_path}"
    log_debug(
        f"Loaded {len(report_labels)} report labels from {csv_path.name} "
        f"({len(mapped_questions)}/{n_questions} questions matched)"
    )
    return report_labels, qids, csv_path


def _get_question_map_csv(language: str, modality: str) -> Path:
    return (
        RATE_CONFIG_DIR
        / f"modalities_{language}"
        / "question_maps"
        / f"question_map_{modality}.csv"
    )


def _get_wide_csv_path(labels_dir: Path, modality: str, split: str) -> Path:
    """Find wide CSV, trying both 'val' and 'valid' naming conventions."""
    base = labels_dir / "question_vectors"
    primary = base / f"question_vector_{modality}_{split}.csv"
    if primary.is_file():
        return primary
    # Some datasets use "valid" instead of "val"
    alt_names = {"val": "valid", "valid": "val"}
    if split in alt_names:
        alt = base / f"question_vector_{modality}_{alt_names[split]}.csv"
        if alt.is_file():
            return alt
    # Return primary path (will be created by _generate_wide_csv if needed)
    return primary


def _load_wide_csv(wide_csv: Path, modality: str, split: str) -> QuestionVectorData:
    """Load pre-pivoted wide-format CSV (one row per report, one column per qid)."""
    df = pd.read_csv(wide_csv)
    assert "report_id" in df.columns, f"Missing report_id column in {wide_csv}"
    qids = [col for col in df.columns if col != "report_id"]
    assert len(qids) > 0, f"No question columns found in {wide_csv}"

    duplicates = int(df["report_id"].duplicated().sum())
    assert duplicates == 0, f"Found {duplicates} duplicate report_ids in {wide_csv}"

    labels_df = df[qids].copy()
    unique_values = set(np.unique(labels_df.to_numpy()))
    assert unique_values.issubset(
        {0, 1}
    ), f"Non-binary values {sorted(unique_values)} in {wide_csv}"

    log_debug(
        f"Loaded {len(df)} report labels from {wide_csv.name} (wide format, {len(qids)} questions)"
    )
    return QuestionVectorData(
        report_ids=df["report_id"].astype(str).tolist(),
        qids=qids,
        labels=labels_df.to_numpy(dtype=np.int64),
        split=split,
        language="",
        modality=modality,
        source_file=wide_csv,
    )


def _generate_wide_csv(
    labels_dir: Path,
    language: str,
    modality: str,
    file_prefix: str,
    split: str,
) -> Path:
    """Generate wide-format CSV from long-format CSV. Returns path to the new file."""
    question_map_csv = _get_question_map_csv(language, modality)
    report_labels, qids, csv_path = load_labels_from_raw_csv(
        labels_dir=labels_dir,
        question_map_csv=question_map_csv,
        file_prefix=file_prefix,
        split=split,
    )

    report_ids = sorted(report_labels.keys())
    labels = np.stack([report_labels[rid] for rid in report_ids])

    wide_csv = _get_wide_csv_path(labels_dir, modality, split)
    wide_csv.parent.mkdir(parents=True, exist_ok=True)

    wide_df = pd.DataFrame(labels, columns=qids)
    wide_df.insert(0, "report_id", report_ids)
    wide_df.to_csv(wide_csv, index=False)
    log_debug(f"Generated wide CSV: {wide_csv} ({len(report_ids)} reports, {len(qids)} questions)")

    # Verify round-trip: reload and compare
    result = _load_wide_csv(wide_csv, modality, split)
    assert len(result.report_ids) == len(
        report_ids
    ), f"Wide CSV round-trip failed: wrote {len(report_ids)} reports, read back {len(result.report_ids)}"
    assert (
        result.labels.shape == labels.shape
    ), f"Wide CSV round-trip failed: wrote shape {labels.shape}, read back {result.labels.shape}"
    assert np.array_equal(result.labels, labels), "Wide CSV round-trip failed: labels mismatch"

    return wide_csv


def load_question_vectors(
    language: str,
    modality: str,
    split: str,
    labels_dir: Path | None = None,
    file_prefix: str | None = None,
) -> QuestionVectorData:
    """Load question labels from wide-format CSV, generating it from long-format if needed."""
    language_norm, modality_norm, split_norm = _validate_inputs(language, modality, split)

    assert labels_dir is not None, "labels_dir must be provided"

    if file_prefix is None:
        file_prefix = f"questions_{modality_norm}"

    wide_csv = _get_wide_csv_path(labels_dir, modality_norm, split_norm)
    if not wide_csv.is_file():
        _generate_wide_csv(labels_dir, language_norm, modality_norm, file_prefix, split_norm)

    result = _load_wide_csv(wide_csv, modality_norm, split_norm)
    result.language = language_norm
    return result


def load_question_vectors_all_splits(
    language: str,
    modality: str,
    labels_dir: Path | None = None,
    file_prefix: str | None = None,
) -> dict[str, QuestionVectorData]:
    out: dict[str, QuestionVectorData] = {}
    for split in ["train", "val", "test"]:
        out[split] = load_question_vectors(
            language=language,
            modality=modality,
            split=split,
            labels_dir=labels_dir,
            file_prefix=file_prefix,
        )
    return out
