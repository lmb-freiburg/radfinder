from pathlib import Path

import pandas as pd
from radfinder.loader_utils import load_csv
from radfinder.paths import get_medv_data_dir

from packg.iotools import dump_json, load_json

ORGAN_TEXT_CATEGORY_RENAME = {
    "Device": "Devices in the Abdomen",
    "Great Vessel": "Great Vessels in the Abdomen",
}

_cache_dir = get_medv_data_dir() / "text_cache"


def get_path_safe_string(input_str: str) -> str:
    """Get a safe string for a path."""
    if isinstance(input_str, Path):
        input_str = input_str.as_posix()
    return input_str.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")


def _get_organ_cache_path(organ_text_dir: str, split: str):
    cache_path = _cache_dir / f"{get_path_safe_string(organ_text_dir)}--split-{split}.json"
    return cache_path


def load_organ_texts(organ_text_dir: str, split: str) -> dict[str, str]:
    """Load organ category texts from all category_findings CSVs in the given directory.

    Args:
        organ_text_dir: Path to directory containing category_findings_*_{split}.csv files.
        split: Data split name (train, val, valid, test).

    Returns:
        {report_id: formatted_text} where formatted_text is
        "Category1: Findings1\\nCategory2: Findings2\\n..." with NRF entries skipped.
    """
    cache_path = _get_organ_cache_path(organ_text_dir, split)
    if cache_path.is_file():
        return load_json(cache_path)

    organ_text_dir_p: Path = Path(organ_text_dir)
    if not organ_text_dir_p.is_dir():
        raise ValueError(f"Organ text directory {organ_text_dir_p} does not exist")
    csv_files = sorted(organ_text_dir_p.glob(f"category_findings_*_{split}.csv"))
    if not csv_files:
        raise ValueError(f"No organ text CSV files found in {organ_text_dir_p} split {split}")
    return load_organ_csvs(organ_text_dir, split, csv_files)


def load_organ_csvs(organ_text_dir: str, split: str, csv_files: list[Path]) -> dict[str, str]:
    organ_text_dir_p = Path(organ_text_dir)
    dfs = [
        load_csv(
            f,
            usecols=["report_id", "category", "findings"],
            dtype={"report_id": "string", "category": "string", "findings": "string"},
        )
        for f in csv_files
    ]
    df = pd.concat(dfs, ignore_index=True)
    df["category"] = df["category"].replace(ORGAN_TEXT_CATEGORY_RENAME)

    # fast and hopefully the same logic
    findings = df["findings"].fillna("").str.strip()
    keep = ~findings.str.lower().str.startswith("no relevant findings")
    df = df.loc[keep, ["report_id", "category"]].copy()
    df["findings"] = findings.loc[keep].values
    if df.empty:
        raise ValueError(f"No organ texts found in {organ_text_dir_p} split {split}")
    df["line"] = df["category"] + ": " + df["findings"]
    result = df.groupby("report_id", sort=False)["line"].agg("\n".join).to_dict()

    cache_path = _get_organ_cache_path(organ_text_dir, split)
    dump_json(result, cache_path, create_parent=True, indent=2)
    return result


def load_no_comparisons_texts(
    report_structuring_dir: str, split: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Load no_comparisons findings and impressions from CSVs.

    Args:
        report_structuring_dir: Path to directory containing no_comparisons_*_{split}.csv files.
        split: Data split name (train, val, valid, test).

    Returns:
        (findings_dict, impressions_dict) keyed by report_id. Empty dicts if files don't exist.
    """
    cache_path = (
        _cache_dir / f"{get_path_safe_string(report_structuring_dir)}--nocomp--split-{split}.json"
    )
    if cache_path.is_file():
        cached = load_json(cache_path)
        return cached["findings"], cached["impressions"]

    dir_path = Path(report_structuring_dir)

    findings_dict: dict[str, str] = {}
    findings_files = sorted(dir_path.glob(f"no_comparisons_findings_*_{split}.csv"))
    for f in findings_files:
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            text = str(row["no_comparison_findings"]).strip()
            if text and text.lower() != "nan":
                findings_dict[str(row["report_id"])] = text

    impressions_dict: dict[str, str] = {}
    impressions_files = sorted(dir_path.glob(f"no_comparisons_impressions_*_{split}.csv"))
    for f in impressions_files:
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            text = str(row["no_comparison_impressions"]).strip()
            if text and text.lower() != "nan":
                impressions_dict[str(row["report_id"])] = text

    dump_json(
        {"findings": findings_dict, "impressions": impressions_dict},
        cache_path,
        create_parent=True,
        indent=2,
    )
    return findings_dict, impressions_dict
