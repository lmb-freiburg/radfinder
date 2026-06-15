"""
Inspect dataset utilities for the RATE report structuring pipeline.

Inspect is a CTPA dataset with ~23k chest CT scans from Stanford.
Reports only contain impressions (no findings section).
Data is stored in versioned TSV files that need to be joined.
"""

from pathlib import Path

import pandas as pd
from radfinder.paths import get_medv_data_dir
from rate.rate_common_utils import load_outputs

INSPECT_DEFAULT_DATA_DIR = get_medv_data_dir() / "public/Inspect"
INSPECT_DEFAULT_OUTPUT_DIR = get_medv_data_dir() / "public/Inspect/report_structuring/p0rate_en"


# Scans that crash dataloaders (4D scans or orientation errors).
BROKEN_KEYS = {
    # 4D scan (4 axes instead of 3), crashes dataloaders
    "PE877f77",
    "PE8636b5",
    "PE9f353a",
    "PE874dae",
    "PE864a1e",
    "PE452aba3",
    "PE9f680b",
    "PE9f63cb",
    "PE9f6ce4",
    "PE9f6d6e",
    "PE9f536a",
    "PEc255bb",
    "PE9f417d",
    # Unable to find out axis 2.0 in start_ornt
    "PE864879",
    "PE86493f",
    "PE8774e7",
    "PE9f6db8",
    "PE9f59c5",
    "PE9f45c6",
    "PE9f4c0d",
}


def get_inspect_tsv(data_dir: Path, prefix: str) -> Path:
    """Find the latest versioned TSV file for a given prefix (e.g. 'impressions')."""
    candidates = sorted(data_dir.glob(f"full/{prefix}_*.tsv"))
    assert len(candidates) > 0, f"No TSV files matching full/{prefix}_*.tsv in {data_dir}"
    return candidates[-1]


def load_inspect_reports(data_dir: Path) -> pd.DataFrame:
    """Load Inspect reports by joining study_mapping, impressions, and splits TSVs."""
    mapping = pd.read_csv(get_inspect_tsv(data_dir, "study_mapping"), sep="\t")
    impressions = pd.read_csv(get_inspect_tsv(data_dir, "impressions"), sep="\t")
    splits = pd.read_csv(get_inspect_tsv(data_dir, "splits"), sep="\t")

    merged = mapping[["impression_id", "image_id"]].merge(impressions, on="impression_id")
    merged = merged.merge(splits[["impression_id", "split"]], on="impression_id")

    # Filter out broken keys
    merged = merged[~merged["image_id"].isin(BROKEN_KEYS)].copy()

    # Drop duplicate image_ids (multiple volumes per impression)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset="image_id", keep="first")
    n_dupes = n_before - len(merged)
    if n_dupes > 0:
        print(f"Dropped {n_dupes} duplicate image_ids")

    merged = merged.set_index("image_id")

    # Filter out empty impressions
    empty = merged["impressions"].isna() | (merged["impressions"].str.strip() == "")
    n_empty = empty.sum()
    if n_empty > 0:
        print(f"Filtering out {n_empty} reports with empty impressions")
    merged = merged[~empty].copy()

    return merged


class InspectRateOutputLoader:
    """Load Inspect reports and existing RATE outputs for incremental processing."""

    def __init__(self, save_dir, data_dir=None, verbose=True):
        self.save_dir = Path(save_dir)
        self.verbose = verbose

        if data_dir is None:
            data_dir = Path(get_medv_data_dir()) / "public/Inspect"
        self.data_dir = Path(data_dir)

        self.reports_df = load_inspect_reports(self.data_dir)

        if verbose:
            n_total = len(self.reports_df)
            print(f"Loaded {n_total} Inspect reports total after filtering:")
            for split in sorted(self.reports_df["split"].unique()):
                n_split = (self.reports_df["split"] == split).sum()
                print(f"  {split}: {n_split} reports")

    def load_data(self, language: str, bodypart: str, split: str):
        """
        Load Inspect data for a specific split.

        Returns: (rdfsplit, no_comp_imprs_dict, catdict, quesdict, mod_cfg)
        """
        verbose = self.verbose
        assert split in ["train", "valid", "test"], f"Invalid {split=} for Inspect"

        save_dir_lang = Path(self.save_dir)  # English only, no suffix

        is_split = self.reports_df["split"] == split
        rdfsplit = self.reports_df[is_split].copy()
        assert len(rdfsplit) > 0, f"No reports found for {split=}"

        if verbose:
            print(f"Loaded {len(rdfsplit)} reports for split '{split}'")

        reports_index = rdfsplit.index.tolist()
        _, no_comp_imprs_dict, catdict, quesdict, mod_cfg = load_outputs(
            save_dir_lang,
            language,
            bodypart,
            split,
            reports_index,
            verbose=verbose,
            do_validate_findings=False,
        )
        return rdfsplit, no_comp_imprs_dict, catdict, quesdict, mod_cfg


def build_reports_inspect(rdf: pd.DataFrame) -> dict[str, str]:
    """
    Build report text dict for the questions stage.

    Inspect only has impressions (no findings), so the report is just the impression.
    """
    reports_dict = {}
    for idx, row in rdf.iterrows():
        impressions = row["impressions"]
        assert (
            impressions and pd.notna(impressions) and str(impressions).strip()
        ), f"Empty impression for report {idx}"
        reports_dict[idx] = f"Impression: {str(impressions).strip()}"
    return reports_dict
