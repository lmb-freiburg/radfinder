"""
Merlin dataset utilities for the RATE report structuring pipeline.

Merlin is a dataset of ~25k abdomen CT scans from Stanford.
Reports come from reports_final.xlsx with a combined Findings column
that contains both findings and impressions (~79.4% have an explicit IMPRESSION(S): header).
"""

import re
from pathlib import Path

import pandas as pd
from radfinder.paths import get_medv_data_dir
from rate.rate_common_utils import load_outputs

MERLIN_DEFAULT_DATA_DIR = get_medv_data_dir() / "public/Merlin"
MERLIN_DEFAULT_OUTPUT_DIR = get_medv_data_dir() / "public/Merlin/report_structuring/p0rate_en"


def split_merlin_report(text: str) -> tuple[str, str]:
    """
    Split Merlin's combined report text into (findings, impressions).

    ~79.4% of reports have an explicit IMPRESSION(S): header. For the rest,
    the entire text is returned as findings with empty impressions.
    """
    match = re.search(r"\s+IMPRESSIONS?\s*:", text, re.IGNORECASE)
    if match:
        return text[: match.start()].strip(), text[match.end() :].strip()
    return text.strip(), ""


def load_merlin_reports(data_dir: Path) -> pd.DataFrame:
    """Load Merlin reports from Excel and split into findings/impressions."""
    reports = pd.read_excel(data_dir / "reports_final.xlsx")

    findings_list = []
    impressions_list = []
    for _, row in reports.iterrows():
        text = row["Findings"]
        if pd.isna(text) or str(text).strip() == "" or str(text).strip().lower() == "nan":
            findings_list.append("")
            impressions_list.append("")
        else:
            findings, impressions = split_merlin_report(str(text))
            findings_list.append(findings)
            impressions_list.append(impressions)

    reports["findings"] = findings_list
    reports["impressions"] = impressions_list
    reports["split"] = reports["Split"].str.lower()
    reports = reports.set_index("study id")
    reports.index = reports.index.astype(str)

    # Filter out rows with empty findings
    empty = reports["findings"].isna() | (reports["findings"].str.strip() == "")
    n_empty = empty.sum()
    if n_empty > 0:
        print(f"Filtering out {n_empty} reports with empty findings")
    reports = reports[~empty].copy()

    return reports


class MerlinRateOutputLoader:
    """Load Merlin reports and existing RATE outputs for incremental processing."""

    def __init__(self, save_dir, data_dir=None, verbose=True):
        self.save_dir = Path(save_dir)
        self.verbose = verbose

        if data_dir is None:
            data_dir = Path(get_medv_data_dir()) / "public/Merlin"
        self.data_dir = Path(data_dir)

        self.reports_df = load_merlin_reports(self.data_dir)

        if verbose:
            n_total = len(self.reports_df)
            print(f"Loaded {n_total} Merlin reports total after filtering:")
            for split in sorted(self.reports_df["split"].unique()):
                n_split = (self.reports_df["split"] == split).sum()
                print(f"  {split}: {n_split} reports")

    def load_data(self, language: str, bodypart: str, split: str):
        """
        Load Merlin data for a specific split.

        Returns: (rdfsplit, no_comp_finds_dict, no_comp_imprs_dict, catdict, quesdict, mod_cfg)
        """
        verbose = self.verbose
        assert split in ["train", "val", "test"], f"Invalid {split=} for Merlin"

        save_dir_lang = Path(self.save_dir)  # English only, no suffix

        is_split = self.reports_df["split"] == split
        rdfsplit = self.reports_df[is_split].copy()
        assert len(rdfsplit) > 0, f"No reports found for {split=}"

        if verbose:
            print(f"Loaded {len(rdfsplit)} reports for split '{split}'")

        reports_index = rdfsplit.index.tolist()
        no_comp_finds_dict, no_comp_imprs_dict, catdict, quesdict, mod_cfg = load_outputs(
            save_dir_lang,
            language,
            bodypart,
            split,
            reports_index,
            verbose=verbose,
        )
        return rdfsplit, no_comp_finds_dict, no_comp_imprs_dict, catdict, quesdict, mod_cfg


def build_reports_merlin(rdf: pd.DataFrame) -> dict[str, str]:
    """
    Build full report text dict for the questions stage.

    Merges findings and impressions (when available) into a single text.
    """
    reports_dict = {}
    for idx, row in rdf.iterrows():
        parts = []
        findings = row["findings"]
        if findings and pd.notna(findings) and str(findings).strip():
            parts.append(f"Findings: {str(findings).strip()}")
        impressions = row["impressions"]
        if impressions and pd.notna(impressions) and str(impressions).strip():
            parts.append(f"Impression: {str(impressions).strip()}")
        assert len(parts) > 0, f"No parts found for report {idx}"
        reports_dict[idx] = "\n\n".join(parts)
    return reports_dict
