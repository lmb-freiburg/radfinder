"""
Layout of valid.csv:
    ipdb> reports.iloc[0]
    VolumeName                                               valid_1_a_1.nii.gz
    ClinicalInformation_EN                                           Not given.
    Technique_EN              Non-contrast images were taken in the axial pl...
    Findings_EN               Trachea, both main bronchi are open. Mediastin...
    Impressions_EN             A few millimetric nonspecific nodules and sli...
    Name: 0, dtype: object
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
from radfinder.paths import get_medv_data_dir
from rate.rate_common_utils import load_outputs

CTRATE_DEFAULT_DATA_DIR = get_medv_data_dir() / "public/CT-RATE"
CTRATE_DEFAULT_OUTPUT_DIR = get_medv_data_dir() / "public/CT-RATE/report_structuring/p0rate_en"


def load_ctrate(data_dir: str | Path, split: str = "train", dedup_reports: bool = False):
    assert split in ["train", "valid"], f"Invalid {split=} for CT-RATE"
    image_paths = get_ctrate_image_paths(data_dir, split)
    subset_for_csv = split
    if split == "valid":
        subset_for_csv = "validation"

    text_path = Path(data_dir) / f"dataset/radiology_text_reports/{subset_for_csv}_reports.csv"
    reports = pd.read_csv(text_path)

    base_names = reports["VolumeName"].str.removesuffix(".nii.gz")
    parts = base_names.str.split("_")
    reports["patient_id"] = parts.str[0] + "_" + parts.str[1]
    reports["report_id"] = parts.str[0] + "_" + parts.str[1] + "_" + parts.str[2]
    reports["volume_id"] = base_names

    image_paths_dict = {Path(p).name: p for p in image_paths}
    reports["volume_path"] = reports["VolumeName"].map(image_paths_dict)
    nas = reports["volume_path"].isna().sum()
    if nas > 0:
        print(f"WARNING: {nas} volumes have missing paths")

    if dedup_reports:
        reports = reports.drop_duplicates(subset="report_id", keep="first").copy()

    return reports
    # return curate_data_for_monai(reports)


def curate_data_for_monai(reports: pd.DataFrame) -> List[Dict[str, str]]:
    reports_dict = reports.set_index("VolumeName")[
        ["Findings_EN", "Impressions_EN", "volume_path", "patient_id", "report_id", "volume_id"]
    ].to_dict("index")

    data = []
    for volume_name, report_data in reports_dict.items():
        data.append(
            {
                "image": report_data["volume_path"],
                "findings": [report_data["Findings_EN"]],
                "impressions": [report_data["Impressions_EN"]],
                "patient_id": report_data["patient_id"],
                "report_id": report_data["report_id"],
                "volume_id": report_data["volume_id"],
            }
        )
    return data


def get_ctrate_image_paths(data_dir: str | Path, subset) -> List[str]:
    if subset == "valid" or subset == "val" or subset == "validation":
        subset = "valid"

    cache_file = Path(data_dir) / f"image_paths_{subset}.txt"
    if cache_file.is_file():
        image_paths = cache_file.read_text().splitlines()
    else:
        print(f"CT-RATE: Globbing image paths for subset {subset}...")
        image_paths = sorted(Path(data_dir).glob(f"dataset/{subset}_fixed/**/*.nii.gz"))
        if len(image_paths) == 0:
            raise FileNotFoundError(
                f"No image paths found in {data_dir}/dataset/{subset}_fixed/**/*.nii.gz"
            )
        cache_file.write_text("\n".join([p.as_posix() for p in image_paths]))
        print(f"Wrote {len(image_paths)} image paths to {cache_file}")
    image_paths = [Path(p).as_posix() for p in image_paths]
    return image_paths


class CTRateRateOutputLoader:
    """
    Note that this loads data on report level and uses report_id as index.
    """

    def __init__(self, save_dir, data_dir, verbose: bool = True):
        self.save_dir = Path(save_dir)
        self.data_dir = Path(data_dir)
        self.verbose = verbose

        # Load CT-RATE reports for all splits (train and validation)
        verbose = self.verbose

        # Load both train and validation splits
        all_reports = []
        for subset in ["train", "valid"]:
            df = load_ctrate(self.data_dir, split=subset, dedup_reports=True)
            n_subset_total = len(df)

            # Filter out reports with empty findings per split
            empty_reports = df["Findings_EN"].isna() | (df["Findings_EN"].str.strip() == "")
            n_empty = empty_reports.sum()
            if n_empty > 0:
                if verbose:
                    print(
                        f"Filtering out {n_empty} reports with empty findings "
                        f"from {subset} split (out of {n_subset_total} total reports)."
                    )
            df = df[~empty_reports].copy()

            df["split"] = subset
            all_reports.append(df)

        reports_df = pd.concat(all_reports, ignore_index=True)
        reports_df = reports_df.set_index("report_id")

        if verbose:
            n_total = len(reports_df)
            print(f"Loaded {n_total} reports total after filtering:")
            for split in reports_df["split"].unique():
                n_split = (reports_df["split"] == split).sum()
                print(f"  {split}: {n_split} reports")

        self.reports_df = reports_df

    def load_data(self, language: str, bodypart: str, split: str):
        """Load CT-RATE data for a specific bodypart and split.

        Note: CT-RATE doesn't have bodypart filtering - all reports are chest CT.
        The bodypart parameter is kept for API compatibility.
        """
        verbose = self.verbose
        assert split in ["train", "valid"], f"Invalid {split=} for CT-RATE"

        reports_df = self.reports_df
        save_dir_lang = Path(self.save_dir)  # only en for ctrate

        # Filter by split
        is_split = reports_df["split"] == split
        rdfsplit = reports_df[is_split].copy()
        assert len(rdfsplit) > 0, f"No reports found for {bodypart=} {split=}"

        if verbose:
            print(f"Loaded {len(rdfsplit)} reports for split '{split}'")

        reports_index = rdfsplit.index.tolist()
        no_comp_finds_dict, no_comp_imprs_dict, catdict, quesdict, mod_cfg = load_outputs(
            save_dir_lang, language, bodypart, split, reports_index, verbose=verbose
        )
        return rdfsplit, no_comp_finds_dict, no_comp_imprs_dict, catdict, quesdict, mod_cfg


def build_reports_ctrate(rdf):
    """
    Build reports from CT-RATE data structure.

    CT-RATE reports have:
    - ClinicalInformation_EN: Clinical context
    - Technique_EN: Imaging technique details
    - Findings_EN: Radiology findings
    - Impressions_EN: Final impressions/conclusions

    We structure them similar to the format expected by the RATE authors.
    """
    reports_dict = {}  # report_id: text

    for idx, row in rdf.iterrows():
        parts = []

        # Clinical Information
        if pd.notna(row["ClinicalInformation_EN"]) and row["ClinicalInformation_EN"].strip():
            clinical_info = clean_report_ctrate(row["ClinicalInformation_EN"])
            if clinical_info.lower() not in ["not given", "not given.", "none", "none."]:
                parts.append(f"Clinical Information: {clinical_info}")

        # Technique
        if pd.notna(row["Technique_EN"]) and row["Technique_EN"].strip():
            technique = clean_report_ctrate(row["Technique_EN"])
            parts.append(f"Technique: {technique}")

        # Findings (main section)
        if pd.notna(row["Findings_EN"]) and row["Findings_EN"].strip():
            findings = clean_report_ctrate(row["Findings_EN"])
            parts.append(f"Findings: {findings}")

        # Impressions
        if pd.notna(row["Impressions_EN"]) and row["Impressions_EN"].strip():
            impressions = clean_report_ctrate(row["Impressions_EN"])
            parts.append(f"Impression: {impressions}")

        assert len(parts) > 0, (
            f"No parts found for report_id {row['report_id']} " f"row {row.to_dict()}"
        )

        reports_dict[row.name] = "\n\n".join(parts)

    return reports_dict


def clean_report_ctrate(content: str):
    # reports have been cleaned already by authors
    return content
