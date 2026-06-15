import math
import re
from pathlib import Path
from typing import Callable

import pandas as pd
from monai.data import Dataset
from radfinder.data.organ_texts import load_no_comparisons_texts, load_organ_texts
from radfinder.paths import get_medv_data_dir
from radfinder.utils.logging_utils import log_debug, log_warning
from radfinder.utils.misc import simple_decap


def split_merlin_report(text: str) -> tuple[str, str]:
    """Split Merlin's combined report text into (findings, impressions).

    The public Merlin ``reports_final.xlsx`` has a single "Findings" column that
    contains the full report body (findings + impressions concatenated).
    Split it into findings and impressions here.

    Strategy (covers 99.5% of reports):
      1. Explicit ``IMPRESSION(S):`` header (~80%).
      2. Numbered conclusions (``1.`` / ``1,`` followed by ``2.``) at the end (~19.5%).
      3. Remaining ~0.5% are left unsplit (no clear boundary).
    """
    # 1) Explicit IMPRESSION(S): header
    match = re.search(r"\s+IMPRESSIONS?\s*:", text, re.IGNORECASE)
    if match:
        return text[: match.start()].strip(), text[match.end() :].strip()

    # 2) Numbered conclusions at the end (e.g. "\n1.  No evidence of ...")
    #    Search backwards for the last "1." or "1," that starts a numbered list
    #    (confirmed by a "2." / "2," appearing later).
    starts = list(re.finditer(r"(?:\n|   )\s*1[.,]\s+", text))
    for m in reversed(starts):
        remainder = text[m.end() :]
        if re.search(r"(?:\n|   )\s*2[.,]\s+", remainder):
            return text[: m.start()].strip(), text[m.start() :].strip()
        # Single-item conclusion near the end of the report
        if m.start() > len(text) * 0.6:
            return text[: m.start()].strip(), text[m.start() :].strip()

    return text.strip(), ""


class MerlinDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path | None = None,
        transform: Callable = None,
        split: str = "train",
        max_datapoints: int | None = None,
        data_fraction: float = 1.0,
        key_subset: list[str] | None = None,
        include_reports: bool = False,
        add_slices: bool = True,
    ):
        if data_dir is None:
            data_dir = Path(get_medv_data_dir()) / "public/Merlin"
        data_dir = Path(data_dir)
        dataset_label = f"dataset=merlin split={split}"

        image_paths = get_merlin_image_paths(data_dir, split)
        keys2paths = {self.get_datapoint_key_from_scan_path(p): p for p in image_paths}
        keys = list(keys2paths.keys())

        if key_subset is not None:
            key_subset_set = set(key_subset)
            keys = [k for k in keys if k in key_subset_set]
            log_debug(f"[{dataset_label}] Filtered to {len(keys)=} from {len(key_subset)=}")
        image_paths = [keys2paths[k] for k in keys]

        if include_reports:
            reports = pd.read_excel(data_dir / "reports_final.xlsx")
            reports_dict = {}
            for _, row in reports.iterrows():
                study_id = row["study id"]
                findings, impressions = split_merlin_report(str(row["Findings"]))
                findings = findings.removeprefix("FINDINGS: ")
                findings = simple_decap(findings)
                impressions = impressions.removeprefix("IMPRESSION: ")
                impressions = simple_decap(impressions)
                reports_dict[study_id] = {"findings": findings, "impressions": impressions}

            data = []
            for image_path in image_paths:
                key = self.get_datapoint_key_from_scan_path(image_path)
                report = reports_dict[key]
                item = {
                    "image": image_path,
                    "scan_key": self.get_datapoint_key_from_scan_path(image_path),
                    "findings": [report["findings"]],
                    "impressions": [report["impressions"]],
                }
                if add_slices:
                    item["slices"] = {}
                data.append(item)

            organ_text_dir = data_dir / "report_structuring" / "p0rate_en"
            if organ_text_dir.is_dir():
                organ_texts = load_organ_texts(organ_text_dir.as_posix(), split)
                n_with_organ = 0
                for item in data:
                    key = self.get_datapoint_key_from_scan_path(item["image"])
                    ot = organ_texts.get(key, "")
                    item["organ_text"] = ot
                    if ot:
                        n_with_organ += 1
                pct = 100 * n_with_organ / len(data)
                log_debug(
                    f"[{dataset_label}] Organ texts: {n_with_organ}/{len(data)} scans ({pct:.1f}%)"
                )
                assert pct > 90, (
                    f"Merlin organ texts: only {n_with_organ}/{len(data)} ({pct:.1f}%) matched. "
                    f"Sample scan key: {self.get_datapoint_key_from_scan_path(data[0]['image'])}, "
                    f"sample organ_text key: {next(iter(organ_texts)) if organ_texts else 'EMPTY DICT'}"
                )

                no_comp_finds, no_comp_imprs = load_no_comparisons_texts(
                    organ_text_dir.as_posix(), split
                )
                n_with_no_comp = 0
                for item in data:
                    key = self.get_datapoint_key_from_scan_path(item["image"])
                    item["no_comp_findings"] = no_comp_finds.get(key, "")
                    item["no_comp_impressions"] = no_comp_imprs.get(key, "")
                    if item["no_comp_findings"]:
                        n_with_no_comp += 1
                pct_nc = 100 * n_with_no_comp / len(data)
                log_debug(
                    f"[{dataset_label}] No-comp texts: {n_with_no_comp}/{len(data)} scans "
                    f"({pct_nc:.1f}%)"
                )
                assert pct_nc > 90, (
                    f"Merlin no_comp_findings: only {n_with_no_comp}/{len(data)} ({pct_nc:.1f}%) matched. "
                    f"Sample scan key: {self.get_datapoint_key_from_scan_path(data[0]['image'])}, "
                    f"sample no_comp key: {next(iter(no_comp_finds)) if no_comp_finds else 'EMPTY DICT'}"
                )
            else:
                log_warning(f"[{dataset_label}] Merlin organ text dir not found: {organ_text_dir}")
                for item in data:
                    item["organ_text"] = ""
                    item["no_comp_findings"] = ""
                    item["no_comp_impressions"] = ""
        else:
            data = [
                {"image": p, "scan_key": self.get_datapoint_key_from_scan_path(p)}
                for p in image_paths
            ]

        lim = None
        if max_datapoints is not None and max_datapoints < len(data):
            log_debug(f"[{dataset_label}] Limiting to max_datapoints={max_datapoints:_d}")
            lim = max_datapoints
        if 0.0 < data_fraction < 1.0:
            lim_frac = int(math.ceil(len(data) * data_fraction))
            if lim is None or lim_frac < lim:
                lim = lim_frac
            log_debug(f"[{dataset_label}] Using fraction {data_fraction}: {lim:_d}")
        if lim is not None:
            data = data[:lim]

        super().__init__(data=data, transform=transform)

    @classmethod
    def get_datapoint_key_from_scan_path(cls, full_path: str) -> str:
        return Path(full_path).name.removesuffix(".nii.gz")

    @classmethod
    def get_feature_subdir_from_datapoint_key(cls, datapoint_key: str) -> Path:
        return Path(datapoint_key)


def get_merlin_image_paths(data_dir: Path, split: str) -> list[str]:
    cache_file = data_dir / f"image_paths_{split}.txt"
    if cache_file.is_file():
        return cache_file.read_text().splitlines()

    reports = pd.read_excel(data_dir / "reports_final.xlsx")
    split_ids = set(reports[reports["Split"] == split]["study id"])
    all_paths = sorted(data_dir.glob("merlin_data/*.nii.gz"))
    image_paths = [p.as_posix() for p in all_paths if p.name.removesuffix(".nii.gz") in split_ids]
    assert len(image_paths) > 0, f"No image paths found for {split=} in {data_dir}"
    cache_file.write_text("\n".join(image_paths))
    log_debug(
        f"[dataset=merlin split={split}] Wrote {len(image_paths)} image paths to {cache_file}"
    )
    return image_paths
