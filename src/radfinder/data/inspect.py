import math
from pathlib import Path
from typing import Callable

import pandas as pd
from monai.data import Dataset
from radfinder.data.organ_texts import load_no_comparisons_texts, load_organ_texts
from radfinder.paths import get_medv_data_dir
from radfinder.utils.logging_utils import log_debug, log_warning
from radfinder.utils.misc import simple_decap

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


class InspectDataset(Dataset):
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
            data_dir = Path(get_medv_data_dir()) / "public/Inspect"
        data_dir = Path(data_dir)
        split = "valid" if split == "val" else split
        dataset_label = f"dataset=inspect split={split}"

        image_paths = get_inspect_image_paths(data_dir, split)
        keys2paths = {self.get_datapoint_key_from_scan_path(p): p for p in image_paths}
        keys = [k for k in keys2paths if k not in BROKEN_KEYS]

        if key_subset is not None:
            key_subset_set = set(key_subset)
            keys = [k for k in keys if k in key_subset_set]
            log_debug(f"[{dataset_label}] Filtered to {len(keys)=} from {len(key_subset)=}")
        image_paths = [keys2paths[k] for k in keys]

        if include_reports:
            impressions_dict = _build_inspect_impressions_dict(data_dir)
            data = []
            for image_path in image_paths:
                key = self.get_datapoint_key_from_scan_path(image_path)
                impression = impressions_dict.get(key, "")
                item = {
                    "image": image_path,
                    "scan_key": self.get_datapoint_key_from_scan_path(image_path),
                    "findings": [],
                    "impressions": [impression],
                }
                if add_slices:
                    item["slices"] = {}
                data.append(item)

            organ_text_dir = data_dir / "report_structuring" / "p0rate_en"
            if not organ_text_dir.is_dir():
                log_warning(
                    f"RATE pipeline output for Inspect not found at {organ_text_dir}. "
                    f"Will not load per-organ texts and comparison-removed reports."
                )
            else:
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
                    f"Inspect organ texts: only {n_with_organ}/{len(data)} ({pct:.1f}%) matched. "
                    f"Sample scan key: {self.get_datapoint_key_from_scan_path(data[0]['image'])}, "
                    f"sample organ_text key: {next(iter(organ_texts)) if organ_texts else 'EMPTY DICT'}"
                )
                no_comp_finds, no_comp_imprs = load_no_comparisons_texts(
                    organ_text_dir.as_posix(), split
                )
                n_with_no_comp = 0
                for item in data:
                    key = self.get_datapoint_key_from_scan_path(item["image"])
                    item["no_comp_findings"] = ""
                    item["no_comp_impressions"] = no_comp_imprs.get(key, "")
                    if item["no_comp_impressions"]:
                        n_with_no_comp += 1
                pct_nc = 100 * n_with_no_comp / len(data)
                log_debug(
                    f"[{dataset_label}] No-comp texts: {n_with_no_comp}/{len(data)} scans ({pct_nc:.1f}%)"
                )
                assert pct_nc > 90, (
                    f"Inspect no_comp_impressions: only {n_with_no_comp}/{len(data)} ({pct_nc:.1f}%) matched. "
                    f"Sample scan key: {self.get_datapoint_key_from_scan_path(data[0]['image'])}, "
                    f"sample no_comp key: {next(iter(no_comp_imprs)) if no_comp_imprs else 'EMPTY DICT'}"
                )
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


def _build_inspect_impressions_dict(data_dir: Path) -> dict[str, str]:
    """Build image_id -> impression text lookup by joining study_mapping and impressions TSVs."""
    mapping = pd.read_csv(get_inspect_tsv(data_dir, "study_mapping"), sep="\t")
    impressions = pd.read_csv(get_inspect_tsv(data_dir, "impressions"), sep="\t")
    merged = mapping[["impression_id", "image_id"]].merge(impressions, on="impression_id")
    ret_dict = {}
    for k, v in zip(merged["image_id"], merged["impressions"].fillna("")):
        v = v.removeprefix("IMPRESSION: ")
        v = simple_decap(v)
        ret_dict[k] = v
    return ret_dict


def get_inspect_image_paths(data_dir: Path, split: str) -> list[str]:
    cache_file = data_dir / f"image_paths_{split}.txt"
    if cache_file.is_file():
        return cache_file.read_text().splitlines()

    mapping = pd.read_csv(get_inspect_tsv(data_dir, "study_mapping"), sep="\t")
    splits = pd.read_csv(get_inspect_tsv(data_dir, "splits"), sep="\t")
    merged = mapping.merge(splits[["impression_id", "split"]], on="impression_id")
    merged = merged[merged["split"] == split]

    ctpa_dir = data_dir / "full" / "CTPA"
    image_paths = []
    for image_id in sorted(merged["image_id"]):
        p = ctpa_dir / f"{image_id}.nii.gz"
        if p.is_file():
            image_paths.append(p.as_posix())
    assert len(image_paths) > 0, f"No image paths found for {split=} in {data_dir}"
    cache_file.write_text("\n".join(image_paths))
    log_debug(
        f"[dataset=inspect split={split}] Wrote {len(image_paths)} image paths to {cache_file}"
    )
    return image_paths
