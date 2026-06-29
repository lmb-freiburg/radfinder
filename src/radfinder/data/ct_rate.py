import ast
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
from monai.data import Dataset
from radfinder.data.ct_rate_splits import CTRATE_TRAINDEV_PATIENTS
from radfinder.data.organ_texts import load_no_comparisons_texts, load_organ_texts
from radfinder.paths import get_medv_data_dir
from radfinder.utils.logging_utils import log_debug, log_info, log_warning

from packg.constclass import Const


class CTRateFilterMode(Const):
    """Single string switch for CT-RATE volume filtering + per-report dedup.

    Format: "<dedup>_<part>" where
      dedup ∈ {dup, first, smallestdeduptext}
        dup               — keep all volumes per report
        first             — keep first volume per report, deduplicate by report key
        smallestdeduptext — keep the volume with smallest in-plane voxel size per
                            report, then drop reports whose Findings_EN text duplicates
                            another report (the COLIPRI paper's CT-RATE eval set)
      part ∈ {all, nohead}
        all       — keep all studies
        nohead    — drop the entire study if any of its volumes is listed in
                    dataset/metadata/no_chest_{train,valid}.txt
    """

    DUP_ALL = "dup_all"
    DUP_NOHEAD = "dup_nohead"
    FIRST_ALL = "first_all"
    FIRST_NOHEAD = "first_nohead"
    SMALLESTDEDUPTEXT_ALL = "smallestdeduptext_all"
    SMALLESTDEDUPTEXT_NOHEAD = "smallestdeduptext_nohead"

    @classmethod
    def parse(cls, value: str) -> tuple[str, str]:
        values = cls.values_list()
        if value not in values:
            raise ValueError(
                f"Invalid CTRateFilterMode={value!r}; expected one of {sorted(values)}"
            )
        dedup, part = value.split("_", maxsplit=1)
        return dedup, part


BROKEN_PATHS = {
    "train_11755_a_3": "Monai transform failed with error: cause[0]: LinAlgError: SVD did not converge cause[1]: RuntimeError: applying transform <monai.transforms.spatial.dictionary.Orientationd object at 0x7869247b7830>",
    "train_11755_a_4": "Monai transform failed with error: cause[0]: LinAlgError: SVD did not converge cause[1]: RuntimeError: applying transform <monai.transforms.spatial.dictionary.Orientationd object at 0x7869247b7830>",
    "train_1267_a_4": "Monai transform failed with error: cause[0]: LinAlgError: SVD did not converge cause[1]: RuntimeError: applying transform <monai.transforms.spatial.dictionary.Orientationd object at 0x7869247b7830>",
    # input width error: too small to sample at 0.5x0.5x1.0mm
    "train_3821_a_5": "AssertionError: Input width (123) should be divisible by patch size (16).",
    "train_9792_a_1": "AssertionError: Input width (24) should be divisible by patch size (16).",
    "train_9792_a_2": "AssertionError: Input width (45) should be divisible by patch size (16).",
    "train_9792_a_3": "Remove leftover since we only want to keep _1 scans.",
    "train_9792_a_4": "Remove leftover since we only want to keep _1 scans.",
    "train_11755_a_2.nii.gz": "numpy.linalg.LinAlgError: SVD did not converge",
}


class CTRateDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path | None = None,
        transform: Callable = None,
        split: str = "train",
        max_datapoints: int | None = None,
        data_fraction: float = 1.0,
        key_subset: list[str] | None = None,
        include_reports: bool = False,
        filter_mode: str = CTRateFilterMode.DUP_ALL,
        add_slices: bool = True,
        sample_scan_per_dedup_report: bool = False,
    ):
        if data_dir is None:
            data_dir = Path(get_medv_data_dir()) / "public/CT-RATE"
        data_dir = Path(data_dir)
        split = "valid" if split == "val" else split
        dataset_label = f"dataset=ctrate split={split}"
        dedup_kind, part_kind = CTRateFilterMode.parse(filter_mode)
        if split in {"traindev", "trainnodev"}:
            image_paths = get_ctrate_new_train_splits(data_dir, split)
        else:
            image_paths = get_ctrate_image_paths(data_dir, split)
        keys2paths = {self.get_datapoint_key_from_scan_path(p): p for p in image_paths}
        keys = list(keys2paths.keys())

        if key_subset is not None:
            key_subset = set(key_subset)
            keys = [k for k in keys if k in key_subset]
            log_debug(f"[{dataset_label}] Filtered to {len(keys)=} from {len(key_subset)=}")
        image_paths = [keys2paths[k] for k in keys if k not in BROKEN_PATHS]

        if split in {"valid", "val"}:
            csv_split = "validation"
        elif split in {"traindev", "trainnodev"}:
            csv_split = "train"
        else:
            csv_split = split
        text_path = Path(data_dir) / f"dataset/radiology_text_reports/{csv_split}_reports.csv"
        reports = pd.read_csv(text_path)

        if part_kind == "nohead":
            # Study-level exclusion: drop every volume of a study if any of that study's
            # volumes is listed as non-chest. The paper removes these head/non-chest CTs.
            no_chest = load_no_chest_volumes(data_dir)
            n_before = len(image_paths)
            flagged_keys = {extract_report_key(p) for p in image_paths if Path(p).name in no_chest}
            image_paths = [p for p in image_paths if extract_report_key(p) not in flagged_keys]
            n_dropped = n_before - len(image_paths)
            # On the full split there are always non-chest studies to drop, so 0 dropped means
            # the no_chest basenames don't match the volume naming (format mismatch). A
            # key_subset (e.g. feature extraction over a slice of scans) may legitimately
            # contain none, so only enforce this when running over the whole split.
            assert n_dropped > 0 or key_subset is not None, (
                f"[{dataset_label}] nohead filter dropped 0/{n_before} volumes — format "
                f"mismatch? Sample volume name: "
                f"{Path(image_paths[0]).name if image_paths else '?'}, sample no_chest entry: "
                f"{next(iter(no_chest)) if no_chest else 'EMPTY SET'}"
            )
            log_debug(
                f"[{dataset_label}] nohead filter: dropped {len(flagged_keys)} studies, "
                f"{n_before} → {len(image_paths)} volumes"
            )

        onetomany = None
        if dedup_kind in {"first", "smallestdeduptext"}:
            df = pd.DataFrame({"image_path": image_paths})
            df["report_key"] = df["image_path"].apply(extract_report_key)
            if dedup_kind == "first":
                df_deduplicated = df.drop_duplicates(subset="report_key", keep="first")
            else:
                df["volume_name"] = df["image_path"].apply(lambda p: Path(p).name)
                meta_path = data_dir / f"dataset/metadata/{csv_split}_metadata.csv"
                meta = pd.read_csv(meta_path, usecols=["VolumeName", "XYSpacing"])
                meta["xy"] = meta["XYSpacing"].apply(_parse_xy_spacing)
                df = df.merge(
                    meta[["VolumeName", "xy"]],
                    left_on="volume_name",
                    right_on="VolumeName",
                    how="left",
                )
                missing = df["xy"].isna().sum()
                assert missing == 0, (
                    f"[{dataset_label}] XYSpacing missing for {missing}/{len(df)} volumes — "
                    f"metadata path/format mismatch ({meta_path}). Sample volume_name: "
                    f"{df.loc[df['xy'].isna(), 'volume_name'].iloc[0]}"
                )
                df_deduplicated = df.sort_values(
                    ["report_key", "xy", "volume_name"]
                ).drop_duplicates(subset="report_key", keep="first")
                # After picking one volume per study, collapse studies that share identical
                # Findings_EN report text (different studies can carry the same report).
                findings = reports.set_index("VolumeName")["Findings_EN"].to_dict()
                df_deduplicated["findings_en"] = df_deduplicated["volume_name"].map(findings)
                n_missing_text = df_deduplicated["findings_en"].isna().sum()
                assert n_missing_text == 0, (
                    f"[{dataset_label}] Findings_EN missing for {n_missing_text}/"
                    f"{len(df_deduplicated)} deduplicated volumes ({text_path}). Sample "
                    f"volume_name: "
                    f"{df_deduplicated.loc[df_deduplicated['findings_en'].isna(), 'volume_name'].iloc[0]}"
                )
                df_deduplicated = df_deduplicated.drop_duplicates(
                    subset="findings_en", keep="first"
                )
            image_paths = df_deduplicated["image_path"].tolist()
            log_debug(
                f"[{dataset_label}] Deduplicated ({dedup_kind}): {len(image_paths)} images "
                f"remaining from {df_deduplicated['report_key'].nunique()} unique reports"
            )
            if sample_scan_per_dedup_report:
                onetomany = defaultdict(list)
                for k, v in zip(df["report_key"].tolist(), df["image_path"].tolist()):
                    onetomany[k].append(v)
                onetomany = dict(onetomany)

        reports_dict = reports.set_index("VolumeName")[["Findings_EN", "Impressions_EN"]].to_dict(
            "index"
        )
        data = []
        for image_path in image_paths:
            image_path = Path(image_path)
            volume_name = image_path.name
            report = reports_dict[volume_name]
            scan_key = self.get_datapoint_key_from_scan_path(image_path.as_posix())
            report_key = scan_key.rsplit("_", maxsplit=1)[0]
            item = {
                "image": image_path.as_posix(),
                "scan_key": scan_key,
                "findings": [report["Findings_EN"]],
                "impressions": [report["Impressions_EN"]],
            }
            if add_slices:
                item["slices"] = {}
            if onetomany is not None:
                item["all_images"] = onetomany[report_key]
            data.append(item)
        organ_text_dir = data_dir / "report_structuring" / "p0rate_en"
        if not organ_text_dir.is_dir():
            log_warning(
                f"RATE pipeline output for CT-RATE not found at {organ_text_dir}. "
                f"Will not load per-organ texts and comparison-removed reports."
            )
        else:
            organ_text_split = "valid" if split in {"valid", "val"} else split
            if split in {"traindev", "trainnodev"}:
                organ_text_split = "train"
            organ_texts = load_organ_texts(organ_text_dir.as_posix(), organ_text_split)
            n_with_organ = 0
            for item in data:
                report_key = extract_report_key(item["image"])
                ot = organ_texts.get(report_key, "")
                item["organ_text"] = ot
                if ot:
                    n_with_organ += 1
            pct = 100 * n_with_organ / len(data)
            log_debug(
                f"[{dataset_label}] Organ texts: {n_with_organ}/{len(data)} scans ({pct:.1f}%)"
            )
            assert pct > 90, (
                f"CT-RATE organ texts: only {n_with_organ}/{len(data)} ({pct:.1f}%) matched. "
                f"Sample report_key: {extract_report_key(data[0]['image'])}, "
                f"sample organ_text key: {next(iter(organ_texts)) if organ_texts else 'EMPTY DICT'}"
            )
            no_comp_finds, no_comp_imprs = load_no_comparisons_texts(
                organ_text_dir.as_posix(), organ_text_split
            )
            n_with_no_comp = 0
            for item in data:
                report_key = extract_report_key(item["image"])
                item["no_comp_findings"] = no_comp_finds.get(report_key, "")
                item["no_comp_impressions"] = no_comp_imprs.get(report_key, "")
                if item["no_comp_findings"]:
                    n_with_no_comp += 1
            log_debug(
                f"[{dataset_label}] No-comp texts: {n_with_no_comp}/{len(data)} scans "
                f"({100*n_with_no_comp/len(data):.1f}%)"
            )

        if not include_reports:
            data = [{"image": item["image"], "scan_key": item["scan_key"]} for item in data]
        # make dataset smaller after deduplicating
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
        return Path(full_path).name.removesuffix(".nii.gz").removesuffix(".npz")

    @classmethod
    def get_feature_subdir_from_datapoint_key(cls, datapoint_key: str) -> Path:
        return Path(datapoint_key)


def extract_report_key(image_path: str | Path) -> str:
    return Path(image_path).name.removesuffix(".nii.gz").rsplit("_", maxsplit=1)[0]


def _parse_xy_spacing(xys: str) -> float:
    a, b = ast.literal_eval(xys)
    return float(a)


def get_ctrate_image_paths(data_dir, subset) -> List[str]:
    cache_file = Path(data_dir) / f"image_paths_{subset}.txt"
    if cache_file.is_file():
        image_paths = cache_file.read_text().splitlines()
    else:
        log_info(
            f"Iterating through CT-RATE image paths in {data_dir} for split={subset}, "
            f"might take a while..."
        )
        image_paths = sorted(
            Path(data_dir).glob(os.path.join("dataset", f"{subset}_fixed", "*", "*", "*.nii.gz"))
        )
        if len(image_paths) == 0:
            raise FileNotFoundError(
                f"No image paths found in {data_dir}/dataset/{subset}_fixed/*/*/*.nii.gz"
            )
        cache_file.write_text("\n".join([p.as_posix() for p in image_paths]))
        log_debug(
            f"[dataset=ctrate split={subset}] Wrote {len(image_paths)} image paths to {cache_file}"
        )
    image_paths = [Path(p).as_posix() for p in image_paths]
    return image_paths


def get_ctrate_new_train_splits(data_dir: str | Path, split: str = "traindev") -> list[str]:
    """Patient-disjoint split of CT-RATE train into traindev (held-out probe) and trainnodev.

    traindev holds 1,301 patients = 1,564 studies (matches the dedup'd val-set size).
    Patient list is hardcoded in `ct_rate_splits.py` so the split is reproducible
    without any cache file and is reviewable in git.
    """
    assert split in {"traindev", "trainnodev"}, f"Invalid {split=} for CT-RATE"
    image_paths = get_ctrate_image_paths(data_dir, "train")
    traindev_patients = set(CTRATE_TRAINDEV_PATIENTS)

    def patient_of(p: str) -> str:
        parts = Path(p).name.split("_")
        return f"{parts[0]}_{parts[1]}"

    if split == "traindev":
        out = [p for p in image_paths if patient_of(p) in traindev_patients]
    else:
        out = [p for p in image_paths if patient_of(p) not in traindev_patients]
    assert len(out) > 0, (
        f"[dataset=ctrate split={split}] Patient filter matched 0/{len(image_paths)} paths — "
        f"sample image path: {image_paths[0] if image_paths else '?'}, "
        f"sample traindev patient: {next(iter(traindev_patients))}"
    )
    log_debug(
        f"[dataset=ctrate split={split}] {len(out)}/{len(image_paths)} volumes after patient filter"
    )
    return out
