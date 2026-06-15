import json
import math
from pathlib import Path
from typing import Callable
from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd
from monai.data import Dataset, ImageReader
from radfinder.paths import RADFINDER_REPO_DIR, get_medv_data_dir
from radfinder.utils.logging_utils import log_debug


class NpzReader(ImageReader):
    """
    MONAI-compatible reader for Rad-ChestCT NPZ files.

    NPZ files contain a single 'ct' key with shape (Z, Y, X) in IPL orientation,
    isotropic 0.8mm spacing, int16 HU values clipped to [-1000, 1000].

    The orientation was determined from the Extrema CSV column names:
      axis0: sup_axis0min → inf_axis0max  (Superior → Inferior = I direction)
      axis1: ant_axis1min → pos_axis1max  (Anterior → Posterior = P direction)
      axis2: rig_axis2min → lef_axis2max  (Right → Left = L direction)
    """

    SPACING = 0.8  # mm, isotropic

    def verify_suffix(self, filename) -> bool:
        return str(filename).endswith(".npz")

    def read(self, data, **kwargs):
        # MONAI's LoadImage wraps filenames in a tuple via ensure_tuple
        if isinstance(data, (list, tuple)):
            return [np.load(str(f))["ct"] for f in data]
        return np.load(str(data))["ct"]

    def get_data(self, img):
        # MONAI passes a list from read(); we expect exactly one image
        if isinstance(img, (list, tuple)):
            if len(img) != 1:
                raise ValueError(f"NpzReader expects 1 file per call, got {len(img)}")
            img = img[0]
        header = {
            "affine": self._make_affine(img.shape),
            "spatial_shape": np.array(img.shape),
        }
        return img.astype(np.float32), header

    @classmethod
    def _make_affine(cls, shape):
        """Construct RAS affine for IPL-oriented data at 0.8mm isotropic spacing."""
        s = cls.SPACING
        affine = np.zeros((4, 4), dtype=np.float64)
        # IPL: axis0 → Inferior (-S), axis1 → Posterior (-A), axis2 → Left (-R)
        affine[2, 0] = -s  # S decreases as axis0 (I) increases
        affine[1, 1] = -s  # A decreases as axis1 (P) increases
        affine[0, 2] = -s  # R decreases as axis2 (L) increases
        affine[3, 3] = 1.0
        # Place origin so center of volume maps to world (0, 0, 0)
        center = np.array([(d - 1) / 2.0 for d in shape])
        affine[:3, 3] = -affine[:3, :3] @ center
        return affine


class RadChestCTDataset(Dataset):
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
            data_dir = Path(get_medv_data_dir()) / "public/Rad-ChestCT"
        data_dir = Path(data_dir)
        if split != "all":
            split = "valid" if split == "val" else split
        dataset_label = f"dataset=radchestct split={split}"

        image_paths = get_radchestct_image_paths(data_dir, split)
        keys2paths = {self.get_datapoint_key_from_scan_path(p): p for p in image_paths}
        keys = list(keys2paths.keys())

        if key_subset is not None:
            key_subset_set = set(key_subset)
            keys = [k for k in keys if k in key_subset_set]
            log_debug(f"[{dataset_label}] Filtered to {len(keys)=} from {len(key_subset)=}")
        image_paths = [keys2paths[k] for k in keys]

        data = [
            {"image": p, "scan_key": self.get_datapoint_key_from_scan_path(p)} for p in image_paths
        ]
        if include_reports:
            for item in data:
                item["findings"] = ""
                item["impressions"] = [""]
                if add_slices:
                    item["slices"] = {}

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
        return Path(full_path).name.removesuffix(".npz")

    @classmethod
    def get_feature_subdir_from_datapoint_key(cls, datapoint_key: str) -> Path:
        return Path(datapoint_key)


LABEL_CSV_MAP = {
    "train": "imgtrain_Abnormality_and_Location_Labels.csv",
    "valid": "imgvalid_Abnormality_and_Location_Labels.csv",
    "test": "imgtest_Abnormality_and_Location_Labels.csv",
}


def load_radchestct_labels(
    data_dir: Path | None = None,
    split: str = "train",
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load native 4368-dim binary labels for Rad-ChestCT.

    Args:
        data_dir: Path to Rad-ChestCT root. If None, uses default.
        split: One of "train", "valid"/"val", "test", "all".

    Returns:
        (scan_key → 4368-dim int64 array, column_names list).
    """
    if data_dir is None:
        data_dir = Path(get_medv_data_dir()) / "public/Rad-ChestCT"
    data_dir = Path(data_dir)
    split = "valid" if split == "val" else split

    if split == "all":
        all_labels: dict[str, np.ndarray] = {}
        col_names = None
        for s in ("train", "valid", "test"):
            s_labels, s_cols = load_radchestct_labels(data_dir, s)
            all_labels.update(s_labels)
            col_names = s_cols
        assert col_names is not None
        log_debug(f"[dataset=radchestct split=all] Loaded {len(all_labels)} labels total")
        return all_labels, col_names

    if split not in LABEL_CSV_MAP:
        raise ValueError(f"Unknown split '{split}', expected one of {list(LABEL_CSV_MAP.keys())}")
    csv_path = data_dir / LABEL_CSV_MAP[split]
    if not csv_path.is_file():
        raise FileNotFoundError(f"Rad-ChestCT label CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    assert len(df) > 0, f"Label CSV is empty: {csv_path}"
    df = df.set_index("NoteAcc_DEID")
    column_names = list(df.columns)

    unique_vals = set(df.values.ravel())
    assert unique_vals <= {
        0,
        1,
        0.0,
        1.0,
    }, f"Expected only 0/1 values, got {unique_vals - {0, 1, 0.0, 1.0}}"

    labels_dict: dict[str, np.ndarray] = {}
    for scan_id in df.index:
        labels_dict[str(scan_id)] = df.loc[scan_id].to_numpy().astype(np.int64)

    assert len(labels_dict) > 0, f"No labels loaded from {csv_path}"
    log_debug(
        f"[dataset=radchestct split={split}] Loaded {len(labels_dict)} labels, "
        f"{len(column_names)} columns"
    )
    return labels_dict, column_names


def load_radchestct_to_ctrate18_mapping(variant: str = "extended") -> dict[str, list[str]]:
    """Load the Rad-ChestCT abnormality → CT-RATE 18 pathology mapping JSON.

    Args:
        variant: "standard" (literature-compatible) or "extended" (medically accurate).
    """
    suffix = "_standard" if variant == "standard" else ""
    mapping_file = (
        RADFINDER_REPO_DIR / f"configs/tasks/binary_zs/radchestct_to_ctrate18{suffix}.json"
    )
    if not mapping_file.is_file():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    with open(mapping_file) as f:
        mapping = json.load(f)
    return mapping


def aggregate_radchestct_to_ctrate18(
    native_labels: np.ndarray,
    column_names: list[str],
    mapping: dict[str, list[str]],
) -> np.ndarray:
    """Aggregate 4368-dim Rad-ChestCT labels to 18-dim CT-RATE pathology labels.

    For each CT-RATE pathology, takes the max across ALL locations for the mapped
    abnormalities. Returns 18-dim int64 array.
    """
    # Circular import: binary_zs_ctrate_task → utils.train_dataloader → data.rad_chestct.
    from radfinder.tasks.binary_zs_ctrate_task import PATHOLOGIES

    # Parse column names: "abnormality*location" → abnormality
    col_abnormalities = [c.split("*")[0] for c in column_names]

    result = np.zeros(len(PATHOLOGIES), dtype=np.int64)
    for i, pathology in enumerate(PATHOLOGIES):
        if pathology not in mapping:
            continue
        abnormality_names = mapping[pathology]
        # Find all column indices where the abnormality matches
        col_indices = [j for j, abn in enumerate(col_abnormalities) if abn in abnormality_names]
        if col_indices:
            result[i] = int(native_labels[col_indices].max())
    return result


def get_radchestct_image_paths(data_dir: Path, split: str) -> list[str]:
    if split == "all":
        all_paths = []
        for s in ("train", "valid", "test"):
            all_paths.extend(get_radchestct_image_paths(data_dir, s))
        log_debug(f"[dataset=radchestct split=all] Found {len(all_paths)} image paths total")
        return all_paths

    cache_file = data_dir / f"image_paths_{split}.txt"
    if cache_file.is_file():
        return cache_file.read_text().splitlines()

    if split not in LABEL_CSV_MAP:
        raise ValueError(f"Unknown split '{split}', expected one of {list(LABEL_CSV_MAP.keys())}")
    label_csv = data_dir / LABEL_CSV_MAP[split]
    if not label_csv.is_file():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")
    ids = sorted(pd.read_csv(label_csv, usecols=["NoteAcc_DEID"])["NoteAcc_DEID"])
    assert len(ids) > 0, f"No scan IDs found in {label_csv}"

    npz_dir = data_dir / "npz"
    image_paths = []
    for scan_id in ids:
        p = npz_dir / f"{scan_id}.npz"
        if p.is_file():
            image_paths.append(p.as_posix())

    assert len(image_paths) > 0, (
        f"No NPZ files found for {split=} in {npz_dir}. "
        f"Expected {len(ids)} labeled scans, 0 found on disk. "
        f"Sample expected ID: {ids[0]}"
    )
    log_debug(
        f"[dataset=radchestct split={split}] {len(image_paths)}/{len(ids)} scans available on disk"
    )

    # Only cache if all expected files are present (rsync might be ongoing)
    if len(image_paths) == len(ids):
        cache_file.write_text("\n".join(image_paths))
        log_debug(
            f"[dataset=radchestct split={split}] Wrote {len(image_paths)} image paths to {cache_file}"
        )

    return image_paths
