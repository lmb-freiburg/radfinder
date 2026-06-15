from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pandas as pd
from loguru import logger
from radfinder.paths import get_medv_output_dir


class DatasetConfig(TypedDict):
    short_name: str
    metrics: dict[str, str]


RETRIEVAL_METRICS: dict[str, str] = {
    "t2i_r1": "r1",
    "t2i_r5": "r5",
    "t2i_r10": "r10",
    "t2i_r50": "r50",
    "t2i_r100": "r100",
    "t2i_meanr": "mr",
    "t2i_medr": "medr",
    "loss_nonaccum": "loss",
}

BINARY_ZS_METRICS: dict[str, str] = {
    "mean_auroc": "auc",
    "mean_prec": "prec",
    "mean_sens": "sens",
    "mean_spec": "spec",
    "mean_f1": "f1",
    "mean_acc": "acc",
}

LOCALIZATION_METRICS: dict[str, str] = {
    "loc_mae_mm": "mae",
    "loc_median_mm": "medae",
    "loc_acc_exact": "exact",
    "loc_acc_within_12mm": "a12",
    "loc_acc_within_24mm": "a24",
}

LOCAL_RETRIEVAL_METRICS: dict[str, str] = {
    "local_t2i_r50": "r50",
    "local_t2i_medr": "medr",
}

V2V_RETRIEVAL_METRICS: dict[str, str] = {
    "vol_map5": "map5",
    "vol_map10": "map10",
    "vol_map50": "map50",
}

POOL_RETRIEVAL_METRICS: dict[str, str] = {
    "find_pool32_r1": "f32r1",
    "find_pool64_r1": "f64r1",
    "find_pool128_r1": "f128r1",
    "find_pool32_r8": "f32r8",
    "find_pool64_r8": "f64r8",
    "find_pool128_r8": "f128r8",
    "impr_pool32_r1": "i32r1",
    "impr_pool64_r1": "i64r1",
    "impr_pool128_r1": "i128r1",
    "impr_pool32_r8": "i32r8",
    "impr_pool64_r8": "i64r8",
    "impr_pool128_r8": "i128r8",
    "full_pool32_r1": "fr32r1",
    "full_pool64_r1": "fr64r1",
    "full_pool128_r1": "fr128r1",
    "full_pool32_r8": "fr32r8",
    "full_pool64_r8": "fr64r8",
    "full_pool128_r8": "fr128r8",
}

DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "ctrate_val": {"short_name": "crv", "metrics": RETRIEVAL_METRICS},
    "ctrate_valdup": {"short_name": "cvd", "metrics": RETRIEVAL_METRICS},
    "ctrate_traindev": {"short_name": "ctd", "metrics": RETRIEVAL_METRICS},
    "bz_ctrate_val": {"short_name": "bzcm", "metrics": BINARY_ZS_METRICS},
    "bz_radchestct_standard": {"short_name": "bzrm", "metrics": BINARY_ZS_METRICS},
    "vr_ctrate_val": {"short_name": "v2v", "metrics": V2V_RETRIEVAL_METRICS},
    "merlin_pool_val": {"short_name": "mpv", "metrics": POOL_RETRIEVAL_METRICS},
    "merlin_pool_test": {"short_name": "mpt", "metrics": POOL_RETRIEVAL_METRICS},
}

SHORT_NAME_ORDER: list[str] = []
SHORT_NAME_CONFIGS: dict[str, DatasetConfig] = {}
for _dataset_config in DATASET_CONFIGS.values():
    _short_name = _dataset_config["short_name"]
    if _short_name not in SHORT_NAME_ORDER:
        SHORT_NAME_ORDER.append(_short_name)
        SHORT_NAME_CONFIGS[_short_name] = _dataset_config

METRIC_GROUPS: dict[str, list[str]] = {
    "default": [
        # Columns of RadFinder paper, retrieval table
        "cvd-r10",
        "v2v-map5",
        "bzcm-auc",
        "bzrm-auc",
        "mpt-f128r1",
        "mpt-i128r1",
    ],
    "all": [
        f"{sn}-{msn}"
        for sn in SHORT_NAME_ORDER
        for msn in SHORT_NAME_CONFIGS[sn]["metrics"].values()
    ],
}
# Guard against drift: every token in every group must be a real "{short}-{metric}" pair.
_VALID_METRIC_TOKENS = set(METRIC_GROUPS["all"])
for _group_name, _tokens in METRIC_GROUPS.items():
    _unknown = [t for t in _tokens if t not in _VALID_METRIC_TOKENS]
    assert not _unknown, f"Unknown metric tokens in METRIC_GROUPS['{_group_name}']: {_unknown}"
METRIC_FORMATS: dict[str, tuple[float, int]] = {
    "r1": (100.0, 2),
    "r5": (100.0, 2),
    "r10": (100.0, 2),
    "r50": (100.0, 2),
    "r100": (100.0, 2),
    "mr": (1.0, 2),
    "medr": (1.0, 2),
    "loss": (1.0, 4),
    "auc": (100.0, 2),
    "prec": (100.0, 2),
    "sens": (100.0, 2),
    "spec": (100.0, 2),
    "f1": (100.0, 2),
    "acc": (100.0, 2),
    "mae": (1.0, 1),
    "medae": (1.0, 1),
    "exact": (100.0, 1),
    "a12": (100.0, 1),
    "a24": (100.0, 1),
    "map5": (100.0, 2),
    "map10": (100.0, 2),
    "map50": (100.0, 2),
}
for _pr_msn in POOL_RETRIEVAL_METRICS.values():
    METRIC_FORMATS.setdefault(_pr_msn, (100.0, 2))
METRIC_DEFAULTS: dict[str, float] = {
    "t2i_r50": -999.0,
    "t2i_mr": 999.0,
    "t2i_medr": 999.0,
    "mean_auroc": -999.0,
    "mean_prec": -999.0,
    "mean_sens": -999.0,
    "mean_spec": -999.0,
    "mean_f1": -999.0,
    "mean_acc": -999.0,
    "loc_mae_mm": 999.0,
    "loc_median_mm": 999.0,
    "loc_acc_exact": -999.0,
    "loc_acc_within_12mm": -999.0,
    "loc_acc_within_24mm": -999.0,
    "local_t2i_r50": -999.0,
    "local_t2i_medr": 999.0,
    "vol_map5": -999.0,
    "vol_map10": -999.0,
    "vol_map50": -999.0,
}
for _pr_mk in POOL_RETRIEVAL_METRICS:
    METRIC_DEFAULTS.setdefault(_pr_mk, -999.0)
METRIC_KEYS_TO_LOAD = (
    list(RETRIEVAL_METRICS.keys())
    + list(BINARY_ZS_METRICS.keys())
    + list(LOCALIZATION_METRICS.keys())
    + list(LOCAL_RETRIEVAL_METRICS.keys())
    + list(V2V_RETRIEVAL_METRICS.keys())
    + list(POOL_RETRIEVAL_METRICS.keys())
    + ["t2i_medr", "n"]
)
BEST_EPOCH_SHORTNAME = "ctd"


def describe_metric_columns(columns: list[str]) -> list[tuple[str, str, str]]:
    """Resolve each '<short>-<metric>' column to its full meaning.

    Returns (column, dataset names, full metric key) per metric column, where the
    dataset names are all candidates mapped to that short name (the displayed value
    comes from whichever candidate is present for a given experiment).
    """
    described: list[tuple[str, str, str]] = []
    for column in columns:
        if "-" not in column:
            continue
        short_name, metric_short = column.split("-", 1)
        datasets = [
            name for name, cfg in DATASET_CONFIGS.items() if cfg["short_name"] == short_name
        ]
        metric_key = ""
        if datasets:
            for key, short in DATASET_CONFIGS[datasets[0]]["metrics"].items():
                if short == metric_short:
                    metric_key = key
                    break
        described.append((column, ", ".join(datasets), metric_key))
    return described


def get_results_dataframe(
    subfolders_raw: str, metric_group: str = "default", all_epochs: bool = False
) -> pd.DataFrame:
    subfolders = [s.strip() for s in subfolders_raw.split(",") if s.strip()]
    assert len(subfolders) > 0, "No valid subfolders provided"
    val_output_dirs: list[Path] = []
    for subfolder in subfolders:
        val_output_dirs.extend(find_val_output_dirs(subfolder))
    assert len(val_output_dirs) > 0, f"No val_output directories found in {subfolders}"
    logger.info(f"Found {len(val_output_dirs)} val_output directories")

    table_rows: list[dict[str, object]] = []
    for val_output_dir in val_output_dirs:
        exp_dir = val_output_dir.parent
        source_folder = val_output_dir.relative_to(get_medv_output_dir()).parts[0]
        finish_date = _get_finish_date(val_output_dir)
        rows = load_experiment_results(val_output_dir)
        if rows is None:
            continue
        rows = summarize_experiment(
            exp_dir, rows, source_folder, metric_group, finish_date, all_epochs
        )
        if rows is None:
            continue
        table_rows.extend(rows)

    return pd.DataFrame(table_rows)


def load_single_experiment(
    subfolder: str,
    experiment_name: str,
    metric_group: str = "all",
    all_epochs: bool = False,
) -> list[dict[str, object]] | None:
    val_output_dir = get_medv_output_dir() / subfolder / experiment_name / "val_output"
    if not val_output_dir.is_dir():
        logger.warning(f"No val_output dir: {val_output_dir}")
        return None
    finish_date = _get_finish_date(val_output_dir)
    rows = load_experiment_results(val_output_dir)
    if rows is None:
        return None
    return summarize_experiment(
        val_output_dir.parent, rows, subfolder, metric_group, finish_date, all_epochs
    )


def get_best_epoch_metrics(
    subfolder: str,
    experiment_name: str,
    epoch: int | None = None,
) -> dict[str, float] | None:
    rows = load_single_experiment(
        subfolder, experiment_name, metric_group="all", all_epochs=epoch is not None
    )
    if rows is None:
        return None
    if epoch is not None:
        matching = [r for r in rows if r.get("epoch") == epoch]
        if not matching:
            logger.warning(f"Epoch {epoch} not found for {subfolder}/{experiment_name}")
            return None
        row = matching[0]
        return {k: v for k, v in row.items() if isinstance(v, (int, float))}
    last_rows = [r for r in rows if r.get("label") == "last"]
    if not last_rows:
        # Fall back to best if no "last" row (happens when best == last)
        last_rows = [r for r in rows if r.get("label") == "best"]
    if not last_rows:
        return None
    row = last_rows[0]
    return {k: v for k, v in row.items() if isinstance(v, (int, float))}


def find_val_output_dirs(subfolder: str) -> list[Path]:
    base_dir = get_medv_output_dir() / subfolder
    assert base_dir.is_dir(), f"Directory does not exist: {base_dir}"
    return sorted([p for p in base_dir.rglob("val_output") if p.is_dir()])


def _get_finish_date(val_output_dir: Path) -> str:
    """Get the modification time of the newest file in val_output as 'YY-MM-DD HH:MM'."""
    json_paths = [
        p
        for p in val_output_dir.glob("epoch_*.json")
        if "_aux" not in p.stem and "_bootstrap" not in p.stem
    ]
    if not json_paths:
        return ""
    last_mtime = max(p.stat().st_mtime for p in json_paths)
    return datetime.fromtimestamp(last_mtime).strftime("%y-%m-%d %H:%M")


def load_experiment_results(val_output_dir: Path) -> list[dict[str, object]] | None:
    json_paths = sorted(
        p
        for p in val_output_dir.glob("epoch_*.json")
        if "_aux" not in p.stem and "_bootstrap" not in p.stem
    )
    if len(json_paths) == 0:
        logger.warning(f"Skip {val_output_dir.as_posix()}: no JSONs")
        return None
    rows: list[dict[str, object]] = []
    metric_keys = METRIC_KEYS_TO_LOAD
    for json_path in json_paths:
        epoch, dataset = _parse_epoch_and_dataset(json_path)
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        is_retrieval = "t2i_r50" in data and "t2i_meanr" in data
        is_binary_zs = "mean_auroc" in data
        is_localization = "loc_mae_mm" in data
        is_local_retrieval = "local_t2i_r50" in data
        is_v2v_retrieval = "vol_map5" in data
        is_pool_retrieval = "find_pool32_r1" in data
        if (
            not is_retrieval
            and not is_binary_zs
            and not is_localization
            and not is_local_retrieval
            and not is_v2v_retrieval
            and not is_pool_retrieval
        ):
            logger.warning(f"Skip {json_path.as_posix()}: missing metrics")
            continue
        row: dict[str, object] = {
            "epoch": epoch,
            "dataset": dataset,
            "file": json_path.name,
        }
        for key in metric_keys:
            if key not in data:
                continue
            row[key] = data[key]
        bootstrap_path = json_path.with_name(json_path.stem + "_bootstrap.json")
        if bootstrap_path.is_file():
            with bootstrap_path.open("r", encoding="utf-8") as bhandle:
                boot_data = json.load(bhandle)
            for bkey, bval in boot_data.items():
                if bkey.endswith("_ci_half"):
                    row[bkey] = bval
        rows.append(row)
    return rows


def summarize_experiment(
    exp_dir: Path,
    rows: list[dict[str, object]],
    source_folder: str,
    metric_group: str = "default",
    finish_date: str = "",
    all_epochs: bool = False,
) -> list[dict[str, object]] | None:
    rows_by_epoch_dataset = {(int(row["epoch"]), str(row["dataset"])): row for row in rows}
    epochs = sorted({int(row["epoch"]) for row in rows})
    datasets = sorted({str(row["dataset"]) for row in rows})
    if len(epochs) == 0:
        logger.warning(f"Skipping {exp_dir.name}: no epochs found")
        return None
    if len(datasets) == 0:
        logger.warning(f"Skipping {exp_dir.name}: no datasets found")
        return None
    dataset_by_short_name = _select_datasets(exp_dir.name, set(datasets))

    best_dataset = dataset_by_short_name.get(BEST_EPOCH_SHORTNAME)
    last_epoch = max(epochs)
    best_rows = (
        [row for row in rows if str(row["dataset"]) == best_dataset]
        if best_dataset is not None
        else []
    )
    if len(best_rows) == 0:
        expected = [
            name
            for name, cfg in DATASET_CONFIGS.items()
            if cfg["short_name"] == BEST_EPOCH_SHORTNAME
        ]
        logger.warning(
            f"{exp_dir.name}: no retrieval results for short_name '{BEST_EPOCH_SHORTNAME}' "
            f"(one of: {expected}) in {exp_dir / 'val_output'} (found: {sorted(datasets)}); "
            f"using last epoch {last_epoch} as best"
        )
        best_epoch = last_epoch
    else:
        best_row = min(best_rows, key=lambda r: float(r["t2i_meanr"]))
        best_epoch = int(best_row["epoch"])

    for dataset in dataset_by_short_name.values():
        if dataset is None:
            continue
        if (best_epoch, dataset) not in rows_by_epoch_dataset:
            logger.warning(
                f"Missing {dataset} results for best epoch {best_epoch} "
                f"in {exp_dir.name}; filling defaults"
            )
        if (last_epoch, dataset) not in rows_by_epoch_dataset:
            logger.warning(
                f"Missing {dataset} results for last epoch {last_epoch} "
                f"in {exp_dir.name}; filling defaults"
            )
    return _build_display_rows(
        source_folder,
        exp_dir.name,
        best_epoch,
        last_epoch,
        epochs,
        rows_by_epoch_dataset,
        dataset_by_short_name,
        metric_group,
        finish_date,
        all_epochs,
    )


def _build_display_rows(
    source_folder: str,
    exp_name: str,
    best_epoch: int,
    last_epoch: int,
    all_epoch_list: list[int],
    rows_by_epoch_dataset: dict[tuple[int, str], dict[str, object]],
    dataset_by_short_name: dict[str, str | None],
    metric_group: str = "default",
    finish_date: str = "",
    all_epochs: bool = False,
) -> list[dict[str, object]]:
    metrics_to_show_set = set(METRIC_GROUPS.get(metric_group, METRIC_GROUPS["default"]))

    def _row_for_epoch(epoch_label: str, epoch: int) -> dict[str, object]:
        row: dict[str, object] = {
            "folder": source_folder,
            "experiment": exp_name,
            "epoch": epoch,
            "label": epoch_label,
            "finished": finish_date,
        }
        for short_name in SHORT_NAME_ORDER:
            config = SHORT_NAME_CONFIGS[short_name]
            dataset = dataset_by_short_name.get(short_name)
            metrics = rows_by_epoch_dataset.get((epoch, dataset), {}) if dataset is not None else {}
            for metric_key, metric_short_name in config["metrics"].items():
                metric_name = f"{short_name}-{metric_short_name}"
                if metric_name not in metrics_to_show_set:
                    continue
                assert (
                    metric_short_name in METRIC_FORMATS
                ), f"Missing format for metric {metric_short_name}"
                metric_value = metrics.get(metric_key)
                if metric_value is None:
                    metric_value = METRIC_DEFAULTS.get(metric_key, 999.0)
                    scale, precision = (1.0, 2)
                else:
                    scale, precision = METRIC_FORMATS[metric_short_name]
                row[metric_name] = round(float(metric_value) * scale, precision)
        return row

    if all_epochs:
        rows = []
        for epoch in all_epoch_list:
            label = str(epoch)
            if epoch == best_epoch:
                label += "*"
            rows.append(_row_for_epoch(label, epoch))
    else:
        rows = [_row_for_epoch("best", best_epoch)]
        if last_epoch != best_epoch:
            rows.append(_row_for_epoch("last", last_epoch))
    return rows


def _parse_epoch_and_dataset(json_path: Path) -> tuple[int, str]:
    stem = json_path.stem
    parts = stem.split("_", 2)
    assert len(parts) == 3, f"Unexpected val_output filename: {json_path.name}"
    assert parts[0] == "epoch", f"Unexpected val_output filename: {json_path.name}"
    epoch = int(parts[1])
    dataset = parts[2]
    return epoch, dataset


def _select_datasets(exp_name: str, datasets_in_rows: set[str]) -> dict[str, str | None]:
    datasets_by_short_name: dict[str, list[str]] = {}
    for dataset_name, config in DATASET_CONFIGS.items():
        short_name = config["short_name"]
        if short_name not in datasets_by_short_name:
            datasets_by_short_name[short_name] = []
        datasets_by_short_name[short_name].append(dataset_name)

    dataset_by_short_name: dict[str, str | None] = {}
    for short_name in SHORT_NAME_ORDER:
        candidates = datasets_by_short_name[short_name]
        selected = next(
            (dataset for dataset in candidates if dataset in datasets_in_rows),
            None,
        )
        if selected is None:
            logger.warning(
                f"Missing dataset for {exp_name}: no dataset found for {short_name} "
                f"(candidates={candidates}, datasets={sorted(datasets_in_rows)})"
            )
            dataset_by_short_name[short_name] = None
        else:
            dataset_by_short_name[short_name] = selected
    return dataset_by_short_name
