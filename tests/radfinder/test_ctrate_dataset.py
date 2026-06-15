from pathlib import Path

import pandas as pd
from radfinder.data import ct_rate
from radfinder.data.ct_rate import CTRateDataset, CTRateFilterMode


def test_ctrate_data_fraction_applies_after_report_deduplication(monkeypatch, tmp_path):
    report_keys = [f"train_{idx}_a" for idx in range(6)]
    image_paths = [
        (tmp_path / "dataset" / "train_fixed" / key / f"{key}_{scan_idx}.nii.gz").as_posix()
        for key in report_keys
        for scan_idx in (1, 2)
    ]
    volume_names = [Path(path).name for path in image_paths]
    reports = pd.DataFrame(
        {
            "VolumeName": volume_names,
            "Findings_EN": [f"findings {name}" for name in volume_names],
            "Impressions_EN": [f"impressions {name}" for name in volume_names],
        }
    )
    organ_texts = {key: f"organ {key}" for key in report_keys}
    no_comp = {key: f"no comparison {key}" for key in report_keys}

    monkeypatch.setattr(ct_rate, "get_ctrate_image_paths", lambda _data_dir, _split: image_paths)
    monkeypatch.setattr(ct_rate.pd, "read_csv", lambda _path: reports)
    monkeypatch.setattr(ct_rate, "load_organ_texts", lambda *_args: organ_texts)
    monkeypatch.setattr(ct_rate, "load_no_comparisons_texts", lambda *_args: (no_comp, no_comp))

    full = CTRateDataset(
        data_dir=tmp_path,
        split="train",
        include_reports=True,
        filter_mode=CTRateFilterMode.FIRST_ALL,
        data_fraction=1.0,
    )
    half = CTRateDataset(
        data_dir=tmp_path,
        split="train",
        include_reports=True,
        filter_mode=CTRateFilterMode.FIRST_ALL,
        data_fraction=0.5,
    )

    assert len(full) == len(report_keys)
    assert len(half) == 3
