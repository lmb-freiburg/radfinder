import pytest
import torch
from radfinder.models.vision_language import LocalizationOutput
from radfinder.paths import get_medv_data_dir
from radfinder.tasks.localization_task import MM_PER_DEPTH_POSITION, run_localization


class DummyLocalizationModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def forward_localization(self, batch):
        return LocalizationOutput(
            scan_slice_emb=torch.tensor(
                [[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]]],
                dtype=torch.float32,
            ),
            scan_valid_depth_mask=torch.tensor([[True, True, True, False]]),
            snippet_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            slice_target_depth_mask=torch.tensor([[False, False, True, False]]),
            slice_batch_idx_valid=torch.tensor([0], dtype=torch.long),
        )


class DummyMissingLocalizationOutputModel(DummyLocalizationModel):
    def forward_localization(self, batch):
        return LocalizationOutput(
            snippet_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            slice_target_depth_mask=torch.tensor([[False, False, True, False]]),
            slice_batch_idx_valid=torch.tensor([0], dtype=torch.long),
        )


def test_run_localization_returns_details():
    batch = {
        "image_grid_shape": torch.tensor([[1, 1, 4]], dtype=torch.long),
        "valid_slices": torch.tensor([True, False]),
        "slices": [
            {
                "slice_a": {"snippet": "first snippet"},
                "slice_b": {"snippet": "ignored snippet"},
            }
        ],
        "filename": [(get_medv_data_dir() / "dummy_dataset" / "scan1" / "image.nii.gz").as_posix()],
        "scan_key": ["scan1"],
    }
    metrics, details = run_localization(
        model=DummyLocalizationModel(),
        dataloader=[batch],
        dataset=[{"scan_key": "scan1"}],
        device="cpu",
        verbose=False,
    )

    assert metrics["n_snippets"] == 1
    assert metrics["loc_acc_within_24mm"] == 1.0
    assert details["n_dataset_scans"] == 1
    assert details["n_total_slices"] == 2
    assert details["n_invalid_slices"] == 1
    assert len(details["rows"]) == 1

    row = details["rows"][0]
    assert row["scan_key"] == "scan1"
    assert row["slice_key"] == "slice_a"
    assert row["snippet_text"] == "first snippet"
    assert row["filename_rel"] == "dummy_dataset/scan1/image.nii.gz"
    assert row["pred_depth_idx"] == 0
    assert row["target_depth_idx"] == 2
    assert row["abs_error_positions"] == 2
    assert row["abs_error_mm"] == 2 * MM_PER_DEPTH_POSITION
    assert row["valid_depth_mask"] == [True, True, True, False]
    assert len(row["cosine_logits"]) == 4


def test_run_localization_fails_loudly_when_forward_omits_depth_embeddings():
    batch = {
        "image_grid_shape": torch.tensor([[1, 1, 4]], dtype=torch.long),
        "valid_slices": torch.tensor([True]),
        "slices": [{"slice_a": {"snippet": "first snippet"}}],
        "filename": [(get_medv_data_dir() / "dummy_dataset" / "scan1" / "image.nii.gz").as_posix()],
        "scan_key": ["scan1"],
    }

    with pytest.raises(RuntimeError, match="scan_slice_emb"):
        run_localization(
            model=DummyMissingLocalizationOutputModel(),
            dataloader=[batch],
            dataset=[{"scan_key": "scan1"}],
            device="cpu",
            verbose=False,
        )
