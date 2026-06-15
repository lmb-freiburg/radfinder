from __future__ import annotations

from types import SimpleNamespace

import torch
from radfinder.save_embeddings_lib import forward_pass
from torch import nn


class DummyImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.sliding_window_size = (4, 4, 2)
        self.patch_size = (2, 2, 1)
        self.embed_dim = 3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_windows = images.shape[0]
        n_tokens = 1 + 8
        values = torch.arange(
            batch_windows * n_tokens * self.embed_dim,
            dtype=images.dtype,
            device=images.device,
        )
        return values.view(batch_windows, n_tokens, self.embed_dim)


class DummyFeatureCombiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 4, bias=False)

    def forward(self, features: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        spatial = self.proj(features)
        cls = spatial.mean(dim=1, keepdim=True)
        return torch.cat([cls, spatial], dim=1)


class DummyTextBackbone(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        return SimpleNamespace(last_hidden_state=hidden)


def test_forward_pass_saves_backbone_feature_comb_projection_and_axis_slices(
    monkeypatch,
    tmp_path,
):
    saved = {}

    def fake_save_embeddings(embeddings, save_paths):
        assert len(save_paths) == 1
        saved[save_paths[0].name] = embeddings.detach().clone()

    monkeypatch.setattr("radfinder.save_embeddings_lib.save_embeddings", fake_save_embeddings)
    model = SimpleNamespace(
        backbone_image=DummyImageBackbone(),
        feature_comb_image=DummyFeatureCombiner(),
        projection_image=nn.Linear(8, 2, bias=False),
        backbone_text=None,
        projection_text=None,
    )
    batch = {
        "image_grid_shape": torch.tensor([[2, 1, 1]]),
        "image": torch.ones(2, 1, 4, 4, 2),
    }

    forward_pass(
        batch=batch,
        model=model,
        do_image_backbone=True,
        do_image_projection=True,
        do_text_backbone=False,
        do_text_projection=False,
        expected_files=["missing.safetensors.zst"],
        save_backbone_patches=True,
        save_sliced_backbone_patches=True,
        save_paths=[tmp_path / "scan1"],
    )

    assert saved["image_backbone_cls"].shape == (1, 2, 1, 1, 3)
    assert saved["image_backbone_patch_average"].shape == (1, 2, 1, 1, 3)
    assert saved["image_backbone_patch"].shape == (1, 2, 1, 1, 2, 2, 2, 3)
    assert saved["image_backbone_patch_axis0"].shape == (1, 2, 1, 1, 2, 3)
    assert saved["image_backbone_patch_axis1"].shape == (1, 2, 1, 1, 2, 3)
    assert saved["image_backbone_patch_axis2"].shape == (1, 2, 1, 1, 2, 3)
    assert saved["image_feature_comb_cls"].shape == (1, 4)
    assert saved["image_feature_comb_patch"].shape == (1, 2, 1, 1, 4)
    assert saved["image_projection"].shape == (1, 2)


def test_forward_pass_skips_batch_when_all_expected_files_exist(monkeypatch, tmp_path):
    called = False

    def fake_save_embeddings(embeddings, save_paths):
        nonlocal called
        called = True

    monkeypatch.setattr("radfinder.save_embeddings_lib.save_embeddings", fake_save_embeddings)
    save_dir = tmp_path / "scan1"
    save_dir.mkdir()
    (save_dir / "image_projection.safetensors.zst").write_text("exists")
    model = SimpleNamespace(
        backbone_image=DummyImageBackbone(),
        feature_comb_image=DummyFeatureCombiner(),
        projection_image=nn.Linear(8, 2, bias=False),
        backbone_text=None,
        projection_text=None,
    )

    forward_pass(
        batch={"image_grid_shape": torch.tensor([[1, 1, 1]]), "image": torch.ones(1, 1, 4, 4, 2)},
        model=model,
        do_image_backbone=True,
        do_image_projection=True,
        do_text_backbone=False,
        do_text_projection=False,
        expected_files=["image_projection.safetensors.zst"],
        save_backbone_patches=False,
        save_sliced_backbone_patches=False,
        save_paths=[save_dir],
    )

    assert not called


def test_forward_pass_saves_text_backbone_and_projection(monkeypatch, tmp_path):
    saved = {}

    def fake_save_embeddings(embeddings, save_paths):
        assert len(save_paths) == 1
        saved[save_paths[0].name] = embeddings.detach().clone()

    monkeypatch.setattr("radfinder.save_embeddings_lib.save_embeddings", fake_save_embeddings)
    model = SimpleNamespace(
        backbone_image=None,
        feature_comb_image=None,
        projection_image=None,
        backbone_text=DummyTextBackbone(),
        projection_text=nn.Linear(4, 2, bias=False),
    )
    batch = {
        "report_input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
        "report_hidden_state_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }

    forward_pass(
        batch=batch,
        model=model,
        do_image_backbone=False,
        do_image_projection=False,
        do_text_backbone=True,
        do_text_projection=True,
        expected_files=["missing.safetensors.zst"],
        save_backbone_patches=False,
        save_sliced_backbone_patches=False,
        save_paths=[tmp_path / "scan1"],
    )

    assert saved["text_backbone"].shape == (2, 4)
    torch.testing.assert_close(saved["text_backbone"][0], torch.full((4,), 2.0))
    torch.testing.assert_close(saved["text_backbone"][1], torch.full((4,), 5.0))
    assert saved["text_projection"].shape == (2, 2)
