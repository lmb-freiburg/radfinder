from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from radfinder.utils.collate import extended_collate_siglip, pad_and_stack


class DummyTokenizer:
    def __call__(
        self,
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=None,
        return_tensors=None,
    ):
        assert return_tensors == "pt"
        max_len = max(len(str(t).split()) for t in texts)
        input_ids = torch.zeros(len(texts), max_len, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, text in enumerate(texts):
            n_tokens = len(str(text).split())
            input_ids[row, :n_tokens] = torch.arange(1, n_tokens + 1)
            attention_mask[row, :n_tokens] = 1
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


def make_frozen_local_item(h: int, w: int, d: int, c: int, offset: float = 0.0) -> dict:
    values = torch.arange(h * w * d * c, dtype=torch.float32).reshape(h, w, d, c)
    return {
        "image_backbone_cls": values + offset,
        "image_backbone_patch_average": values + offset + 10_000,
        "scan_key": f"scan_{h}_{w}_{d}_{int(offset)}",
    }


def test_pad_and_stack_preserves_values_and_builds_window_mask():
    tensors = [
        torch.ones(2, 1, 1, 3),
        torch.full((1, 2, 3, 3), 2.0),
    ]

    padded, mask, grid_shape = pad_and_stack(tensors)

    assert padded.shape == (2, 2, 2, 3, 3)
    assert mask.shape == (2, 2, 2, 3)
    assert grid_shape.tolist() == [[2, 1, 1], [1, 2, 3]]
    torch.testing.assert_close(padded[0, :2, :1, :1], tensors[0])
    torch.testing.assert_close(padded[1, :1, :2, :3], tensors[1])
    assert mask[0, :2, :1, :1].all()
    assert not mask[0, :, 1:, :].any()
    assert mask[1, :1, :2, :3].all()
    assert not mask[1, 1:, :, :].any()


def test_collate_frozen_local_features_preserves_values_and_masks():
    batch = [
        make_frozen_local_item(2, 1, 1, 4, offset=0),
        make_frozen_local_item(1, 2, 3, 4, offset=100),
    ]
    originals = [
        {
            "image_backbone_cls": item["image_backbone_cls"].clone(),
            "image_backbone_patch_average": item["image_backbone_patch_average"].clone(),
        }
        for item in batch
    ]

    collated = extended_collate_siglip(batch, tokenizer=DummyTokenizer())

    assert collated["image_backbone_cls"].shape == (2, 2, 2, 3, 4)
    assert collated["image_backbone_patch_average"].shape == (2, 2, 2, 3, 4)
    assert collated["window_mask"].shape == (2, 2, 2, 3)
    assert collated["image_grid_shape"].tolist() == [[2, 1, 1], [1, 2, 3]]
    assert collated["scan_key"] == ["scan_2_1_1_0", "scan_1_2_3_100"]

    for batch_idx, original in enumerate(originals):
        h, w, d, _ = original["image_backbone_cls"].shape
        torch.testing.assert_close(
            collated["image_backbone_cls"][batch_idx, :h, :w, :d],
            original["image_backbone_cls"],
        )
        torch.testing.assert_close(
            collated["image_backbone_patch_average"][batch_idx, :h, :w, :d],
            original["image_backbone_patch_average"],
        )
        assert collated["window_mask"][batch_idx, :h, :w, :d].all()

    assert not collated["window_mask"][0, :, 1:, :].any()
    assert not collated["window_mask"][1, 1:, :, :].any()


def test_collate_axis2_features_keeps_patch_axis_and_adds_grid_mask():
    batch = [
        make_frozen_local_item(2, 1, 1, 3, offset=0),
        make_frozen_local_item(1, 2, 1, 3, offset=100),
    ]
    for item in batch:
        h, w, d, c = item["image_backbone_cls"].shape
        item["image_backbone_patch_axis2"] = torch.randn(h, w, d, 2, c)

    collated = extended_collate_siglip(batch, tokenizer=DummyTokenizer())

    assert collated["image_backbone_patch_axis2"].shape == (2, 2, 2, 1, 2, 3)
    assert collated["window_mask"].tolist() == [
        [[[True], [False]], [[True], [False]]],
        [[[True], [True]], [[False], [False]]],
    ]


def test_collate_frozen_global_features_stacks_cls_and_pads_patches():
    batch = [
        {
            "image_feature_comb_cls": torch.tensor([1.0, 2.0]),
            "image_feature_comb_patch": torch.ones(1, 2, 1, 2),
        },
        {
            "image_feature_comb_cls": torch.tensor([3.0, 4.0]),
            "image_feature_comb_patch": torch.full((2, 1, 1, 2), 5.0),
        },
    ]

    collated = extended_collate_siglip(batch, tokenizer=DummyTokenizer())

    torch.testing.assert_close(
        collated["image_feature_comb_cls"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert collated["image_feature_comb_patch"].shape == (2, 2, 2, 1, 2)
    assert collated["image_grid_shape"].tolist() == [[1, 2, 1], [2, 1, 1]]
    assert collated["window_mask"][0, :1, :2, :1].all()
    assert collated["window_mask"][1, :2, :1, :1].all()


def test_collate_tokenizes_reports():
    batch = [
        {"report": "first report"},
        {"report": "second"},
    ]

    collated = extended_collate_siglip(batch, tokenizer=DummyTokenizer())

    assert collated["report_input_ids"].shape == (2, 2)
    assert collated["report_hidden_state_mask"].tolist() == [[1, 1], [1, 0]]
    assert "sentence_input_ids" not in collated


def test_collate_fails_loudly_on_missing_keys():
    batch = [
        {"report": "present"},
        {"scan_key": "missing-report"},
    ]

    with pytest.raises(ValueError, match="Keys don't match"):
        extended_collate_siglip(batch, tokenizer=DummyTokenizer())


def test_collate_allow_none_filters_failed_samples():
    batch = [
        make_frozen_local_item(1, 1, 1, 2),
        None,
    ]

    collated = extended_collate_siglip(batch, tokenizer=DummyTokenizer(), allow_none=True)

    assert collated["image_backbone_cls"].shape == (1, 1, 1, 1, 2)
    assert collated["scan_key"] == ["scan_1_1_1_0"]


def test_collate_disallows_none_by_default():
    with pytest.raises(ValueError, match="None samples are not allowed"):
        extended_collate_siglip([None], tokenizer=DummyTokenizer())
