from __future__ import annotations

import pytest
import radfinder.models.load_model as load_model
import torch
from radfinder.models.load_model import FeatMode
from radfinder.models.vision_language import SigLIP, SnippetAlignmentModeC
from torch import nn


class MarkerModule(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.weight = nn.Parameter(torch.tensor(1.0))


class DummyFeatureCombiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_cls_initialized = False
        self.axis_patch_initialized = False
        self.local_cls_token = None
        self.axis_patch_proj = None
        self.stored_init_kwargs = {}

    def init_local_cls_token(self):
        self.local_cls_initialized = True
        self.local_cls_token = nn.Parameter(torch.zeros(1, 1, 5))

    def init_axis_patch_proj(self):
        self.axis_patch_initialized = True
        self.axis_patch_proj = nn.Linear(4, 5)


def test_create_spectre_component_wiring_by_feature_mode(monkeypatch):
    monkeypatch.setattr(
        load_model,
        "create_image_backbone",
        lambda model_config: MarkerModule("image_backbone"),
    )
    monkeypatch.setattr(
        load_model,
        "create_image_feature_comb",
        lambda model_config: MarkerModule("image_feature_comb"),
    )
    monkeypatch.setattr(
        load_model,
        "create_image_projection",
        lambda model_config: MarkerModule("image_projection"),
    )
    monkeypatch.setattr(
        load_model,
        "create_text_backbone",
        lambda model_config: MarkerModule("text_backbone"),
    )
    monkeypatch.setattr(
        load_model,
        "create_text_projection",
        lambda model_config: MarkerModule("text_projection"),
    )

    full = load_model.create_spectre({}, FeatMode.FULL, FeatMode.FULL)
    assert [m.name if m is not None else None for m in full] == [
        "image_backbone",
        "image_feature_comb",
        "image_projection",
        "text_backbone",
        "text_projection",
    ]

    frozen_local = load_model.create_spectre({}, FeatMode.FROZEN_LOCAL, FeatMode.FULL)
    assert [m.name if m is not None else None for m in frozen_local] == [
        None,
        "image_feature_comb",
        "image_projection",
        "text_backbone",
        "text_projection",
    ]

    frozen_global_no_text_projection = load_model.create_spectre(
        {},
        FeatMode.FROZEN_GLOBAL,
        FeatMode.NONE,
    )
    assert [m.name if m is not None else None for m in frozen_global_no_text_projection] == [
        None,
        "image_feature_comb",
        "image_projection",
        "text_backbone",
        None,
    ]


def test_create_siglip_axis_localization_initializes_required_components(monkeypatch):
    feature_combiner = DummyFeatureCombiner()
    monkeypatch.setattr(
        load_model,
        "create_spectre",
        lambda *args, **kwargs: (
            None,
            feature_combiner,
            nn.Linear(10, 3),
            MarkerModule("text_backbone"),
            nn.Linear(4, 3),
        ),
    )
    model_config = {
        "text_hidden_size": 4,
        "feature_comb_embed_dim": 5,
    }
    train_config = {
        "model": {
            "learnable_t": False,
            "learnable_b": False,
            "normalize": False,
            "init_t": 0.0,
            "init_b": 0.0,
        },
        "train": {
            "do_snippet_alignment": {
                "enabled": True,
                "snippet_mode": SnippetAlignmentModeC.AXIS_LOCALIZATION,
                "dual_cls_token": False,
                "axis2_use_cls_input": False,
                "localization_sigma": 2.0,
                "localization_tau": 0.1,
            },
            "model_settings": {},
        },
    }

    model = load_model.create_siglip(
        model_config,
        image_feat_mode=FeatMode.FROZEN_LOCAL,
        text_feat_mode=FeatMode.FULL,
        train_config=train_config,
    )

    assert isinstance(model, SigLIP)
    assert model.loc_criterion is not None
    assert not hasattr(model, "ot_criterion")
    assert feature_combiner.axis_patch_initialized


def test_create_siglip_axis_localization_can_reuse_cls_input_without_axis_patch_proj(monkeypatch):
    feature_combiner = DummyFeatureCombiner()
    monkeypatch.setattr(
        load_model,
        "create_spectre",
        lambda *args, **kwargs: (
            None,
            feature_combiner,
            nn.Linear(10, 3),
            MarkerModule("text_backbone"),
            nn.Linear(4, 3),
        ),
    )
    model_config = {
        "text_hidden_size": 4,
        "feature_comb_embed_dim": 5,
    }
    train_config = {
        "model": {
            "learnable_t": False,
            "learnable_b": False,
            "normalize": False,
            "init_t": 0.0,
            "init_b": 0.0,
        },
        "train": {
            "do_snippet_alignment": {
                "enabled": True,
                "snippet_mode": SnippetAlignmentModeC.AXIS_LOCALIZATION,
                "axis2_use_cls_input": True,
                "localization_sigma": 2.0,
                "localization_tau": 0.1,
            },
            "model_settings": {},
        },
    }

    model = load_model.create_siglip(
        model_config,
        image_feat_mode=FeatMode.FROZEN_LOCAL,
        text_feat_mode=FeatMode.FULL,
        train_config=train_config,
    )

    assert isinstance(model, SigLIP)
    assert model.loc_criterion is not None
    assert not feature_combiner.axis_patch_initialized
