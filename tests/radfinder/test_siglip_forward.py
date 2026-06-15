from __future__ import annotations

from types import SimpleNamespace

import torch
from radfinder.models.modeling import last_token_pool
from radfinder.models.vision_language import GlobalContrastiveOutput, GlobalResC, MaskUsageC, SigLIP
from torch import nn


class TinyImageBackbone(nn.Module):
    def __init__(self, embed_dim: int = 4, n_patches: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_patches = n_patches
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.sliding_window_size = (4, 4, 2)
        self.patch_size = (2, 2, 1)

    def forward(
        self, image: torch.Tensor, attn_mask_3d: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch = image.shape[0]
        base = image.flatten(1).mean(dim=1).view(batch, 1, 1)
        template = torch.arange(
            (self.n_patches + 1) * self.embed_dim,
            dtype=image.dtype,
            device=image.device,
        ).view(1, self.n_patches + 1, self.embed_dim)
        return template + self.scale * base


class TinyTextBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(64, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyFeatureCombiner(nn.Module):
    def __init__(self, input_dim: int = 8, axis_dim: int = 4, embed_dim: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.axis_dim = axis_dim
        self.patch_proj = nn.Linear(input_dim, embed_dim, bias=False)
        self.axis_patch_proj = None
        self.local_cls_token = None
        self.stored_init_kwargs = {
            "input_dim": input_dim,
            "axis_dim": axis_dim,
            "embed_dim": embed_dim,
        }

    def init_axis_patch_proj(self):
        self.axis_patch_proj = nn.Linear(self.axis_dim, self.embed_dim, bias=False)

    def init_local_cls_token(self):
        self.local_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

    def forward(
        self,
        x: torch.Tensor,
        grid_size: tuple[int, int, int] | torch.Tensor | None = None,
        attn_mask_3d: torch.Tensor | None = None,
        coord_divisor=None,
    ) -> torch.Tensor:
        return self.forward_features(x, grid_size, attn_mask_3d, coord_divisor)

    def forward_features(
        self,
        x: torch.Tensor,
        grid_size: tuple[int, int, int] | torch.Tensor | None = None,
        attn_mask_3d: torch.Tensor | None = None,
        coord_divisor=None,
    ) -> torch.Tensor:
        return self.forward_features_projected(
            self.patch_proj(x),
            grid_size=grid_size,
            attn_mask_3d=attn_mask_3d,
            coord_divisor=coord_divisor,
        )

    def forward_features_projected(
        self,
        x: torch.Tensor,
        grid_size: tuple[int, int, int] | torch.Tensor | None = None,
        attn_mask_3d: torch.Tensor | None = None,
        coord_divisor=None,
    ) -> torch.Tensor:
        if attn_mask_3d is None:
            pooled = x.mean(dim=1)
        else:
            mask = attn_mask_3d.reshape(x.shape[0], -1).to(x.device)
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(
                min=1
            )
        prefixes = [pooled.unsqueeze(1)]
        if self.local_cls_token is not None:
            prefixes.append((pooled + 1.0).unsqueeze(1))
        return torch.cat(prefixes + [x], dim=1)


def make_siglip(
    *,
    image_backbone: nn.Module | None = None,
    model_settings: dict | None = None,
    mask_usage_eval: str = MaskUsageC.TRUE,
) -> SigLIP:
    torch.manual_seed(0)
    feature_combiner = TinyFeatureCombiner()
    image_projection = nn.Linear(10, 3, bias=False)
    text_projection = nn.Linear(4, 3, bias=False)
    model = SigLIP(
        image_backbone=image_backbone,
        text_backbone=TinyTextBackbone(),
        image_feature_comb=feature_combiner,
        image_projection=image_projection,
        text_projection=text_projection,
        mask_usage_train=mask_usage_eval,
        mask_usage_eval=mask_usage_eval,
        model_settings=model_settings,
        do_snippet_alignment={"enabled": False},
    )
    model.eval()
    return model


def make_frozen_local_inputs():
    image_backbone_cls = torch.arange(2 * 2 * 1 * 2 * 4, dtype=torch.float32).view(2, 2, 1, 2, 4)
    image_backbone_patch_average = image_backbone_cls + 100.0
    image_grid_shape = torch.tensor([[2, 1, 1], [1, 1, 2]])
    window_mask = torch.zeros(2, 2, 1, 2, dtype=torch.bool)
    window_mask[0, :2, :1, :1] = True
    window_mask[1, :1, :1, :2] = True
    report_input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    report_hidden_state_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    return {
        "image_backbone_cls": image_backbone_cls,
        "image_backbone_patch_average": image_backbone_patch_average,
        "image_grid_shape": image_grid_shape,
        "window_mask": window_mask,
        "report_input_ids": report_input_ids,
        "report_hidden_state_mask": report_hidden_state_mask,
    }


def clone_inputs(inputs: dict) -> dict:
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def test_siglip_forward_global_contrastive_returns_image_and_text_outputs():
    model = make_siglip()
    batch = make_frozen_local_inputs()

    output = model.forward_global_contrastive(batch)

    # Localization fields are intentionally not part of GlobalContrastiveOutput;
    # the typed return enforces that contract at the schema level.
    assert isinstance(output, GlobalContrastiveOutput)
    assert output.image_embeddings.shape == (2, 3)
    assert output.image_embeddings_secondary.shape == (2, 3)
    assert output.text_embeddings.shape == (2, 3)


def test_siglip_forward_default_dispatches_to_global_contrastive():
    """`model(batch)` is kept as a thin alias for forward_global_contrastive."""
    model = make_siglip()
    batch = make_frozen_local_inputs()

    direct = model.forward_global_contrastive(clone_inputs(batch))
    via_call = model(clone_inputs(batch))

    torch.testing.assert_close(direct.image_embeddings, via_call.image_embeddings)
    torch.testing.assert_close(direct.text_embeddings, via_call.text_embeddings)


def test_siglip_forward_image_only_omits_text_embeddings():
    model = make_siglip()
    batch = make_frozen_local_inputs()
    batch.pop("report_input_ids")
    batch.pop("report_hidden_state_mask")

    output = model.forward_image_only(batch)

    assert output.image_embeddings.shape == (2, 3)
    assert output.image_embeddings_secondary.shape == (2, 3)
    assert output.text_embeddings is None


def test_siglip_forward_image_only_matches_global_contrastive_image_side():
    model = make_siglip()
    batch = make_frozen_local_inputs()

    via_global = model.forward_global_contrastive(clone_inputs(batch))
    via_image_only = model.forward_image_only(clone_inputs(batch))

    torch.testing.assert_close(via_image_only.image_embeddings, via_global.image_embeddings)
    torch.testing.assert_close(
        via_image_only.image_embeddings_secondary, via_global.image_embeddings_secondary
    )
    assert via_image_only.text_embeddings is None


def test_siglip_forward_image_only_tolerates_extra_batch_keys():
    """Extra dict keys (text, slices) are simply not read by forward_image_only."""
    model = make_siglip()
    batch = clone_inputs(make_frozen_local_inputs())
    batch["snippet_input_ids"] = torch.tensor([[1, 2]])
    batch["snippet_attention_mask"] = torch.tensor([[1, 1]])
    batch["slices"] = [{"a": 1}]
    batch["report_hidden_state"] = None
    batch["report_pooled"] = None

    output = model.forward_image_only(batch)
    assert output.image_embeddings.shape == (2, 3)
    assert output.text_embeddings is None


def test_siglip_encode_text_report_matches_direct_component_path():
    model = make_siglip()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    hidden_state = model.backbone_text(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    pooled = last_token_pool(hidden_state, attention_mask)
    expected = model.projection_text(pooled)

    encoded = model.encode_text_report(input_ids, attention_mask)
    encoded_from_hidden = model.encode_text_report(
        None,
        attention_mask,
        report_hidden_state=hidden_state,
    )
    encoded_from_pool = model.encode_text_report(None, None, report_pooled=pooled)

    torch.testing.assert_close(encoded, expected)
    torch.testing.assert_close(encoded_from_hidden, expected)
    torch.testing.assert_close(encoded_from_pool, expected)


def test_siglip_encode_prompt_texts_matches_direct_component_path():
    model = make_siglip()
    input_ids = torch.tensor([[1, 2, 0], [6, 7, 8]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    hidden_state = model.backbone_text(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    expected = model.projection_text(last_token_pool(hidden_state, attention_mask))

    encoded = model.encode_prompt_texts(input_ids, attention_mask)

    torch.testing.assert_close(encoded, expected)


def test_siglip_encode_snippets_matches_direct_component_path():
    model = make_siglip()
    input_ids = torch.tensor([[4, 5, 0], [9, 10, 11]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    snippet_hidden = model.backbone_text(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    snippet_pooled = last_token_pool(snippet_hidden, attention_mask)
    expected = model.project_slice_text_features(snippet_pooled)

    encoded = model.encode_snippets(input_ids, attention_mask)

    torch.testing.assert_close(encoded, expected)


def test_siglip_forward_full_image_backbone_fixed_grid_without_window_mask():
    model = make_siglip(image_backbone=TinyImageBackbone(), mask_usage_eval=MaskUsageC.FALSE)
    image = torch.arange(2 * 1 * 4 * 4 * 2, dtype=torch.float32).view(2, 1, 4, 4, 2)

    output = model(
        {
            "image": image,
            "image_grid_shape": torch.tensor([[2, 1, 1]]),
            "window_mask": None,
            "report_input_ids": torch.tensor([[1, 2, 3]]),
            "report_hidden_state_mask": torch.tensor([[1, 1, 1]]),
        }
    )

    assert output.image_embeddings.shape == (1, 3)
    assert output.image_embeddings_secondary.shape == (1, 3)
    assert output.text_embeddings.shape == (1, 3)


def test_siglip_forward_axis2_global_produces_image_embeddings():
    model = make_siglip(model_settings={"global_res": GlobalResC.AXIS2})
    model.feature_comb_image.init_axis_patch_proj()
    image_grid_shape = torch.tensor([[1, 1, 2]])
    window_mask = torch.tensor([[[[True, True]]]])
    image_backbone_cls = torch.randn(1, 1, 1, 2, 4)
    image_backbone_patch_average = torch.randn(1, 1, 1, 2, 4)
    image_backbone_patch_axis2 = torch.randn(1, 1, 1, 2, 2, 4)

    output = model(
        {
            "image_grid_shape": image_grid_shape,
            "window_mask": window_mask,
            "image_backbone_cls": image_backbone_cls,
            "image_backbone_patch_average": image_backbone_patch_average,
            "image_backbone_patch_axis2": image_backbone_patch_axis2,
            "report_input_ids": torch.tensor([[1, 2, 3]]),
            "report_hidden_state_mask": torch.tensor([[1, 1, 1]]),
        }
    )

    # scan_slice_emb / scan_valid_depth_mask belong to forward_localization,
    # not to the global contrastive path — the typed return enforces this.
    assert isinstance(output, GlobalContrastiveOutput)
    assert output.image_embeddings.shape == (1, 3)
    assert output.image_embeddings_secondary.shape == (1, 3)


def test_siglip_forward_localization_skips_batches_without_slices():
    model = make_siglip()
    output = model.forward_localization(
        {
            "image_grid_shape": torch.tensor([[1, 1, 2]]),
            "window_mask": torch.tensor([[[[True, True]]]]),
            "image_backbone_cls": torch.randn(1, 1, 1, 2, 4),
            "image_backbone_patch_axis2": torch.randn(1, 1, 1, 2, 2, 4),
            "slices": None,
        }
    )
    assert output.scan_slice_emb is None
    assert output.snippet_emb is None


def test_siglip_encode_axis2_depth_uses_cls_input_for_zero_shot_config():
    model = make_siglip()
    assert model.feature_comb_image.axis_patch_proj is None

    image_grid_shape = torch.tensor([[1, 1, 2]])
    window_mask = torch.tensor([[[[True, True]]]])
    image_backbone_cls = torch.randn(1, 1, 1, 2, 4)
    image_backbone_patch_axis2 = torch.randn(1, 1, 1, 2, 2, 4)

    scan_slice_emb, scan_valid_depth_mask = model.encode_axis2_depth(
        image_grid_shape=image_grid_shape,
        window_mask=window_mask,
        image_backbone_cls=image_backbone_cls,
        image_backbone_patch_axis2=image_backbone_patch_axis2,
    )

    expected_emb, expected_mask, *_ = model._forward_axis2_combiner(
        feature_comb_image=model.feature_comb_image,
        image_grid_shape=image_grid_shape,
        window_mask=window_mask,
        image_backbone_patch_axis2=image_backbone_patch_axis2,
        image_backbone_cls=image_backbone_cls,
        axis2_use_cls_input=True,
    )
    torch.testing.assert_close(scan_slice_emb, expected_emb)
    torch.testing.assert_close(scan_valid_depth_mask, expected_mask)


def test_siglip_forward_localization_uses_axis2_features_for_zero_shot_config():
    model = make_siglip()
    assert model.feature_comb_image.axis_patch_proj is None

    image_grid_shape = torch.tensor([[1, 1, 2]])
    window_mask = torch.tensor([[[[True, True]]]])
    image_backbone_cls = torch.randn(1, 1, 1, 2, 4)
    image_backbone_patch_axis2 = torch.randn(1, 1, 1, 2, 2, 4)

    snippet_input_ids = torch.tensor([[4, 5, 0]])
    snippet_attention_mask = torch.tensor([[1, 1, 0]])

    output = model.forward_localization(
        {
            "image_grid_shape": image_grid_shape,
            "window_mask": window_mask,
            "image_backbone_cls": image_backbone_cls,
            "image_backbone_patch_axis2": image_backbone_patch_axis2,
            "slices": [{"slice_a": {"snippet": "first snippet"}}],
            "slice_batch_idx": torch.tensor([0]),
            "slice_target_depth_mask": torch.tensor([[False, False, True, False]]),
            "valid_slices": torch.tensor([True]),
            "snippet_input_ids": snippet_input_ids,
            "snippet_attention_mask": snippet_attention_mask,
        }
    )

    assert output.scan_slice_emb.shape == (1, 4, 3)
    assert output.scan_valid_depth_mask.tolist() == [[True, True, True, True]]
    assert output.snippet_emb.shape == (1, 3)
    assert output.slice_batch_idx_valid.tolist() == [0]

    expected_scan_slice_emb, expected_valid_mask, *_ = model._forward_axis2_combiner(
        feature_comb_image=model.feature_comb_image,
        image_grid_shape=image_grid_shape,
        window_mask=window_mask,
        image_backbone_patch_axis2=image_backbone_patch_axis2,
        image_backbone_cls=image_backbone_cls,
        axis2_use_cls_input=True,
    )
    torch.testing.assert_close(output.scan_slice_emb, expected_scan_slice_emb)
    torch.testing.assert_close(output.scan_valid_depth_mask, expected_valid_mask)

    snippet_hidden = model.backbone_text(
        input_ids=snippet_input_ids,
        attention_mask=snippet_attention_mask,
    ).last_hidden_state
    snippet_pooled = last_token_pool(snippet_hidden, snippet_attention_mask)
    expected_snippet_emb = model.project_slice_text_features(snippet_pooled)
    torch.testing.assert_close(output.snippet_emb, expected_snippet_emb)
