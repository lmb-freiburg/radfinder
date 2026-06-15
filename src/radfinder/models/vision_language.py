"""
Implementation of the CLIP framework for text-image feature alignment.

This module provides the necessary components to train the CLIP framework an is based on the
original paper: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021),
https://arxiv.org/abs/2103.00020

Addional resources:
Hamamci et al., "Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography" (2024),
https://arxiv.org/abs/2403.17834
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from radfinder.losses.localization_loss import GaussianLocalizationLoss
from radfinder.losses.siglip_loss import SigLIPLoss
from radfinder.models.modeling import last_token_pool, masked_average
from radfinder.models.siglip_head import SigLIPProjectionHead
from radfinder.models.vision_transformer import get_default_snippet_alignment
from radfinder.models.vision_transformer_features import FeatureVisionTransformer
from radfinder.paths import get_medv_output_dir
from radfinder.utils.collate import eq, pad_and_stack
from radfinder.utils.logging_utils import log_error, log_info, log_warning
from radfinder.utils.misc import load_safetensors_state
from transformers.utils import ModelOutput

from packg.constclass import Const
from packg.iotools.pathspec_matcher import make_git_pathspec
from packg.strings.formatters import dict_to_str_comma_equals


class MaskUsageC(Const):
    """torch graph might be faster if we fix the if path using this setting, instead of dynamic."""

    DYNAMIC = "dynamic"
    TRUE = "true"
    FALSE = "false"


class SnippetAlignmentModeC(Const):
    AXIS_LOCALIZATION = "axis_localization"
    NONE = "none"


class GlobalResC(Const):
    LOW_RES = "low_res"
    AXIS2 = "axis2"


@dataclass
class GlobalContrastiveOutput(ModelOutput):
    """Image-text contrastive output: image + report embeddings in the shared space."""

    image_embeddings: torch.Tensor | None = None
    image_embeddings_secondary: torch.Tensor | None = None
    text_embeddings: torch.Tensor | None = None


@dataclass
class LocalizationOutput(ModelOutput):
    """
    Localization output: per-depth scan embeddings + snippet embeddings.

    All fields are `None` on the graceful-skip path (no slices in the batch or
    no axis2 features).
    """

    scan_slice_emb: torch.Tensor | None = None
    scan_valid_depth_mask: torch.Tensor | None = None
    snippet_emb: torch.Tensor | None = None
    slice_target_depth_mask: torch.Tensor | None = None
    slice_batch_idx_valid: torch.Tensor | None = None


class SigLIP(nn.Module):
    def __init__(
        self,
        image_backbone: nn.Module | None = None,
        text_backbone: nn.Module | None = None,
        image_feature_comb: FeatureVisionTransformer = None,
        image_projection: nn.Module | None = None,
        text_projection: nn.Module | None = None,
        criterion: SigLIPLoss | None = None,
        loc_criterion: GaussianLocalizationLoss | None = None,
        mask_usage_train: MaskUsageC = MaskUsageC.FALSE,  # default: force fixed images during train
        mask_usage_eval: MaskUsageC = MaskUsageC.TRUE,  # default: force variable images during eval
        model_settings: dict | None = None,
        do_snippet_alignment: dict = get_default_snippet_alignment(),
    ):
        super().__init__()
        self.backbone_image = image_backbone
        self.feature_comb_image = image_feature_comb
        self.projection_image = image_projection
        self.backbone_text = text_backbone
        self.projection_text = text_projection
        self.criterion = criterion
        self.loc_criterion = loc_criterion
        self.mask_usage_train = mask_usage_train
        self.mask_usage_eval = mask_usage_eval
        self.mask_usage = self.mask_usage_train
        self.do_snippet_alignment = do_snippet_alignment
        self.model_settings = model_settings or {}

        projector_mode_image = self.do_snippet_alignment.get("projector_mode_image", "same")
        projector_mode_text = self.do_snippet_alignment.get("projector_mode_text", "same")
        self.projection_image_copy = None
        if projector_mode_image == "copy" and self.projection_image is not None:
            self.projection_image_copy = SigLIPProjectionHead(
                **self._get_projection_values(self.projection_image)
            )
            self._update_projection_image_copy_weights()

        self.projection_text_copy = None
        if projector_mode_text == "copy" and self.projection_text is not None:
            self.projection_text_copy = SigLIPProjectionHead(
                **self._get_projection_values(self.projection_text)
            )
            self._update_projection_text_copy_weights()

    def _update_projection_image_copy_weights(self):
        self.projection_image_copy.load_state_dict(deepcopy(self.projection_image.state_dict()))

    def _update_projection_text_copy_weights(self):
        self.projection_text_copy.load_state_dict(deepcopy(self.projection_text.state_dict()))

    def _get_projection_values(self, projection_module: SigLIPProjectionHead):
        copy_keys = [
            "input_dim",
            "output_dim",
            "hidden_dim",
            "bottleneck_dim",
            "layer_norm",
            "freeze_last_layer",
            "norm_last_layer",
        ]
        copy_values = [getattr(projection_module, key) for key in copy_keys]
        copy_dict = dict(zip(copy_keys, copy_values))
        return copy_dict

    # ---------- forward pass related methods ----------

    def _encode_lowres_image(
        self,
        image_backbone_cls: torch.Tensor | None,
        image_backbone_patch_average: torch.Tensor | None,
        image_feature_comb_cls: torch.Tensor | None,
        image_feature_comb_patch: torch.Tensor | None,
        image_grid_shape: torch.Tensor,
        window_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        image_feature_comb_input = None
        image_feature_comb_cls_secondary = None

        if self.feature_comb_image is not None and image_backbone_cls is not None:
            assert image_backbone_patch_average is not None
            (
                image_feature_comb_input,
                image_feature_comb_cls,
                image_feature_comb_cls_secondary,
                image_feature_comb_patch,
            ) = _run_lowres_feature_combiner(
                self.feature_comb_image,
                image_backbone_cls,
                image_backbone_patch_average,
                image_grid_shape,
                window_mask,
            )
        else:
            # backward comp: old experiments with frozen_global setting that don't have a local cls
            image_feature_comb_cls_secondary = image_feature_comb_cls

        image_embeddings = None
        image_embeddings_secondary = None
        if self.projection_image is not None:
            assert image_feature_comb_cls is not None and image_feature_comb_patch is not None
            assert image_feature_comb_cls_secondary is not None
            avg = _pool_feature_combiner_patch(image_feature_comb_patch, window_mask)
            # Combine cls and mean of patches for projection
            image_embeddings = torch.cat([image_feature_comb_cls, avg], dim=1)
            image_embeddings = self.projection_image(image_embeddings)

            # create local image embeddings for prompt rate training
            # IFF dual cls is False, and copy projector is False, this is same as image_embeddings
            image_embeddings_secondary = torch.cat([image_feature_comb_cls_secondary, avg], dim=1)
            image_embeddings_secondary = self.project_slice_vision_features(
                image_embeddings_secondary
            )

        return (
            image_embeddings,
            image_embeddings_secondary,
            image_feature_comb_input,
            image_feature_comb_cls,
            image_feature_comb_cls_secondary,
            image_feature_comb_patch,
        )

    def _project_axis2_global(
        self,
        axis2_cls_token: torch.Tensor,
        axis2_local_cls_token: torch.Tensor,
        axis2_perslice: torch.Tensor,
        scan_valid_depth_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pool: [CLS; masked_avg(perslice)] → projection_image → (B, 512)
        mask = scan_valid_depth_mask.unsqueeze(-1)  # (B, D2, 1)
        avg_perslice = (axis2_perslice * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B, E_comb)
        axis2_proj_input = torch.cat([axis2_cls_token, avg_perslice], dim=1)  # (B, 2*E_comb)
        image_embeddings = self.projection_image(axis2_proj_input)  # (B, 512)
        # create local image embeddings for prompt rate training
        # IFF dual cls is False, and copy projector is False, this is same as image_embeddings
        local_proj_input = torch.cat([axis2_local_cls_token, avg_perslice], dim=1)
        image_embeddings_secondary = self.project_slice_vision_features(local_proj_input)
        return image_embeddings, image_embeddings_secondary

    def encode_text_report(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        report_hidden_state: torch.Tensor | None = None,
        report_pooled: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if report_hidden_state is None and report_pooled is None:
            text_backbone_output = self.backbone_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            report_hidden_state = text_backbone_output.last_hidden_state
            del text_backbone_output

        if self.projection_text is None:
            return None

        # run projection on last token last hidden state (B, embed_dim)
        if report_pooled is None:
            report_pooled = last_token_pool(report_hidden_state, attention_mask)
            del report_hidden_state, attention_mask
        text_embeddings = self.projection_text(report_pooled)
        del report_pooled
        return text_embeddings

    def encode_prompt_texts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_backbone_output = self.backbone_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = last_token_pool(text_backbone_output.last_hidden_state, attention_mask)
        del text_backbone_output
        return self.projection_text(pooled)

    def encode_snippets(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_backbone_output = self.backbone_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        snippet_hidden_state = text_backbone_output.last_hidden_state
        snippet_pooled = last_token_pool(snippet_hidden_state, attention_mask)
        snippet_emb = self.project_slice_text_features(snippet_pooled)  # (S, 512)
        return snippet_emb

    def _encode_image_no_slices(
        self,
        image: torch.Tensor | None,
        image_grid_shape: torch.Tensor,
        window_mask: torch.Tensor | None,
        image_patches_mask: torch.Tensor | None,
        image_backbone_cls: torch.Tensor | None,
        image_backbone_patch_average: torch.Tensor | None,
        image_feature_comb_cls: torch.Tensor | None,
        image_feature_comb_patch: torch.Tensor | None,
        image_backbone_patch_axis2: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Shared image-encoding path for the no-slice forward methods.

        Handles low-res (and AXIS2_LOC outside localization) and AXIS2 global.
        Returns: (image_embeddings, image_embeddings_secondary, image_feature_comb_cls,
        image_feature_comb_cls_secondary, image_feature_comb_patch, resolved_window_mask).
        """
        global_res = self.model_settings.get("global_res", GlobalResC.LOW_RES)
        if global_res == GlobalResC.AXIS2 and image_backbone_patch_axis2 is None:
            raise ValueError(
                f"global_res={global_res} but image_backbone_patch_axis2 not provided in the batch"
            )
        window_mask = _resolve_window_mask(self.mask_usage, image_grid_shape, window_mask)

        if self.backbone_image is not None:
            image_backbone_cls, image_backbone_patch_average = _run_image_backbone(
                self.backbone_image,
                image,
                image_patches_mask,
                image_grid_shape,
                window_mask,
            )

        is_low_res_forward = global_res == GlobalResC.LOW_RES
        image_embeddings = None
        image_embeddings_secondary = None
        image_feature_comb_cls_secondary = image_feature_comb_cls
        if is_low_res_forward:
            (
                image_embeddings,
                image_embeddings_secondary,
                _image_feature_comb_input,
                image_feature_comb_cls,
                image_feature_comb_cls_secondary,
                image_feature_comb_patch,
            ) = self._encode_lowres_image(
                image_backbone_cls=image_backbone_cls,
                image_backbone_patch_average=image_backbone_patch_average,
                image_feature_comb_cls=image_feature_comb_cls,
                image_feature_comb_patch=image_feature_comb_patch,
                image_grid_shape=image_grid_shape,
                window_mask=window_mask,
            )

        if global_res == GlobalResC.AXIS2:
            assert image_backbone_patch_axis2 is not None
            axis2_use_cls_input = self.do_snippet_alignment.get("axis2_use_cls_input", False)
            (
                _scan_slice_emb,
                scan_valid_depth_mask,
                axis2_cls_token,
                axis2_local_cls_token,
                axis2_perslice,
                _axis2_spatial_3d,
            ) = self._forward_axis2_combiner(
                feature_comb_image=self.feature_comb_image,
                image_grid_shape=image_grid_shape,
                window_mask=window_mask,
                image_backbone_patch_axis2=image_backbone_patch_axis2,
                image_backbone_cls=image_backbone_cls,
                axis2_use_cls_input=axis2_use_cls_input,
            )
            image_embeddings, image_embeddings_secondary = self._project_axis2_global(
                axis2_cls_token=axis2_cls_token,
                axis2_local_cls_token=axis2_local_cls_token,
                axis2_perslice=axis2_perslice,
                scan_valid_depth_mask=scan_valid_depth_mask,
            )

        return (
            image_embeddings,
            image_embeddings_secondary,
            image_feature_comb_cls,
            image_feature_comb_cls_secondary,
            image_feature_comb_patch,
            window_mask,
        )

    def forward_image_only(self, batch: dict) -> "GlobalContrastiveOutput":
        """Image to global embedding. No text, no slices."""
        image_grid_shape = batch["image_grid_shape"]
        window_mask = batch["window_mask"]
        # The image-feature keys below are mode-dependent: each `image_feat_mode`
        # (`full`, `frozen_local`, `frozen_global`) populates a different subset,
        # and `global_res=axis2` additionally requires `image_backbone_patch_axis2`.
        # `_encode_image_no_slices` validates that the right ones are present for
        # the active config and raises otherwise.
        image = batch.get("image")
        image_patches_mask = batch.get("image_patches_mask")
        image_backbone_cls = batch.get("image_backbone_cls")
        image_backbone_patch_average = batch.get("image_backbone_patch_average")
        image_feature_comb_cls = batch.get("image_feature_comb_cls")
        image_feature_comb_patch = batch.get("image_feature_comb_patch")
        image_backbone_patch_axis2 = batch.get("image_backbone_patch_axis2")

        image_embeddings, image_embeddings_secondary, *_ = self._encode_image_no_slices(
            image=image,
            image_grid_shape=image_grid_shape,
            window_mask=window_mask,
            image_patches_mask=image_patches_mask,
            image_backbone_cls=image_backbone_cls,
            image_backbone_patch_average=image_backbone_patch_average,
            image_feature_comb_cls=image_feature_comb_cls,
            image_feature_comb_patch=image_feature_comb_patch,
            image_backbone_patch_axis2=image_backbone_patch_axis2,
        )
        return GlobalContrastiveOutput(
            image_embeddings=image_embeddings,
            image_embeddings_secondary=image_embeddings_secondary,
        )

    def forward_global_contrastive(self, batch: dict) -> "GlobalContrastiveOutput":
        """
        Global image-report contrastive path.

        Returns image, local-image, and text embeddings in the shared SigLIP space.
        Does NOT produce localization fields — use `forward_localization` for
        `scan_slice_emb` / `scan_valid_depth_mask`.
        """
        image_grid_shape = batch["image_grid_shape"]
        window_mask = batch["window_mask"]
        # Image-feature keys are mode-dependent (image_feat_mode determines which
        # subset is in the batch); downstream helpers validate the active mode.
        image = batch.get("image")
        image_patches_mask = batch.get("image_patches_mask")
        image_backbone_cls = batch.get("image_backbone_cls")
        image_backbone_patch_average = batch.get("image_backbone_patch_average")
        image_feature_comb_cls = batch.get("image_feature_comb_cls")
        image_feature_comb_patch = batch.get("image_feature_comb_patch")
        image_backbone_patch_axis2 = batch.get("image_backbone_patch_axis2")
        # Text input: either raw token ids (text backbone runs) or pre-computed
        # hidden state / pooled (frozen text mode). `encode_text_report` picks
        # which path to run and raises if neither set is present.
        report_input_ids = batch.get("report_input_ids")
        report_hidden_state_mask = batch.get("report_hidden_state_mask")
        report_hidden_state = batch.get("report_hidden_state")
        report_pooled = batch.get("report_pooled")

        global_res = self.model_settings.get("global_res", GlobalResC.LOW_RES)
        if global_res == GlobalResC.AXIS2 and image_backbone_patch_axis2 is None:
            raise ValueError(
                f"global_res={global_res} but image_backbone_patch_axis2 not provided in the batch"
            )
        window_mask = _resolve_window_mask(self.mask_usage, image_grid_shape, window_mask)

        if self.backbone_image is not None:
            image_backbone_cls, image_backbone_patch_average = _run_image_backbone(
                self.backbone_image,
                image,
                image_patches_mask,
                image_grid_shape,
                window_mask,
            )

        is_low_res_forward = global_res == GlobalResC.LOW_RES
        image_embeddings = None
        image_embeddings_secondary = None
        if is_low_res_forward:
            (
                image_embeddings,
                image_embeddings_secondary,
                _image_feature_comb_input,
                _image_feature_comb_cls,
                _image_feature_comb_cls_secondary,
                _image_feature_comb_patch,
            ) = self._encode_lowres_image(
                image_backbone_cls=image_backbone_cls,
                image_backbone_patch_average=image_backbone_patch_average,
                image_feature_comb_cls=image_feature_comb_cls,
                image_feature_comb_patch=image_feature_comb_patch,
                image_grid_shape=image_grid_shape,
                window_mask=window_mask,
            )

        text_embeddings = self.encode_text_report(
            report_input_ids,
            report_hidden_state_mask,
            report_hidden_state=report_hidden_state,
            report_pooled=report_pooled,
        )

        # AXIS2 global override: compute global image embedding from axis2 features.
        if global_res == GlobalResC.AXIS2:
            assert image_backbone_patch_axis2 is not None
            (
                _scan_slice_emb,
                scan_valid_depth_mask,
                axis2_cls_token,
                axis2_local_cls_token,
                axis2_perslice,
                _axis2_spatial_3d,
            ) = self._forward_axis2_combiner(
                feature_comb_image=self.feature_comb_image,
                image_grid_shape=image_grid_shape,
                window_mask=window_mask,
                image_backbone_patch_axis2=image_backbone_patch_axis2,
                image_backbone_cls=image_backbone_cls,
                axis2_use_cls_input=self.do_snippet_alignment.get("axis2_use_cls_input", False),
            )
            image_embeddings, image_embeddings_secondary = self._project_axis2_global(
                axis2_cls_token=axis2_cls_token,
                axis2_local_cls_token=axis2_local_cls_token,
                axis2_perslice=axis2_perslice,
                scan_valid_depth_mask=scan_valid_depth_mask,
            )

        return GlobalContrastiveOutput(
            image_embeddings=image_embeddings,
            image_embeddings_secondary=image_embeddings_secondary,
            text_embeddings=text_embeddings,
        )

    def encode_axis2_depth(
        self,
        image_grid_shape: torch.Tensor,
        window_mask: torch.Tensor,
        image_backbone_cls: torch.Tensor | None,
        image_backbone_patch_axis2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Axis2 features → per-depth projected image embeddings + valid-depth mask.

        Chooses the axis2 combiner input format from the active config:
        - `snippet_mode=AXIS_LOCALIZATION`: uses the configured `axis2_use_cls_input`
          flag (default False → axis_patch_proj path used during training).
        - Otherwise: forces the `[CLS, patch_avg]` input format, which matches the
          pretrained combiner. This is the path used for zero-shot localization eval
          where snippet alignment isn't configured.

        Returns `(scan_slice_emb, scan_valid_depth_mask)`.
        """
        snippet_mode = self.do_snippet_alignment.get("snippet_mode", SnippetAlignmentModeC.NONE)
        if snippet_mode == SnippetAlignmentModeC.AXIS_LOCALIZATION:
            axis2_use_cls_input = self.do_snippet_alignment.get("axis2_use_cls_input", False)
        else:
            axis2_use_cls_input = True

        if axis2_use_cls_input:
            assert (
                image_backbone_cls is not None
            ), "axis2_use_cls_input=True requires image_backbone_cls"

        (
            scan_slice_emb,
            scan_valid_depth_mask,
            _cls,
            _local_cls,
            _perslice,
            _spatial_3d,
        ) = self._forward_axis2_combiner(
            feature_comb_image=self.feature_comb_image,
            image_grid_shape=image_grid_shape,
            window_mask=window_mask,
            image_backbone_patch_axis2=image_backbone_patch_axis2,
            image_backbone_cls=image_backbone_cls,
            axis2_use_cls_input=axis2_use_cls_input,
        )
        return scan_slice_emb, scan_valid_depth_mask

    def forward_localization(self, batch: dict) -> "LocalizationOutput":
        """
        Encode axis2 image + snippets for localization tasks.

        Returns a LocalizationOutput with `scan_slice_emb`, `scan_valid_depth_mask`,
        `snippet_emb`, `slice_target_depth_mask`, `slice_batch_idx_valid`.

        Returns an all-`None` LocalizationOutput when the batch carries no slices
        or no axis2 features
        """
        # .get on the two skip-trigger keys: this method is called from the
        # trainer for every micro-batch, and many batches legitimately have no
        # slices or no axis2 features (mixed-dataset training). Falling back to
        # the empty-output short-circuit is the documented behavior.
        slices = batch.get("slices")
        image_backbone_patch_axis2 = batch.get("image_backbone_patch_axis2")
        has_slices = slices is not None and len(slices) > 0
        if not has_slices or image_backbone_patch_axis2 is None:
            return LocalizationOutput()

        image_grid_shape = batch["image_grid_shape"]
        window_mask = batch["window_mask"]
        valid_slices = batch["valid_slices"]
        slice_batch_idx = batch["slice_batch_idx"]
        slice_target_depth_mask = batch["slice_target_depth_mask"]
        snippet_input_ids = batch["snippet_input_ids"]
        snippet_attention_mask = batch["snippet_attention_mask"]
        # Image-feature inputs depend on image_feat_mode; the active mode picks
        # the right pair below.
        image = batch.get("image")
        image_patches_mask = batch.get("image_patches_mask")
        image_backbone_cls = batch.get("image_backbone_cls")
        image_backbone_patch_average = batch.get("image_backbone_patch_average")

        window_mask = _resolve_window_mask(self.mask_usage, image_grid_shape, window_mask)

        if self.backbone_image is not None:
            image_backbone_cls, image_backbone_patch_average = _run_image_backbone(
                self.backbone_image,
                image,
                image_patches_mask,
                image_grid_shape,
                window_mask,
            )

        scan_slice_emb, scan_valid_depth_mask = self.encode_axis2_depth(
            image_grid_shape=image_grid_shape,
            window_mask=window_mask,
            image_backbone_cls=image_backbone_cls,
            image_backbone_patch_axis2=image_backbone_patch_axis2,
        )

        snippet_emb = self.encode_snippets(snippet_input_ids, snippet_attention_mask)
        slice_batch_idx_valid = slice_batch_idx[valid_slices]
        snippet_emb_valid = snippet_emb[valid_slices]

        return LocalizationOutput(
            scan_slice_emb=scan_slice_emb,
            scan_valid_depth_mask=scan_valid_depth_mask,
            snippet_emb=snippet_emb_valid,
            slice_target_depth_mask=slice_target_depth_mask,
            slice_batch_idx_valid=slice_batch_idx_valid,
        )

    def run_image_backbone_axis2(
        self,
        image: torch.Tensor,
        image_patches_mask: torch.Tensor | None,
        image_grid_shape: torch.Tensor,
        window_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the image backbone on raw windows and return per-window CLS plus the
        depth-axis (axis2) patch features that the localization path needs.

        `_run_image_backbone` only keeps the CLS + patch-average; localization also
        needs depth-resolved features. This reproduces the feature-export pooling
        (`save_embeddings_lib`): reshape each window's patch tokens to the patch grid
        and average over the two in-plane axes, keeping the depth axis. Both outputs
        are padded to the largest grid in the batch (padded windows are zero and get
        masked out later via `window_mask`).

        Returns:
            image_backbone_cls: (B, Hpmax, Wpmax, Dpmax, E)
            image_backbone_patch_axis2: (B, Hpmax, Wpmax, Dpmax, p2, E)
        """
        assert self.backbone_image is not None, "raw-image localization needs the image backbone"
        image_tokens = self.backbone_image(image, attn_mask_3d=image_patches_mask)  # (BN, 1+P, E)
        embed_dim = image_tokens.shape[-1]
        cls = image_tokens[:, 0, :]  # (BN, E)
        patches = image_tokens[:, 1:, :]  # (BN, P, E)

        window_size = self.backbone_image.sliding_window_size
        patch_size = self.backbone_image.patch_size
        p0, p1, p2 = (int(w // p) for w, p in zip(window_size, patch_size, strict=True))
        n_patches = patches.shape[1]
        assert n_patches == p0 * p1 * p2, f"{n_patches=} != {p0 * p1 * p2=} (patch grid mismatch)"
        # per-window depth features: average over the two in-plane patch axes, keep depth
        axis2 = patches.view(-1, p0, p1, p2, embed_dim).mean(dim=(1, 2))  # (BN, p2, E)

        batch_size = len(image_grid_shape)
        if window_mask is None:
            grid_shape = image_grid_shape.unique(dim=0).tolist()
            assert len(grid_shape) == 1, f"{window_mask=} but {image_grid_shape=}"
            Hp, Wp, Dp = grid_shape[0]
            cls = cls.view(batch_size, Hp, Wp, Dp, embed_dim)
            axis2 = axis2.view(batch_size, Hp, Wp, Dp, p2, embed_dim)
            return cls, axis2

        cls_list, axis2_list = [], []
        bstart = 0
        for Hp, Wp, Dp in image_grid_shape.tolist():
            bend = bstart + Hp * Wp * Dp
            cls_list.append(cls[bstart:bend].view(Hp, Wp, Dp, embed_dim))
            axis2_list.append(axis2[bstart:bend].view(Hp, Wp, Dp, p2, embed_dim))
            bstart = bend
        image_backbone_cls, mask_c, grid_c = pad_and_stack(cls_list)
        # pad_and_stack keys off the last (embed) dim; the axis2 mask/grid include the
        # extra p2 axis, so reuse window_mask (resolved by the caller) instead.
        image_backbone_patch_axis2, _, _ = pad_and_stack(axis2_list)
        assert eq(window_mask, mask_c), f"{window_mask=} != {mask_c=}"
        assert eq(image_grid_shape, grid_c), f"{image_grid_shape=} != {grid_c=}"
        return image_backbone_cls, image_backbone_patch_axis2

    def encode_localization(
        self,
        batch: dict,
        snippet_input_ids: torch.Tensor,
        snippet_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference localization from raw windows + snippet texts.

        Produces axis2 backbone features from the raw windows, projects them to
        per-depth image embeddings via the axis2 combiner + secondary image head,
        and encodes the snippets via the secondary text head. Compare them (dot /
        cosine) to localize each snippet along the scan depth.

        Returns:
            scan_slice_emb: (B, D2, projection_dim) per-depth image embeddings.
            scan_valid_depth_mask: (B, D2) True at valid scan depths.
            snippet_emb: (S, projection_dim) snippet text embeddings.
        """
        image_grid_shape = batch["image_grid_shape"]
        window_mask = _resolve_window_mask(
            self.mask_usage, image_grid_shape, batch.get("window_mask")
        )
        image_backbone_cls, image_backbone_patch_axis2 = self.run_image_backbone_axis2(
            batch.get("image"),
            batch.get("image_patches_mask"),
            image_grid_shape,
            window_mask,
        )
        scan_slice_emb, scan_valid_depth_mask = self.encode_axis2_depth(
            image_grid_shape=image_grid_shape,
            window_mask=window_mask,
            image_backbone_cls=image_backbone_cls,
            image_backbone_patch_axis2=image_backbone_patch_axis2,
        )
        snippet_emb = self.encode_snippets(snippet_input_ids, snippet_attention_mask)
        return scan_slice_emb, scan_valid_depth_mask, snippet_emb

    def forward(self, batch: dict) -> "GlobalContrastiveOutput":
        """
        Default forward dispatches to `forward_global_contrastive`.

        Kept so `model(batch)` continues to work for retrieval/pool/volume tasks.
        For explicit method calls, prefer `forward_global_contrastive`,
        `forward_image_only`, `forward_feature_export`, or `forward_localization`.
        """
        return self.forward_global_contrastive(batch)

    def _forward_axis2_combiner(
        self,
        feature_comb_image: FeatureVisionTransformer,
        image_grid_shape: torch.Tensor,
        window_mask: torch.Tensor,
        image_backbone_patch_axis2: torch.Tensor,
        image_backbone_cls: torch.Tensor | None = None,
        disable_detach: bool = False,
        axis2_use_cls_input=False,
    ):
        """
        Run axis2 combiner: project patches, run feature transformer, pool per-depth.

        When axis2_use_cls_input is True, constructs [cls_repeat_interleaved, axis2_patch]
        (dim 2*E) and uses forward_features() with the pretrained patch_proj. Otherwise
        uses axis_patch_proj → forward_features_projected().

        Returns:
            scan_slice_emb: (B, D2, 512) per-depth projected embeddings
            scan_valid_depth_mask: (B, D2) True for valid depth positions
            cls_token: (B, E_comb) CLS token from combiner
            perslice: (B, D2, E_comb) per-depth averaged spatial features
        """
        assert feature_comb_image is not None
        if not axis2_use_cls_input:
            assert feature_comb_image.axis_patch_proj is not None
        B, Hpmax, Wpmax, Dpmax, p2, E = image_backbone_patch_axis2.shape

        # feature mask for feature transformer is the upscaled window mask
        # (B, Hpmax, Wpmax, Dpmax) -> (B, Hpmax, Wpmax, Dpmax * p2)
        window_mask_axis2 = window_mask.repeat_interleave(p2, dim=-1)
        # same for the grid shape (B, 3) -> (B, 3) with last axis increased by p2
        image_grid_shape_axis2 = image_grid_shape.clone()
        image_grid_shape_axis2[:, 2] *= p2

        # valid depth mask is all the slices that are valid in the full scan
        scan_valid_depth_mask = window_mask_axis2.any(dim=(1, 2))  # (B, Dpmax * p2)
        assert (scan_valid_depth_mask.sum(-1) == image_grid_shape_axis2[:, 2]).all()

        # Get axis2 features for each snippet's scan
        axis2 = image_backbone_patch_axis2  # (B, Hp, Wp, Dp, p2, E)
        _, Hpmax, Wpmax, Dpmax, p2, E = axis2.shape
        # note other axes would need a permute first
        axis2_flat = axis2.reshape(B, Hpmax * Wpmax * Dpmax * p2, E)

        # coord_divisor for physical rope mode
        coord_divisor = None
        if self.do_snippet_alignment.get("rope_mode", "regular") == "physical":
            # hack rope division to work as in pretraining
            coord_divisor = image_grid_shape

        loc_mode = self.do_snippet_alignment.get("localization_slice_mode", "default")
        if axis2_use_cls_input:
            # Use pretrained input format: [CLS_repeated, axis2_patch] (dim 2*E)
            # then pretrained patch_proj (2*E → E_comb) via forward_features()
            assert image_backbone_cls is not None, "axis2_use_cls_input requires image_backbone_cls"
            # image_backbone_cls: (B, Hp, Wp, Dp, E) → repeat each window CLS p2 times along depth
            cls_flat = image_backbone_cls.repeat_interleave(p2, dim=3).reshape(
                B, Hpmax * Wpmax * Dpmax * p2, E
            )
            # concat to match pretrained format: [CLS, avg_patch] per position → (B, N, 2*E)
            combined = torch.cat([cls_flat, axis2_flat], dim=2)
            combiner_output = feature_comb_image.forward_features(
                combined,
                grid_size=image_grid_shape_axis2,
                attn_mask_3d=window_mask_axis2,
                coord_divisor=coord_divisor,
            )
            # reshape output back to 3D
            axis2_cls_token, axis2_local_cls_token, axis2_spatial_3d = (
                _unpack_feature_combiner_output(
                    feature_comb_image, combiner_output, Hpmax, Wpmax, Dpmax * p2
                )
            )
        elif loc_mode == "sep":
            # separately input the slices to the projector
            axis2_proj = feature_comb_image.axis_patch_proj(axis2_flat)  # (B, N, E_comb)
            # where N = Hp*Wp*Dp*p2
            axis2_proj_3d = axis2_proj.view(B, Hpmax, Wpmax, Dpmax * p2, -1)
            axis2_proj_batched = axis2_proj_3d.permute(0, 3, 1, 2, 4).reshape(
                B * Dpmax * p2, Hpmax, Wpmax, -1
            )
            new_axis = axis2_proj_batched.view(B * Dpmax * p2, Hpmax * Wpmax, -1)
            new_grid = image_grid_shape_axis2[:, :2].repeat_interleave(Dpmax * p2, dim=0)
            new_grid = torch.cat([new_grid, new_grid.new_ones((new_grid.size(0), 1))], dim=1)
            new_mask = window_mask_axis2.permute(0, 3, 1, 2).reshape(
                B * Dpmax * p2, Hpmax, Wpmax, 1
            )

            combiner_output = feature_comb_image.forward_features_projected(
                new_axis,
                grid_size=new_grid,
                attn_mask_3d=new_mask,
                coord_divisor=None,
            )
            slice_cls, slice_local_cls, slice_spatial_3d = _unpack_feature_combiner_output(
                feature_comb_image, combiner_output, Hpmax, Wpmax, 1
            )
            new_mask_3d = new_mask.view(B, Dpmax * p2, Hpmax, Wpmax)
            slice_cls_3d = slice_cls.view(B, Dpmax * p2, -1)
            new_mask_sum = new_mask_3d.sum(dim=(2, 3)).unsqueeze(-1)  # (B, Dpmax * p2, 1)
            axis2_cls_token = (slice_cls_3d * new_mask_sum).sum(1) / new_mask_sum.sum(1).clamp(
                min=1
            )  # (B, E_comb)
            slice_local_cls_3d = slice_local_cls.view(B, Dpmax * p2, -1)
            axis2_local_cls_token = (slice_local_cls_3d * new_mask_sum).sum(1) / new_mask_sum.sum(
                1
            ).clamp(
                min=1
            )  # (B, E_comb)
            axis2_spatial_3d = slice_spatial_3d.view(B, Dpmax * p2, Hpmax, Wpmax, -1).permute(
                0, 2, 3, 1, 4
            )
        else:
            # Original path: axis_patch_proj (E → E_comb) → forward_features_projected()
            axis2_proj = feature_comb_image.axis_patch_proj(axis2_flat)
            combiner_output = feature_comb_image.forward_features_projected(
                axis2_proj,
                grid_size=image_grid_shape_axis2,
                attn_mask_3d=window_mask_axis2,
                coord_divisor=coord_divisor,
            )
            # reshape output back to 3D
            axis2_cls_token, axis2_local_cls_token, axis2_spatial_3d = (
                _unpack_feature_combiner_output(
                    feature_comb_image, combiner_output, Hpmax, Wpmax, Dpmax * p2
                )
            )

        # average the valid tokens of axes 0, 1 to get per-slice features
        perslice_sum = (axis2_spatial_3d * window_mask_axis2.unsqueeze(-1)).sum(dim=(1, 2))
        perslice_count = window_mask_axis2.sum(dim=(1, 2))
        # clamp to avoid 0/0 NaN at invalid depth positions (count=0 → NaN grad in backward)
        axis2_perslice = perslice_sum / perslice_count.clamp(min=1).unsqueeze(-1)  # (B, D2, E_comb)

        # the projector expects 2*E_comb features (CLS + avg_patch per depth)
        # use local_cls_out for per-depth projections (specializes for local features)
        local_cls_rep = axis2_local_cls_token.unsqueeze(1).repeat(1, Dpmax * p2, 1)
        perslice_proj_input = torch.cat([local_cls_rep, axis2_perslice], dim=2)
        scan_slice_emb = self.project_slice_vision_features(
            perslice_proj_input, disable_detach=disable_detach
        )  # (B, D2, 512)

        return (
            scan_slice_emb,
            scan_valid_depth_mask,
            axis2_cls_token,
            axis2_local_cls_token,
            axis2_perslice,
            axis2_spatial_3d,
        )

    def project_slice_vision_features(self, slice_proj_input, disable_detach=False):
        detach_grad = self.do_snippet_alignment.get("localization_detach_grad_image", None)
        if detach_grad is None:
            detach_grad = self.do_snippet_alignment.get("localization_detach_grad", False)
        if detach_grad and not disable_detach:
            # train only this projection head with the localization loss, not the entire model
            slice_proj_input = slice_proj_input.detach()
        if self.projection_image_copy is not None:
            slice_vision_emb = self.projection_image_copy(slice_proj_input)
        else:
            slice_vision_emb = self.projection_image(slice_proj_input)
        return slice_vision_emb

    def project_slice_text_features(self, slice_proj_input, disable_detach=False):
        detach_grad = self.do_snippet_alignment.get("localization_detach_grad_text", None)
        if detach_grad is None:
            detach_grad = self.do_snippet_alignment.get("localization_detach_grad", False)
        if detach_grad and not disable_detach:
            # train only this projection head with the localization loss, not the entire model
            slice_proj_input = slice_proj_input.detach()
        if self.projection_text_copy is not None:
            slice_text_emb = self.projection_text_copy(slice_proj_input)
        else:
            slice_text_emb = self.projection_text(slice_proj_input)
        return slice_text_emb

    # ---------- checkpoint loading and state management ----------

    def load_checkpoint(self, ckpt_file: Path | str, device: str = "cpu", allow_unexpected=False):
        """Load checkpoint into SigLIP model."""
        ckpt_file = Path(ckpt_file)
        if not ckpt_file.is_absolute():
            ckpt_file = get_medv_output_dir() / ckpt_file
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        if ckpt_file.is_dir() or ckpt_file.name.endswith(".safetensors"):
            hf_state = load_safetensors_state(ckpt_file)
            m = "model."
            state = {(k[len(m) :] if k.startswith(m) else k): v for k, v in hf_state.items()}
            info_str = f"from safetensors: {ckpt_file.as_posix()}"
        else:
            log_info(f"Loading checkpoint from {ckpt_file}")
            checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
            state = checkpoint["model"]
            epoch = checkpoint["epoch"]
            info_str = f"from epoch {epoch}"
        load_result = self.load_state_dict(state, strict=False)

        mk, uk = load_result.missing_keys, load_result.unexpected_keys
        assert len(mk) < len(self.state_dict()), (
            f"No keys in {ckpt_file} matched the model. Checkpoint keys look like "
            f"{list(state)[:2]}, model keys look like {list(self.state_dict())[:2]}"
        )
        if mk:
            log_info(f"Missing keys: {len(mk)} (frozen parameters)\n{mk}")
        if uk:
            self._handle_unexpected_keys(uk, ckpt_file, allow_unexpected)

        log_info(f"Checkpoint loaded successfully {info_str}")
        return mk

    def _handle_unexpected_keys(self, uk: list[str], ckpt_file: Path, allow_unexpected: bool):
        """
        Tolerate keys of submodules that are not built at all (reduced feat modes)
        and the known auxiliary training modules; anything else is fatal.
        """
        model_roots = {k.split(".", 1)[0] for k in self.state_dict()}
        absent_roots = sorted({k.split(".", 1)[0] for k in uk} - model_roots)
        if absent_roots:
            uk = [k for k in uk if k.split(".", 1)[0] in model_roots]
            log_info(f"Skipping checkpoint keys of submodules that are not built: {absent_roots}")
        if not uk:
            return
        # ignore a model that was trained with some auxiliary modules, and in the current
        # inference doesn't need those modules.
        ignore_keys = [
            "projection_image_copy.*",
            "projection_text_copy.*",
            "feature_comb_image.axis_patch_proj.*",
            "feature_comb_image.local_cls_token",
        ]
        spec = make_git_pathspec(ignore_keys)
        unexpected_keys = list(spec.match_files(uk, negate=True))
        unexpected_keys_counts = defaultdict(int)
        for k in unexpected_keys:
            base = str(k).split(".")[0]
            unexpected_keys_counts[base] += 1
        unexpected_keys_repr = dict_to_str_comma_equals(dict(unexpected_keys_counts))
        if len(unexpected_keys) == 0:
            log_warning(f"Removed unexpected keys: {ignore_keys}")
        elif allow_unexpected:
            log_warning(f"Ignored unexpected keys: {unexpected_keys_repr}")
        else:
            log_error("*" * 80)
            log_error(f"Unexpected keys: {uk}")
            log_error(f"Unexpected keys counts: {unexpected_keys_repr}")
            log_error(
                "Keys above are unexpected! A module is now frozen that was not frozen before?"
            )
            log_error("*" * 80)
            raise ValueError(f"Checkpoint architecture mismatch in {ckpt_file} (See above)")

    def set_frozen_state(
        self, model_config: dict, train_config: dict, is_init=True, epoch: int = 0
    ):
        """
        Args:
            is_init: True if starting training, False if per-epoch update
            model_config: model configuration
        """
        image_backbone = self.backbone_image
        text_backbone = self.backbone_text

        # weights to freeze or thaw at the beginning of each epoch
        freeze_image_backbone_epochs = train_config["optim"]["freeze_image_backbone_epochs"]
        if freeze_image_backbone_epochs != 0 and image_backbone is not None:
            if freeze_image_backbone_epochs == -1:
                freeze = True
            elif epoch < freeze_image_backbone_epochs:
                freeze = True
            else:
                freeze = False
            log_info(f"Backbone freeze: {epoch=} {freeze_image_backbone_epochs=} {freeze=}")
            for param in image_backbone.parameters():
                param.requires_grad = not freeze

        for freeze_setting_key, layer in [
            ("freeze_image_proj_last_layer_epochs", self.projection_image.last_layer),
            ("freeze_text_proj_last_layer_epochs", self.projection_text.last_layer),
        ]:
            freeze_setting = train_config["optim"][freeze_setting_key]
            if freeze_setting == 0 or layer is None:
                continue
            if freeze_setting == -1 or epoch < freeze_setting:
                freeze = True
            else:
                freeze = False
            log_info(f"{freeze_setting_key} freeze: {epoch=} {freeze_setting=} {freeze=}")
            for param in layer.parameters():  # type: ignore
                param.requires_grad = not freeze
            # this part should be kept frozen always
            layer.parametrizations.weight.original0.requires_grad = False  # type: ignore

        if not is_init:
            return

        # weights to freeze at the start of training
        if image_backbone is not None:
            if train_config["optim"]["freeze_patch_embedding"]:
                log_info("Freezing image backbone patch embedding layer")
                for p in image_backbone.patch_embed.parameters():  # type: ignore
                    p.requires_grad = False

        if text_backbone is not None:
            # Freeze all parameters except LoRA adapters in text backbone
            text_backbone_kwargs = model_config["text_backbone_kwargs"]
            if text_backbone_kwargs["use_lora"] and text_backbone_kwargs["lora_r"] > 0:
                # Only LoRA parameters are trainable
                for n, p in text_backbone.named_parameters():
                    p.requires_grad = "lora_" in n
                trainable = sum(p.numel() for p in text_backbone.parameters() if p.requires_grad)
                total = sum(p.numel() for p in text_backbone.parameters())
                log_info(
                    f"Text backbone: LoRA enabled. Trainable: {trainable:,} / {total:,} parameters"
                )
            else:
                # Freeze all text backbone parameters
                for p in text_backbone.parameters():
                    p.requires_grad = False
                text_backbone.eval()
                log_info("Text backbone: all parameters frozen (no LoRA)")

        if self.projection_image is not None:
            if train_config["optim"]["freeze_image_projection"]:
                log_info("Freezing image projection layer")
                for p in self.projection_image.parameters():
                    p.requires_grad = False

        if self.projection_text is not None:
            if train_config["optim"]["freeze_text_projection"]:
                log_info("Freezing text projection layer")
                for p in self.projection_text.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.mask_usage = self.mask_usage_train if mode else self.mask_usage_eval


# ---------- forward helpers ----------


def _resolve_window_mask(
    mask_usage: str,
    image_grid_shape: torch.Tensor,
    window_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    n_shapes = len(image_grid_shape.unique(dim=0))

    MaskUsageC.verify_value(mask_usage)
    if mask_usage == MaskUsageC.DYNAMIC:
        # for batch size 1 the mask should not be needed because there is no padding.
        use_mask = n_shapes > 1
    elif mask_usage == MaskUsageC.TRUE:
        use_mask = True
    elif mask_usage == MaskUsageC.FALSE:
        use_mask = False
    else:
        raise ValueError(f"Unknown mask_usage: {mask_usage}")

    if use_mask:
        assert window_mask is not None, "window_mask must be provided if mask_usage is TRUE"
        return window_mask

    assert n_shapes == 1, f"{mask_usage=} but {image_grid_shape=} -> {n_shapes=}"
    return None


def _run_image_backbone(
    image_backbone: nn.Module,
    image: torch.Tensor,
    image_patches_mask: torch.Tensor | None,
    image_grid_shape: torch.Tensor,
    window_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # run backbone.
    # patch size is 16x16x8, so there are 8x8x8 = 512 patches per window + 1 cls token
    image_tokens = image_backbone(
        image, attn_mask_3d=image_patches_mask
    )  # (B*N, n_patches=513, emb_dim)
    image_backbone_cls = image_tokens[:, 0, :]  # (B*N, emb_dim)
    image_backbone_patch_average = image_tokens[:, 1:, :].mean(dim=1)  # (B*N, emb_dim)

    batch_size = len(image_grid_shape)
    if window_mask is None:
        # fixed batch size, no masks etc. needed
        grid_shape = image_grid_shape.unique(dim=0).tolist()
        assert len(grid_shape) == 1, f"{window_mask=} but {image_grid_shape=}"
        Hp, Wp, Dp = grid_shape[0]
        image_backbone_cls = image_backbone_cls.view(batch_size, Hp, Wp, Dp, -1)
        image_backbone_patch_average = image_backbone_patch_average.view(batch_size, Hp, Wp, Dp, -1)
        return image_backbone_cls, image_backbone_patch_average

    # variable batch size, need to pad in 3d
    ibb_cls_list, ibb_pavg_list = [], []
    bstart = 0
    for _bi, (Hp, Wp, Dp) in enumerate(image_grid_shape):
        bend = bstart + Hp * Wp * Dp
        ibb_cls = image_backbone_cls[bstart:bend, :]  # (N, emb_dim)
        ibb_pavg = image_backbone_patch_average[bstart:bend, :]
        ibb_cls_list.append(ibb_cls.view(Hp, Wp, Dp, -1))
        ibb_pavg_list.append(ibb_pavg.view(Hp, Wp, Dp, -1))
        bstart = bend
    image_backbone_cls, mask1, grid1 = pad_and_stack(ibb_cls_list)
    image_backbone_patch_average, mask2, grid2 = pad_and_stack(ibb_pavg_list)
    assert eq(window_mask, mask1), f"{window_mask=} != {mask1=}"
    assert eq(image_grid_shape, grid1), f"{image_grid_shape=} != {grid1=}"
    assert eq(window_mask, mask2), f"{window_mask=} != {mask2=}"
    assert eq(image_grid_shape, grid2), f"{image_grid_shape=} != {grid2=}"
    return image_backbone_cls, image_backbone_patch_average


def _run_lowres_feature_combiner(
    feature_comb_image: FeatureVisionTransformer,
    image_backbone_cls: torch.Tensor,
    image_backbone_patch_average: torch.Tensor,
    image_grid_shape: torch.Tensor,
    window_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # backbone_cls and backbone_patch_average are either computed above or passed in
    batch_size, Hpmax, Wpmax, Dpmax, embed_dim = image_backbone_cls.shape
    n_windows = Hpmax * Wpmax * Dpmax
    backbone_cls = image_backbone_cls.view(batch_size, n_windows, embed_dim)
    backbone_patch_average = image_backbone_patch_average.view(batch_size, n_windows, embed_dim)
    # Combine cls and mean of patches for feature combiner input: (B, N, 2 * embed_dim)
    image_feature_comb_input = torch.cat([backbone_cls, backbone_patch_average], dim=2)

    if window_mask is None:
        # batch size 1 or all shapes are equal, regular call
        image_tokens = feature_comb_image(image_feature_comb_input, grid_size=(Hpmax, Wpmax, Dpmax))
    else:
        # pass mask and grids
        image_tokens = feature_comb_image(
            image_feature_comb_input,
            grid_size=image_grid_shape,
            attn_mask_3d=window_mask,
        )
    # output is (B, N, embed_dim)
    image_feature_comb_cls, image_feature_comb_cls_secondary, image_feature_comb_patch = (
        _unpack_feature_combiner_output(feature_comb_image, image_tokens, Hpmax, Wpmax, Dpmax)
    )
    return (
        image_feature_comb_input,
        image_feature_comb_cls,
        image_feature_comb_cls_secondary,
        image_feature_comb_patch,
    )


def _pool_feature_combiner_patch(
    image_feature_comb_patch: torch.Tensor,
    window_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, _, _, _, embed_dim = image_feature_comb_patch.shape
    image_feature_comb_patch_flat = image_feature_comb_patch.view(batch_size, -1, embed_dim)
    # average pool the feature combiner spatial tokens
    if window_mask is None:
        return image_feature_comb_patch_flat.mean(dim=1)
    # take mask into account for avg pooling
    return masked_average(image_feature_comb_patch_flat, window_mask.view(batch_size, -1))


def _unpack_feature_combiner_output(
    feature_comb_image: FeatureVisionTransformer,
    combiner_output: torch.Tensor,
    Hpout: int,
    Wpout: int,
    Dpout: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, _, E_comb = combiner_output.shape
    has_local_cls = feature_comb_image.local_cls_token is not None
    n_prefix = 2 if has_local_cls else 1
    cls_token = combiner_output[:, 0, :]  # global CLS (B, E_comb)
    local_cls_out = combiner_output[:, 1, :] if has_local_cls else cls_token
    spatial_tokens = combiner_output[:, n_prefix:, :]  # skip prefix -> (B, Hp*Wp*D2, E_comb)
    spatial_3d = spatial_tokens.view(B, Hpout, Wpout, Dpout, E_comb)
    return cls_token, local_cls_out, spatial_3d
