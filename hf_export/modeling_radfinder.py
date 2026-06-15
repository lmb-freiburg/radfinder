"""
HuggingFace model wrapper around the radfinder SigLIP model.

Allows loading via `AutoModel.from_pretrained(repo, trust_remote_code=True)`.
"""

import torch
from radfinder.models.load_model import FeatMode, create_siglip
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

try:
    from .configuration_radfinder import RadFinderConfig
except ImportError:
    from configuration_radfinder import RadFinderConfig


class RadFinderModel(PreTrainedModel):
    config_class = RadFinderConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def __init__(self, config: RadFinderConfig):
        super().__init__(config)
        self.model = create_siglip(
            config.radfinder_model_config,
            image_feat_mode=FeatMode.FULL,
            text_feat_mode=FeatMode.FULL,
            do_snippet_alignment=config.do_snippet_alignment,
            model_settings=config.model_settings,
        )
        self.model.eval()
        self.post_init()

    def check_supports_localization(self) -> bool:
        """True iff this checkpoint was trained for axis snippet localization."""
        sa = self.model.do_snippet_alignment or {}
        return bool(sa.get("enabled", False)) and sa.get("snippet_mode") == "axis_localization"

    def encode_image_retrieval(
        self,
        pixel_values: torch.Tensor,
        grid_size: torch.Tensor,
    ) -> torch.Tensor:
        """Encode CT window grids to global retrieval image embeddings.

        Args:
            pixel_values: (N, C, H, W, D) windows of all scans concatenated in batch
                order, N = sum over scans of Hg*Wg*Dg.
            grid_size: (B, 3) int tensor, per-scan window grid (Hg, Wg, Dg).

        Returns:
            Image embeddings, shape (B, projection_dim).
        """
        batch = _build_image_batch(pixel_values, grid_size)
        return self.model.forward_image_only(batch).image_embeddings

    def encode_image_disease(
        self,
        pixel_values: torch.Tensor,
        grid_size: torch.Tensor,
    ) -> torch.Tensor:
        """Encode CT window grids to disease image embeddings.

        Returns:
            Image embeddings, shape (B, projection_dim).
        """
        batch = _build_image_batch(pixel_values, grid_size)
        return self.model.forward_image_only(batch).image_embeddings_secondary

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokenized text to projected text embeddings.

        Args:
            input_ids: (B, L) tokenizer output.
            attention_mask: (B, L) tokenizer output.

        Returns:
            Text embeddings, shape (B, projection_dim).
        """
        return self.model.encode_text_report(input_ids, attention_mask)

    def localize(
        self,
        pixel_values: torch.Tensor,
        grid_size: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
    ):
        """Localize snippet texts along the scan depth axis.

        Encodes per-depth image embeddings (axis2 combiner + secondary image head)
        and snippet text embeddings (secondary text head). Score them, e.g.
        `einsum("bde,se->bsd", normalize(scan_slice_emb), normalize(snippet_emb))`,
        to get a per-depth heatmap for each snippet.

        Args:
            pixel_values: (N, C, H, W, D) windows, as in `encode_image_retrieval`.
            grid_size: (B, 3) int tensor, per-scan window grid (Hg, Wg, Dg).
            input_ids: (S, L) tokenized snippets.
            attention_mask: (S, L) tokenized snippets.

        Returns:
            scan_slice_emb (B, D2, projection_dim), scan_valid_depth_mask (B, D2),
            snippet_emb (S, projection_dim)
        """
        if not self.check_supports_localization():
            raise NotImplementedError(
                "This checkpoint was not trained for localization (do_snippet_alignment "
                "enabled + snippet_mode='axis_localization')"
            )
        batch = _build_image_batch(pixel_values, grid_size)
        scan_slice_emb, scan_valid_depth_mask, snippet_emb = self.model.encode_localization(
            batch, input_ids, attention_mask
        )
        if not return_dict:
            return scan_slice_emb, scan_valid_depth_mask, snippet_emb
        return ModelOutput(
            scan_slice_emb=scan_slice_emb,
            scan_valid_depth_mask=scan_valid_depth_mask,
            snippet_emb=snippet_emb,
        )

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        grid_size: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs,
    ):
        if pixel_values is not None and input_ids is None:
            assert grid_size is not None
            image_embeddings = self.encode_image_retrieval(pixel_values, grid_size)
            text_embeddings = None
        elif input_ids is not None and pixel_values is None:
            image_embeddings = None
            text_embeddings = self.encode_text(input_ids, attention_mask)
        else:
            assert grid_size is not None
            batch = _build_image_batch(pixel_values, grid_size)
            batch["report_input_ids"] = input_ids
            batch["report_hidden_state_mask"] = attention_mask
            out = self.model.forward_global_contrastive(batch)
            image_embeddings = out.image_embeddings
            text_embeddings = out.text_embeddings

        if not return_dict:
            return image_embeddings, text_embeddings

        return ModelOutput(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
        )


def _build_image_batch(pixel_values: torch.Tensor, grid_size: torch.Tensor) -> dict:
    """Build the batch dict the underlying SigLIP forward methods expect:
    concatenated windows, per-scan grid shapes, and the mask marking each scan's
    valid windows within the largest grid."""
    assert (
        pixel_values.ndim == 5
    ), f"pixel_values must be (N, C, H, W, D), got shape {tuple(pixel_values.shape)}"
    grid_size = torch.as_tensor(grid_size, dtype=torch.int32, device=pixel_values.device)
    assert (
        grid_size.ndim == 2 and grid_size.shape[1] == 3
    ), f"grid_size must be (B, 3), got shape {tuple(grid_size.shape)}"
    n_windows = int(grid_size.detach().cpu().numpy().prod(axis=1).sum())
    assert pixel_values.shape[0] == n_windows, (
        f"Expected N={n_windows} windows for grid_size={grid_size.tolist()}, "
        f"got {pixel_values.shape[0]}"
    )
    B = len(grid_size)
    Hmax, Wmax, Dmax = grid_size.max(dim=0).values.tolist()
    window_mask = torch.zeros((B, Hmax, Wmax, Dmax), dtype=torch.bool, device=pixel_values.device)
    for i, (Hg, Wg, Dg) in enumerate(grid_size.tolist()):
        window_mask[i, :Hg, :Wg, :Dg] = True
    return {
        "image": pixel_values,
        "image_grid_shape": grid_size,
        "window_mask": window_mask,
    }
