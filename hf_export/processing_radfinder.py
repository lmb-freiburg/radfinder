"""
Image processor for HF RadFinder.

Accepts a list of NIfTI file paths and returns `{"pixel_values", "grid_size"}`
"""

from pathlib import Path
from typing import Sequence

import torch
from transformers.image_processing_utils import BaseImageProcessor

try:
    from .configuration_radfinder import RadFinderConfig
except ImportError:
    from configuration_radfinder import RadFinderConfig

from radfinder.models.load_model import FeatMode
from radfinder.transforms.eval_transform import TextTransformMode, get_eval_transform


class RadFinderImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "grid_size"]

    def __init__(
        self,
        pixdim: Sequence[float] = (0.75, 0.75, 3.0),
        sliding_window_size: Sequence[int] = (128, 128, 32),
        min_area_for_padding: float = 0.0,
        dtype: str = "float16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pixdim = tuple(pixdim)
        self.sliding_window_size = tuple(sliding_window_size)
        self.min_area_for_padding = min_area_for_padding
        self.dtype = dtype
        self._transform = get_eval_transform(
            pixdim=self.pixdim,
            sliding_window_size=self.sliding_window_size,
            image_feat_mode=FeatMode.FULL,
            text_feat_mode=FeatMode.NONE,
            text_transform_mode=TextTransformMode.NONE,
            min_area_for_padding=self.min_area_for_padding,
            dtype=self.dtype,
        )

    @classmethod
    def from_radfinder_config(cls, config: RadFinderConfig) -> "RadFinderImageProcessor":
        prep = config.image_preprocessing
        return cls(
            pixdim=prep.get("pixdim", (0.75, 0.75, 3.0)),
            sliding_window_size=prep.get("sliding_window_size", (128, 128, 32)),
            min_area_for_padding=prep.get("min_area_for_padding", 0.0),
            dtype=prep.get("dtype", "float16"),
        )

    def to_dict(self) -> dict:
        # Drop the live MONAI Compose so `save_pretrained` can serialize the
        # rest. The transform is rebuilt from the JSON-roundtrippable fields
        # (pixdim, sliding_window_size, ...) on the next __init__.
        output = super().to_dict()
        output.pop("_transform", None)
        return output

    def __call__(self, image_paths: str | Path | Sequence[str | Path]) -> dict:
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        outputs = []
        for path in image_paths:
            sample = {"image": str(path)}
            outputs.append(self._transform(sample))

        images = [s["image"] for s in outputs]  # each (N, C, H, W, D)
        grids = [_infer_grid_from_locations(s) for s in outputs]
        # return plain tensors: the MONAI MetaTensor metadata is only needed for the
        # grid inference above, and MetaTensor ops fail inside the model forward
        images = [img.as_tensor() if hasattr(img, "as_tensor") else img for img in images]
        for image, grid in zip(images, grids):
            assert (
                image.shape[0] == grid[0] * grid[1] * grid[2]
            ), f"Window count N={image.shape[0]} does not match inferred grid_size={grid}"
        return {
            "pixel_values": torch.cat(images, dim=0),
            "grid_size": torch.tensor(grids, dtype=torch.int32),
        }


def _infer_grid_from_locations(sample: dict) -> tuple[int, int, int]:
    """Read the GridPatchd-produced 'location' metadata to derive the
    (Hg, Wg, Dg) grid layout."""
    image = sample["image"]
    meta = image.meta if hasattr(image, "meta") else sample.get("image_meta_dict", {})
    loc = meta.get("location")
    if loc is None:
        raise KeyError(
            "Missing 'location' metadata. Did GridPatchd run? Check the eval " "transform pipeline."
        )
    # loc shape: (3, N) — three axis offsets per window.
    import numpy as np

    loc = np.asarray(loc)
    if loc.shape[0] != 3:
        loc = loc.T
    unique_per_axis = [len(np.unique(loc[axis])) for axis in range(3)]
    return tuple(unique_per_axis)
