"""
Utilities for updating metadata after eval-time crop and padding.
"""

from __future__ import annotations

from typing import Sequence

import torch


def crop_affine_for_eval(
    affine: torch.Tensor | list[list[float]],
    spatial_shape: Sequence[int],
    sliding_window_size: Sequence[int],
    min_area_for_padding: float = 0.0,
) -> torch.Tensor:
    """
    Update affine for eval-time crop/pad (LargestMultipleCenterCropd + SpatialPadd).
    """
    assert min_area_for_padding >= 0.0, "min_area_for_padding must be >= 0.0"
    crop_size = []
    crop_start = []
    pad_left = []
    for size, patch in zip(spatial_shape, sliding_window_size):
        if min_area_for_padding <= 0.0:
            multiple = size // patch
            crop = multiple * patch if multiple > 0 else size
            crop_size.append(crop)
            crop_start.append(max((size - crop) // 2, 0))
            pad_left.append(0)
            continue

        multiple = size // patch
        remainder = size - (multiple * patch)
        target = multiple * patch
        if multiple == 0:
            target = patch
        elif remainder > min_area_for_padding * patch:
            target = (multiple + 1) * patch
        crop = min(size, target)
        crop_size.append(crop)
        crop_start.append(max((size - crop) // 2, 0))
        pad_total = max(target - crop, 0)
        pad_left.append(pad_total // 2)

    affine_tensor = torch.as_tensor(affine, dtype=torch.float64)
    affine_tensor = _apply_index_shift_to_affine(affine_tensor, tuple(crop_start))
    affine_tensor = _apply_index_shift_to_affine(
        affine_tensor, (-pad_left[0], -pad_left[1], -pad_left[2])
    )
    return affine_tensor


def _apply_index_shift_to_affine(affine: torch.Tensor, shift: tuple[int, int, int]) -> torch.Tensor:
    shift_tensor = torch.tensor(shift, dtype=affine.dtype, device=affine.device)
    world_shift = affine[:3, :3] @ shift_tensor
    affine_out = affine.clone()
    affine_out[:3, 3] = affine_out[:3, 3] + world_shift
    return affine_out
