from typing import Sequence

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import CenterSpatialCrop, Cropd, SpatialPad


class LargestMultipleCenterCropd(Cropd):
    """
    Dictionary-based transform for channel-first arrays only.

    Args:
        keys: keys of the corresponding items to be transformed.
        patch_size: sequence of ints, e.g. (128, 128, 64). Number of components must match image spatial dims.
        min_area_for_padding: if > 0, enable padding to next multiple when remainder exceeds this fraction.
        allow_missing_keys: don't raise if key missing.
        lazy: whether the internal cropper should be lazy.
    """

    def __init__(
        self,
        keys: KeysCollection,
        patch_size: Sequence[int],
        min_area_for_padding: float = 0.0,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        self.patch_size = patch_size
        self.min_area_for_padding = min_area_for_padding
        assert self.min_area_for_padding >= 0.0, "min_area_for_padding must be >= 0.0"
        cropper = CenterSpatialCrop(roi_size=patch_size, lazy=lazy)
        padder = SpatialPad(spatial_size=patch_size, mode="constant", value=0.0, lazy=lazy)
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)
        self.padder = padder

    def compute_crop_and_target_sizes(
        self, img: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        spatial_dims = np.asarray(img.shape[1:], dtype=int)  # Exclude channel dim
        patch_size = np.asarray(self.patch_size, dtype=int)
        multiples = spatial_dims // patch_size
        remainder = spatial_dims - (multiples * patch_size)
        target_size = multiples * patch_size
        target_size = np.where(multiples == 0, patch_size, target_size)
        threshold = self.min_area_for_padding * patch_size
        target_size = np.where(remainder > threshold, (multiples + 1) * patch_size, target_size)
        crop_size = np.minimum(spatial_dims, target_size)
        return (
            tuple(int(x) for x in crop_size),
            tuple(int(x) for x in target_size),
        )

    def compute_largest_multiple_crop(self, img: np.ndarray) -> np.ndarray:
        spatial_dims = img.shape[1:]  # Exclude channel dim
        patch_size = np.asarray(self.patch_size, dtype=int)
        multiples = np.array(spatial_dims) // patch_size
        crop_size = (multiples * patch_size).astype(int)
        crop_size = np.where(crop_size == 0, spatial_dims, crop_size)
        return tuple(int(x) for x in crop_size)

    def __call__(self, data: dict, lazy: bool | None = None) -> dict:
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        assert (
            self.min_area_for_padding == 0.0 or not lazy_
        ), f"{self.min_area_for_padding=} {lazy_=} incomatible / not implemented."
        for key in self.key_iterator(d):
            if self.min_area_for_padding <= 0.0:
                # reset cropper based on current image shape
                self.cropper.roi_size = self.compute_largest_multiple_crop(d[key].cpu().numpy())
                d[key] = self.cropper(d[key], lazy=lazy_)  # type: ignore
                continue
            crop_size, target_size = self.compute_crop_and_target_sizes(d[key].cpu().numpy())
            self.cropper.roi_size = crop_size
            d[key] = self.cropper(d[key], lazy=lazy_)  # type: ignore
            if any(target_size[i] > crop_size[i] for i in range(len(target_size))):
                self.padder.spatial_size = target_size
                d[key] = self.padder(d[key])
        return d


def get_per_window_patches_masks(
    spatial_shape_final: np.ndarray,  # (B, 3)
    image_grid_shape: torch.Tensor,  # (B, 3)
    patch_size: tuple[int, int, int],  # 16, 16, 4
    sliding_window_size: tuple[int, int, int],  # 128, 128, 32
    min_area_for_padding: float,
):
    """
    Build per-window patch masks for crop+pad pipelines.

    Returns:
        torch.Tensor[bool] with shape (sum_b Hp*Wp*Dp, pH, pW, pD), where True means
        the patch contains valid (non-zero) voxels and should be kept.
    """
    assert min_area_for_padding >= 0.0, "min_area_for_padding must be >= 0.0"
    assert (
        image_grid_shape.ndim == 2 and image_grid_shape.shape[1] == 3
    ), f"{image_grid_shape.shape=}"
    assert len(sliding_window_size) == 3, f"{sliding_window_size=}"
    assert len(patch_size) == 3, f"{patch_size=}"
    assert all(
        win % patch == 0 for win, patch in zip(sliding_window_size, patch_size)
    ), f"{sliding_window_size=} must be divisible by {patch_size=}"

    if isinstance(spatial_shape_final, np.ndarray):
        shape_final_tensor = torch.from_numpy(spatial_shape_final)
    else:
        shape_final_tensor = torch.as_tensor(spatial_shape_final)
    shape_final_tensor = shape_final_tensor.to(dtype=torch.int64, device=image_grid_shape.device)
    grid_shape = image_grid_shape.to(dtype=torch.int64)
    assert (
        shape_final_tensor.shape == grid_shape.shape
    ), f"{shape_final_tensor.shape=} != {grid_shape.shape=}"

    wh, ww, wd = (int(v) for v in sliding_window_size)
    ph, pw, pd = (int(v) for v in patch_size)
    pH, pW, pD = wh // ph, ww // pw, wd // pd

    per_window_voxel_valid = []
    for b in range(shape_final_tensor.shape[0]):
        spatial_dims = shape_final_tensor[b]
        target_size = torch.zeros(3, dtype=torch.int64, device=shape_final_tensor.device)
        crop_size = torch.zeros(3, dtype=torch.int64, device=shape_final_tensor.device)
        pad_left = torch.zeros(3, dtype=torch.int64, device=shape_final_tensor.device)

        for axis, patch in enumerate((wh, ww, wd)):
            size = int(spatial_dims[axis].item())
            multiple = size // patch
            remainder = size - (multiple * patch)

            if min_area_for_padding <= 0.0:
                target = multiple * patch if multiple > 0 else size
            else:
                if multiple == 0:
                    target = patch
                else:
                    target = multiple * patch
                    if remainder > (min_area_for_padding * patch):
                        target = (multiple + 1) * patch

            crop = min(size, target)
            pad_total = max(target - crop, 0)
            target_size[axis] = target
            crop_size[axis] = crop
            pad_left[axis] = pad_total // 2

        hp = int(target_size[0].item()) // wh
        wp = int(target_size[1].item()) // ww
        dp = int(target_size[2].item()) // wd
        assert (
            hp,
            wp,
            dp,
        ) == tuple(
            int(v) for v in grid_shape[b].tolist()
        ), f"{(hp, wp, dp)=} != {tuple(grid_shape[b].tolist())=}"

        valid = torch.zeros(tuple(int(v) for v in target_size.tolist()), dtype=torch.float32)
        h0, w0, d0 = (int(v) for v in pad_left.tolist())
        hc, wc, dc = (int(v) for v in crop_size.tolist())
        valid[h0 : h0 + hc, w0 : w0 + wc, d0 : d0 + dc] = 1.0

        # (Hp, wh, Wp, ww, Dp, wd) -> (Hp, Wp, Dp, wh, ww, wd) -> (N, wh, ww, wd)
        valid_windows = (
            valid.view(hp, wh, wp, ww, dp, wd)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(hp * wp * dp, wh, ww, wd)
        )
        per_window_voxel_valid.append(valid_windows)

    per_window_voxel_valid = torch.cat(per_window_voxel_valid, dim=0)
    assert per_window_voxel_valid.shape[1:] == (wh, ww, wd), f"{per_window_voxel_valid.shape=}"

    # (N, pH, ph, pW, pw, pD, pd) -> (N, pH, pW, pD, ph, pw, pd)
    patch_blocks = per_window_voxel_valid.view(
        per_window_voxel_valid.shape[0], pH, ph, pW, pw, pD, pd
    ).permute(0, 1, 3, 5, 2, 4, 6)
    patch_abs_sum = patch_blocks.abs().sum(dim=(4, 5, 6))
    per_window_patch_mask = patch_abs_sum > 1e-6
    assert per_window_patch_mask.shape[1:] == (pH, pW, pD), f"{per_window_patch_mask.shape=}"
    return per_window_patch_mask
