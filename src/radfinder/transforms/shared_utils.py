"""
Shared utilities for transforms
"""

import torch
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    GridPatchd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.transforms.croppad.dictionary import SpatialPadd
from radfinder.loader_utils import NibabelReader  # type: ignore
from radfinder.models.load_model import FeatMode
from radfinder.transforms.largest_multiple_crop import LargestMultipleCenterCropd

from packg.constclass import Const


class Language(Const):
    EN = "en"
    DE = "de"
    BOTH = "both"


class TransformKindC(Const):
    SPECTRE = "spectre"  # default windowed pipeline used for SPECTRE backbones


class TextTransformMode(Const):
    DEFAULT = "default"  # spectre authors: english findings + impression
    NONE = "none"  # keep text as is


class LoadTextMode(Const):
    REPORTS = "reports"
    NONE = "none"


def get_image_device_dtype(dtype: str, image_feat_mode: str, use_gds: bool = False):
    # handle device and dtype for image transforms
    assert dtype in ["float16", "float32"], "dtype must be either 'float16' or 'float32'"
    device = "cuda" if (use_gds and torch.cuda.is_available()) else "cpu"
    if dtype == "float16" and device == "cpu":
        # try to skip conversion if possible
        if image_feat_mode != FeatMode.FULL:
            # features are float16 cpu, so skip ensuring dtype and device
            device, dtype = None, None
        else:
            # loading raw nii gz keep ensuring the dtype and device
            pass
    else:
        if image_feat_mode not in {FeatMode.FULL, FeatMode.RETURN_SPACED_IMAGE}:
            raise ValueError(
                f"Features are hardcoded as float16 cpu, but requested to load as "
                f"{image_feat_mode=} {dtype=} {device=}. Either change the request or implement "
                f"EnsureTyped for the features."
            )
        else:
            # for full and return_spaced_image the transform is implemented and will run.
            pass
    return device, dtype


def get_image_transform_load_and_space(pixdim: tuple[float, float, float], reader=None):
    """Load image from raw nii gz file, normalize scale, orientation, and spacing"""
    if reader is None:
        reader = NibabelReader()
    image_transform_load_and_space = [
        LoadImaged(keys=("image",), reader=reader),
        EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
        ScaleIntensityRanged(
            keys=("image",), a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        # Orientationd: labels None means to autodetect LPS vs RAS tensors. But I think ours are
        # all RAS anyway. labels=(("L", "R"), ("P", "A"), ("I", "S")) means to force RAS tensors.
        # there should be nothing wrong with leaving the autodetect on.
        Orientationd(keys=("image",), axcodes="RAS", labels=None),
        Spacingd(keys=("image",), pixdim=pixdim, mode=("bilinear",)),
    ]
    return image_transform_load_and_space


def get_image_transform_crop_and_aug(
    img_size: tuple[int, int, int],  # (384, 384, 256)
    sliding_window_size: tuple[int, int, int],  # (128, 128, 64)
    max_shift: tuple[int, int, int],  # (64, 64, 32)
    random_flip: float,
    dtype: str | None,
    device: str | None,
):
    """Training augmentations on the image: random crop + random flips"""
    base_crop_size = tuple(img_size[i] + 2 * max_shift[i] for i in range(3))
    image_transform_crop_and_aug = [
        # do pad OR centercrop to base_crop_size
        ResizeWithPadOrCropd(keys=("image",), spatial_size=base_crop_size)
    ]
    if dtype is not None and device is not None:
        # only convert to dtype and device if necessary
        image_transform_crop_and_aug += [
            EnsureTyped(keys=("image",), dtype=getattr(torch, dtype), device=device)
        ]
    image_transform_crop_and_aug += [
        RandSpatialCropd(
            keys=("image",),
            roi_size=img_size,
            random_size=False,
        ),
        RandFlipd(keys=("image",), spatial_axis=0, prob=random_flip),
        RandFlipd(keys=("image",), spatial_axis=1, prob=random_flip),
        RandFlipd(keys=("image",), spatial_axis=2, prob=random_flip),
        GridPatchd(
            keys=("image",),
            patch_size=sliding_window_size,
            overlap=0.0,
        ),
    ]
    return image_transform_crop_and_aug


def get_image_transform_eval_crop(
    sliding_window_size: tuple[int, int, int],
    dtype: str | None,
    device: str | None,
    min_area_for_padding: float = 0.0,
):
    """Evaluation: crop the biggest possible multiple of window_size"""
    image_transform_eval_crop = []
    if dtype is not None and device is not None:
        # only convert to dtype and device if necessary
        image_transform_eval_crop += [
            EnsureTyped(keys=("image",), dtype=getattr(torch, dtype), device=device)
        ]
    image_transform_eval_crop += [
        LargestMultipleCenterCropd(
            keys=("image",),
            patch_size=sliding_window_size,
            min_area_for_padding=min_area_for_padding,
        ),
        # pad things that are smaller than one window
        SpatialPadd(keys=("image",), spatial_size=sliding_window_size, mode="constant", value=0.0),
        # cut into grids
        GridPatchd(
            keys=("image",),
            patch_size=sliding_window_size,
            overlap=0.0,
        ),
    ]
    return image_transform_eval_crop


def get_image_transform_eval_crop_volume(
    sliding_window_size: tuple[int, int, int],
    dtype: str | None,
    device: str | None,
    min_area_for_padding: float = 0.0,
):
    """Evaluation crop/pad without GridPatchd, keeping the cropped 3D volume intact."""
    return get_image_transform_eval_crop(
        sliding_window_size=sliding_window_size,
        dtype=dtype,
        device=device,
        min_area_for_padding=min_area_for_padding,
    )[:-1]
