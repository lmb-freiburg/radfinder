"""
Heavily modified version of the collate function.

Example image metadata that comes out of .nii.gz files:

sizeof_hdr (()): 348
extents (()): 0
session_error (()): 0
dim_info (()): 0
dim ((8,)): [  3 512 512 107   1   1   1   1]
intent_p1 (()): 0.0
intent_p2 (()): 0.0
intent_p3 (()): 0.0
intent_code (()): 0
datatype (()): 4
bitpix (()): 16
slice_start (()): 0
pixdim ((8,)): [-1.         0.7871094  0.7871094  5.         0.         0.
  0.         0.       ]
vox_offset (()): 0.0
scl_slope (()): nan
scl_inter (()): nan
slice_end (()): 0
slice_code (()): 0
xyzt_units (()): 10
cal_max (()): 0.0
cal_min (()): 0.0
slice_duration (()): 0.0
toffset (()): 0.0
glmax (()): 0
glmin (()): 0
qform_code (()): 1
sform_code (()): 1
quatern_b (()): 0.0
quatern_c (()): 1.0
quatern_d (()): 0.0
qoffset_x (()): 189.1064453125
qoffset_y (()): -59.1064453125
qoffset_z (()): -1156.0
srow_x ((4,)): [ -0.7871094  -0.          0.        189.10645  ]
srow_y ((4,)): [  0.          0.7871094   0.        -59.106445 ]
srow_z ((4,)): [    0.     0.     5. -1156.]
affine (torch.Size([4, 4])): tensor([[ 7.5000e-01,  0.0000e+00,  0.0000e+00, -2.0411e+02],
        [ 0.0000e+00,  7.5000e-01,  0.0000e+00, -5.0106e+01],
        [ 0.0000e+00,  0.0000e+00,  3.0000e+00, -1.1290e+03],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
       dtype=torch.float64)
original_affine ((4, 4)): [[-7.87109375e-01 -0.00000000e+00  0.00000000e+00  1.89106445e+02]
 [ 0.00000000e+00  7.87109375e-01  0.00000000e+00 -5.91064453e+01]
 [ 0.00000000e+00  0.00000000e+00  5.00000000e+00 -1.15600000e+03]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
as_closest_canonical (<class 'bool'>): False
spatial_shape ((3, 80)): [[128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128
  128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128
  128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128
  128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128
  128 128 128 128 128 128 128 128]
 [128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128
  128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128
  128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128
  128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128
  128 128 128 128 128 128 128 128]
 [ 32  32  32  32  32  32  32  32  32  32  32  32  32  32  32  32  32  32
   32  32  32  32  32  32  32  32  32  32  32  32  32  32  32  32  32  32
   32  32  32  32  32  32  32  32  32  32  32  32  32  32  32  32  32  32
   32  32  32  32  32  32  32  32  32  32  32  32  32  32  32  32  32  32
   32  32  32  32  32  32  32  32]]
space (<enum 'SpaceKeys'>): RAS
original_channel_dim (<class 'float'>): nan
filename_or_obj (<class 'str'>): /path/to/scan.nii.gz
location ((3, 80)): [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128
  128 128 128 128 256 256 256 256 256 256 256 256 256 256 256 256 256 256
  256 256 256 256 256 256 384 384 384 384 384 384 384 384 384 384 384 384
  384 384 384 384 384 384 384 384]
 [  0   0   0   0   0 128 128 128 128 128 256 256 256 256 256 384 384 384
  384 384   0   0   0   0   0 128 128 128 128 128 256 256 256 256 256 384
  384 384 384 384   0   0   0   0   0 128 128 128 128 128 256 256 256 256
  256 384 384 384 384 384   0   0   0   0   0 128 128 128 128 128 256 256
  256 256 256 384 384 384 384 384]
 [  0  32  64  96 128   0  32  64  96 128   0  32  64  96 128   0  32  64
   96 128   0  32  64  96 128   0  32  64  96 128   0  32  64  96 128   0
   32  64  96 128   0  32  64  96 128   0  32  64  96 128   0  32  64  96
  128   0  32  64  96 128   0  32  64  96 128   0  32  64  96 128   0  32
   64  96 128   0  32  64  96 128]]
count (<class 'int'>): 80
offset (<class 'tuple'>): (0, 0, 0)
"""

from copy import deepcopy
from functools import partial
from typing import Callable, List

import numpy as np
import torch
from monai.data import list_data_collate
from monai.data.meta_tensor import MetaTensor
from radfinder.transforms.crop_metadata import crop_affine_for_eval
from radfinder.transforms.largest_multiple_crop import get_per_window_patches_masks
from radfinder.utils.logging_utils import log_warning
from radfinder.utils.scan_utils import downscale_affine_to_windows, get_slice_from_scan_with_affines
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def extended_collate_dino(samples_list: List) -> dict:
    """
    Applies MONAI's list_data_collate first and then extends it with DINOv2 masking logic.

    Args:
        samples_list: List of samples containing 'global_crops' and 'local_crops'.
        mask_ratio: Tuple defining the range of masking ratios.
        mask_probability: Probability of applying masking.
        dtype: Data type to cast the collated tensors.
        n_tokens: Number of tokens for masking.
        mask_generator: Function to generate masks.

    Returns:
        A dictionary with collated global/local crops and corresponding masks.
    """
    # Apply MONAI's list_data_collate
    collated_data = list_data_collate(samples_list)

    # Extract crops
    global_views = torch.cat(collated_data["image_global_views"], dim=0)
    local_views = torch.cat(collated_data["image_local_views"], dim=0)

    return {
        "global_views": global_views,
        "local_views": local_views,
    }


def extended_collate_siglip(
    samples_list: List,
    tokenizer: PreTrainedTokenizerFast | None = None,
    tokenizer_padding: bool = True,
    tokenizer_truncation: bool = True,
    tokenizer_max_length: int | None = 1024,
    allow_none: bool = False,
    sliding_window_size: tuple = (128, 128, 64),
    min_area_for_padding: float = 0.0,
) -> dict:
    """
    Applies SigLIP collate and then extends it with tokenization logic.
    Handles variable-shaped frozen features by padding and adding masks.

    Args:
        samples_list: List of samples containing different inputs depending on settings.
        tokenizer: Tokenizer function to apply on the reports.

    Returns:
        A dictionary with collated images and tokenized text.
    """
    # remove nones or raise error
    if allow_none:
        samples_list_new = [s for s in samples_list if s is not None]
        if len(samples_list_new) < len(samples_list):
            log_warning(
                f"Removed None samples from batch. Original size: {len(samples_list)}, "
                f"New size: {len(samples_list_new)}"
            )
        if len(samples_list_new) == 0:
            log_warning("All samples are None. Returning None.")
            return None
        samples_list = samples_list_new
    else:
        for samp in samples_list:
            if samp is None:
                raise ValueError("None samples are not allowed when allow_none is False")
    image_meta_keys = "affine", "original_affine", "dim", "pixdim"

    # when mixing datasets we have to ensure all keys match, otherwise collate will fail
    union_keys = set()
    for samp in samples_list:
        union_keys.update(samp.keys())
    union_keys = sorted(union_keys)

    for nsamp, samp in enumerate(samples_list):
        for expected_key in union_keys:
            if expected_key in samp:
                continue
            all_missing_keys = sorted(set(union_keys) - set(samp.keys()))
            raise ValueError(
                f"Keys don't match. Expected: {', '.join(union_keys)}, Current sample {nsamp} is "
                f"missing {all_missing_keys}. Possible reasons: 1) A dataset provided some key "
                f"that should have been used up and removed by one of the transforms, but the "
                f"transform was not setup properly. 2) Datasets are mixed where the mixing is not "
                f"properly implemented or supported. Solution: Fix the dataset, fix the transform, "
                f"or remove the superfluous key here in case it is not needed later."
            )

    #################### collate vision input ####################
    coll = {}
    image_grid_shape = None
    if "image" in samples_list[0]:
        # get images and extract information from the metadata
        images: list[MetaTensor] = [s.pop("image") for s in samples_list]
        metas = [i.data.meta for i in images]  # type: ignore
        filenames = [m["filename_or_obj"] for m in metas]
        coll["filename"] = filenames
        coll["image_metadata"] = _repack_metas(metas, image_meta_keys)
        if "location" in metas[0]:
            # image window inputs after GridPatchd applied, each (N, C, H, W, D)
            # get grid shapes from metadata
            locs = [m["location"] for m in metas]
            # location and spatial_shape are (3, N_total) for each image and give position and size
            for m in metas:
                _shape = np.unique(m["spatial_shape"], axis=1).squeeze(1).tolist()
                assert _shape == list(sliding_window_size), f"{_shape=} != {sliding_window_size=}"
            # of each window in the image.
            image_grid_shape = torch.tensor(
                [get_grid_shape_from_loc(loc) for loc in locs], dtype=torch.int32
            )
            # build the mask for the windows
            Hp_max, Wp_max, Dp_max = image_grid_shape.max(dim=0).values
            window_mask = torch.zeros((len(images), Hp_max, Wp_max, Dp_max), dtype=torch.bool)
            for i, (Hp, Wp, Dp) in enumerate(image_grid_shape):
                window_mask[i, :Hp, :Wp, :Dp] = True
            # convert to torch Tensor since MetaTensor will cause problems otherwise
            images = tuple(i.as_tensor() for i in images)
            # we could pad and stack the images so we get one big tensor (B, Nmax, C, H, W, D)
            # however since the backbone will operate on window individually it's better to stack
            # the windows as (B*N, C, H, W, D)
            images = torch.cat(images, dim=0)
            # done
            coll["image"] = images
        else:
            # raw images after Spacingd but before any crops or grids.
            # used for FeatMode.RETURN_SPACED_IMAGE (variable shape, B==1) and for
            # single-volume models (fixed shape, B>=1).
            window_mask = None
            image_grid_shape = None
            if len(images) == 1:
                # write it back to the samples_list because we popped it above.
                # let monai auto-collate via list_data_collate at the end.
                samples_list[0]["image"] = images[0]
            else:
                shapes = {tuple(int(s) for s in img.shape) for img in images}
                assert len(shapes) == 1, (
                    f"Cannot stack images of different shapes: {shapes}. "
                    "Single-volume mode requires the transform to produce a fixed shape "
                )
                coll["image"] = torch.stack([img.as_tensor() for img in images], dim=0)

    elif "image_backbone_cls" in samples_list[0]:
        # pooled output from window-level transformer (B, Hp, Wp, Dp, emb_dim)
        image_backbone_cls, window_mask, image_grid_shape = pad_and_stack(
            [s.pop("image_backbone_cls") for s in samples_list]
        )
        image_backbone_patch_average, mask2, grid2 = pad_and_stack(
            [s.pop("image_backbone_patch_average") for s in samples_list]
        )
        assert eq(window_mask, mask2), f"{window_mask=} != {mask2=}"
        assert eq(image_grid_shape, grid2), f"{image_grid_shape=} != {grid2=}"
        coll["image_backbone_cls"] = image_backbone_cls
        coll["image_backbone_patch_average"] = image_backbone_patch_average
        if "image_backbone_patch_axis2" in samples_list[0]:
            # shapes (B, Hp, Wp, Dp, p0, E)
            # pad and stack will think p0 is a variable axis, but as long as it is not,
            # nothing bad will happen (bad=things are zero padded and we discard the masks here)
            image_backbone_patch_axis2, _mask2, _grid2 = pad_and_stack(
                [s.pop("image_backbone_patch_axis2") for s in samples_list]
            )
            coll["image_backbone_patch_axis2"] = image_backbone_patch_axis2
    elif "image_feature_comb_cls" in samples_list[0]:
        # output from feature combiner (B, emb_dim) CLS, (B, Hp, Wp, Dp, emb_dim) patches
        image_feature_comb_cls = torch.stack(
            [s.pop("image_feature_comb_cls") for s in samples_list]
        )  # (B, embed_dim)
        image_feature_comb_patch, window_mask, image_grid_shape = pad_and_stack(
            [s.pop("image_feature_comb_patch") for s in samples_list]
        )
        coll["image_feature_comb_cls"] = image_feature_comb_cls
        coll["image_feature_comb_patch"] = image_feature_comb_patch
    else:
        # image loading disabled completely
        window_mask = None
        image_grid_shape = None

    coll["window_mask"] = window_mask
    coll["image_grid_shape"] = image_grid_shape

    image_patches_mask = None
    if "image_metadata_uncropped" in samples_list[0]:
        metas = [s.pop("image_metadata_uncropped") for s in samples_list]
        if "image_metadata" in coll:
            # original images were loaded, the monai transform transformed the affines too,
            # so the metadata is already correct, keep for reference
            coll["image_metadata_uncropped"] = metas
            if min_area_for_padding > 0.0:
                # store the shapes of the images before the padding and cropping
                patch_size = 16, 16, 4  # other patch sizes not implemented!
                per_window_patches_mask = get_per_window_patches_masks(
                    torch.from_numpy(np.stack([m["spatial_shape_final"] for m in metas], axis=0)),
                    image_grid_shape,
                    patch_size,
                    sliding_window_size,
                    min_area_for_padding=min_area_for_padding,
                )
                image_patches_mask = per_window_patches_mask
        else:
            # windows are from scan cropped with LargestMultipleCenterCropd.
            # metadata is from the raw nifty. so we have to fix the affine.
            # apply the get_image_transform_eval_crop transform to the metadata
            new_metas = []
            for m in metas:
                new_affine = crop_affine_for_eval(
                    m["affine"],
                    m["spatial_shape_final"],  # after Spacingd, before LargestMultipleCenterCropd
                    sliding_window_size,
                    min_area_for_padding=min_area_for_padding,
                )
                new_m = deepcopy(m)
                new_m["uncropped_affine"] = deepcopy(m["affine"])
                new_m["affine"] = new_affine
                new_metas.append(new_m)
            coll["image_metadata"] = _repack_metas(new_metas, image_meta_keys)
    coll["image_patches_mask"] = image_patches_mask

    p_axis2 = None
    if "image_metadata" in coll and image_grid_shape is not None:
        # image_grid_shape=None => image wasn't cropped and we can't & don't need downscaled affines
        # Might be better to get p_axis2 etc from sliding_window_size and patch_embed_size?
        if "image_backbone_patch_axis2" in coll:
            p_axis2 = int(coll["image_backbone_patch_axis2"].shape[-2])
        wh, ww, wd = sliding_window_size
        for m in coll["image_metadata"]:
            # downscale the affine to the window size (from voxel affine to feature window affine)
            m["window_affine"] = downscale_affine_to_windows(m["affine"], wh, ww, wd)
            if p_axis2 is not None:
                assert wd % p_axis2 == 0, f"{wd=} not divisible by {p_axis2=}"
                m["axis2_affine"] = downscale_affine_to_windows(m["affine"], wh, ww, wd // p_axis2)

    if "feature_crop_box" in samples_list[0]:
        feature_crop_box = [s.pop("feature_crop_box") for s in samples_list]
        # if we need affines, we would need to run a function similar to crop_affine_for_eval
        # for now just delete all the affines to avoid accidental use of the wrong affines
        for m in coll["image_metadata"]:
            m.pop("affine", None)
            m.pop("original_affine", None)
            m.pop("window_affine", None)
            m.pop("axis2_affine", None)

    #################### collate text input ####################
    tokenizer_fn = partial(
        tokenizer,
        add_special_tokens=True,
        padding=tokenizer_padding,
        truncation=tokenizer_truncation,
        max_length=tokenizer_max_length,
        return_tensors="pt",
    )
    if "report_hidden_state" in samples_list[0]:
        # collate report hidden states
        report_hidden_state, report_mask, report_shape = pad_and_stack(
            [s.pop("report_hidden_state") for s in samples_list]
        )
        coll["report_hidden_state"] = report_hidden_state
        coll["report_hidden_state_mask"] = report_mask

    elif "report" in samples_list[0]:
        # tokenize raw text
        tokenizer_output = tokenizer_fn([s["report"] for s in samples_list])
        coll["report_input_ids"] = tokenizer_output.input_ids
        coll["report_hidden_state_mask"] = tokenizer_output.attention_mask
    # do not do both together, because the mask can be different due to different max lengths.

    #################### scan-slice alignment ####################
    if "slices" in samples_list[0]:
        slices_batched = [s.pop("slices", {}) for s in samples_list]
        if image_grid_shape is not None:
            # only if we actually cropped the image, otherwise we can't align the slices
            coll.update(
                collate_slices(
                    slices_batched,
                    tokenizer_fn,
                    coll["image_metadata"],
                    coll["image_grid_shape"],
                    p_axis2,
                )
            )

    #################### collate scan_key (string, not tensorizable) ####################
    if "scan_key" in samples_list[0]:
        coll["scan_key"] = [s.pop("scan_key") for s in samples_list]

    #################### collate remaining tensors ####################
    # print(f"Collating remaining tensors with keys: {list(samples_list[0].keys())}")
    coll_other: dict = list_data_collate(samples_list)
    for k in coll_other.keys():
        assert k not in coll, f"Key {k} already in manually collated data"
    return {**coll, **coll_other}


def _repack_metas(metas, image_meta_keys):
    imd = []
    for m in metas:
        imdhere = {}
        for k in image_meta_keys:
            if k in m:
                imdhere[k] = m[k]
        imd.append(imdhere)
    return imd


def get_masks(arange3d, scanaffine, dcmaffine, shape, max_h, max_w, max_d):
    coords = get_slice_from_scan_with_affines(
        arange3d,
        scanaffine,
        dcmaffine,
        shape,
        cval=-1,
        order=0,
        mode="grid-constant",
    )
    # Direct boolean assignment instead of np.unique+sort
    valid = coords[coords >= 0].astype(np.int64)
    mask = np.zeros((max_h, max_w, max_d), dtype=np.bool_)
    if valid.size > 0:
        mask_idx = np.unravel_index(valid, arange3d.shape)
        mask[mask_idx] = True
    return mask


def collate_slices(
    slices_batched: list[dict],
    tokenizer_fn: Callable,
    image_metadata: list[dict],
    image_grid_shape: torch.Tensor,
    p_axis2: int | None = None,
) -> dict:

    n_slices_total = sum(len(slices) for slices in slices_batched)
    if n_slices_total == 0:
        return {}
    coll_new = {}
    coll_new["slices"] = slices_batched
    # tokenize the snippets
    slice_batch_idx = []
    slice_texts = []
    for batch_idx, slices in enumerate(slices_batched):
        for _slicersopid, slicedata in slices.items():
            slice_batch_idx.append(batch_idx)
            slice_texts.append(slicedata["snippet"])
    slice_batch_idx = torch.tensor(slice_batch_idx, dtype=torch.long)
    coll_new["slice_batch_idx"] = slice_batch_idx
    tokenizer_output = tokenizer_fn(slice_texts)
    coll_new["snippet_input_ids"] = tokenizer_output.input_ids
    coll_new["snippet_attention_mask"] = tokenizer_output.attention_mask

    # build the masks for the slices, do it here since scipy map_coordinates runs on CPU only
    per_axis = p_axis2 is not None
    Hpmax, Wpmax, Dpmax = image_grid_shape.max(dim=0).values.tolist()
    D2max = Dpmax * p_axis2 if per_axis else None
    slice_window_mask, slice_batchi = [], []
    slice_axis2_mask_list = [] if per_axis else None
    for batch_idx, slices in enumerate(slices_batched):
        hp, wp, dp = image_grid_shape[batch_idx].tolist()
        window_number = np.arange(hp * wp * dp).reshape((hp, wp, dp))
        winaffine = image_metadata[batch_idx]["window_affine"]
        if per_axis:
            axis2_number = np.arange(hp * wp * dp * p_axis2).reshape((hp, wp, dp * p_axis2))
            axis2_affine = image_metadata[batch_idx]["axis2_affine"]
        for _slicersopid, slicemetadata in slices.items():
            # for now do everything in numpy on cpu, should be fast enough
            shape = slicemetadata["shape"]
            dcmaffine = np.array(slicemetadata["dcmaffine"])
            mask = get_masks(
                window_number,
                winaffine,
                dcmaffine,
                shape,
                Hpmax,
                Wpmax,
                Dpmax,
            )
            slice_window_mask.append(mask)
            slice_batchi.append(batch_idx)
            if per_axis:
                mask2 = get_masks(
                    axis2_number,
                    axis2_affine,
                    dcmaffine,
                    shape,
                    Hpmax,
                    Wpmax,
                    D2max,
                )
                slice_axis2_mask_list.append(mask2)
    slice_window_mask = np.stack(slice_window_mask)
    slice_window_mask = torch.from_numpy(slice_window_mask).bool()  # (S, Hpmax, Wpmax, Dpmax)
    slice_batchi = torch.from_numpy(np.stack(slice_batchi)).long()
    assert eq(slice_batchi, slice_batch_idx), f"{slice_batchi=} != {slice_batch_idx=}"
    coll_new["slice_window_mask"] = slice_window_mask
    if per_axis:
        slice_axis2_mask = torch.from_numpy(np.stack(slice_axis2_mask_list)).bool()
        coll_new["slice_axis2_mask"] = slice_axis2_mask

    # ---------- at this point filter slices that are invalid ----------

    # some of the slice masks are sum 0 so fully masked, if the slice was not found
    # this can happen e.g. if the crop cropped away the slice
    slice_patch_count = slice_window_mask.sum(dim=(1, 2, 3))
    slice_target_depth_mask = None
    valid_slices = slice_patch_count > 0.5
    if per_axis:
        # require axis2 mask to have at least one True position
        axis2_patch_count = slice_axis2_mask.sum(dim=(1, 2, 3))
        valid_slices = valid_slices & (axis2_patch_count > 0.5)

        # Fixed slice orientation: replace affine mask with single-depth full-spatial mask
        # most of the affines should look like this anyway and it simplifies everything
        # Per-depth counts: sum over Hp, Wp -> (S, D2)
        loc_slice_axis_mask = slice_axis2_mask[valid_slices]  # (S, Hp, Wp, D2)
        S2, sHpmax, sWpmax, sD2max = loc_slice_axis_mask.shape
        assert sHpmax == Hpmax, f"{sHpmax=} != {Hpmax=}"
        assert sWpmax == Wpmax, f"{sWpmax=} != {Wpmax=}"
        assert sD2max == D2max, f"{sD2max=} != {D2max=}"

        if S2 == 0:
            # All slices fell outside the valid window grid (can happen)
            slice_target_depth_mask = None
        else:
            depth_counts = loc_slice_axis_mask.sum(dim=(1, 2))
            best_depth = depth_counts.argmax(dim=-1)  # (S,)
            # target mask for the single best depth per snippet
            slice_target_depth_mask = torch.zeros(S2, D2max, dtype=torch.bool)
            slice_target_depth_mask[torch.arange(S2), best_depth] = True
    coll_new["slice_target_depth_mask"] = slice_target_depth_mask
    coll_new["valid_slices"] = valid_slices
    return coll_new


def eq(a, b):
    is_equals = a == b
    if isinstance(is_equals, bool):
        return is_equals
    if isinstance(is_equals, torch.Tensor):
        return is_equals.all()
    raise ValueError(f"Cannot determine equality between {type(a)=} {type(b)=} {type(is_equals)=}")


def pad_and_stack(tensors) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(tensors) == 0:
        raise ValueError("No tensors to pad and stack")
        # return torch.tensor([]), torch.tensor([], dtype=torch.bool)
    device = tensors[0].device
    dtype = tensors[0].dtype
    shapes = [t.shape for t in tensors]
    embed_dim = shapes[0][-1]
    assert all(s[-1] == embed_dim for s in shapes), "All tensors must have same emb dim"
    shape_tensor = torch.tensor(shapes, device=device, dtype=torch.int32)[:, :-1]
    max_shape = torch.max(shape_tensor, dim=0).values
    if all(s == shapes[0] for s in shapes):
        full_mask = torch.ones((len(tensors), *max_shape.tolist()), device=device, dtype=torch.bool)
        return torch.stack(tensors), full_mask, shape_tensor

    batch_size = len(tensors)
    out = torch.zeros((batch_size, *max_shape.tolist(), embed_dim), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, *max_shape.tolist()), device=device, dtype=torch.bool)
    for i, t in enumerate(tensors):
        slices = (i,) + tuple(slice(0, s) for s in t.shape[:-1])
        out[slices] = t
        mask[slices] = True
    return out, mask, shape_tensor


def get_grid_shape_from_loc(loc):
    """
    Instead of changing the monai transforms to get the grid shape, reverse engineer
    the results of the tranform to get the grid shape.
    """
    Hp, Wp, Dp = tuple(int(np.unique(loc[i, :]).size) for i in range(3))
    return Hp, Wp, Dp
