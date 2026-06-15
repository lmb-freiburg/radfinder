from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data.meta_tensor import MetaTensor
from radfinder.models.modeling import last_token_pool
from radfinder.models.vision_language import SigLIP
from radfinder.utils.logging_utils import log_debug, log_info

from packg.iotools.jsonext import dump_json
from typedparser import TaskSplitterArgs
from visiontext.iotools.feature_compression import convert_to_fp16_torch, dump_single_safetensor_zst


def save_embeddings(embeddings, save_paths):
    if isinstance(embeddings, torch.Tensor):
        embeddings = torch.split(embeddings, 1, dim=0)
        embeddings = [emb.squeeze(0) for emb in embeddings if emb.numel() > 0]
    elif isinstance(embeddings, list):
        embeddings = [
            emb for emb in embeddings if isinstance(emb, torch.Tensor) and emb.numel() > 0
        ]
    else:
        raise ValueError("Embeddings must be a tensor or a list of tensors.")

    assert len(embeddings) > 0, "No valid embeddings to save."
    for emb, save_path in zip(embeddings, save_paths, strict=True):
        save_path = Path(save_path).with_suffix(".safetensors.zst")
        emb = convert_to_fp16_torch(emb)
        log_info(f"Saving {save_path} with shape {emb.shape} and dtype {emb.dtype}")
        dump_single_safetensor_zst(emb, save_path, create_parent=True, level=5)
        # np.save(save_path, emb.cpu().numpy())


def save_images(images: MetaTensor, save_paths):
    assert isinstance(images, MetaTensor)  # B, C=1, H, W, D
    assert images.shape[0] == 1, f"Batch size must be 1, got {images.shape[0]}"
    meta_dict = images.data.meta
    meta_dict["spatial_shape_final"] = [list(images.shape[2:])]
    save_metas(meta_dict, save_paths)
    images = torch.split(images, 1, dim=0)  # list of (1, C, H, W, D)
    images = [img.squeeze(0) for img in images]  # list of (C, H, W, D)

    assert len(images) > 0, "No valid images to save."
    assert len(images) == len(save_paths), "Number of images and save paths must match."

    for img, save_path in zip(images, save_paths):
        log_info(f"Saving {save_path} with shape {img.shape} and dtype {img.dtype}")
        save_path = Path(save_path).with_suffix(".safetensors.zst")
        img = convert_to_fp16_torch(img)
        dump_single_safetensor_zst(img, save_path, create_parent=True, level=5)


def save_metas(meta_dict, save_paths):
    batch_size = len(save_paths)
    # convert dict of key->batch_values to list of dicts of key->single_value
    meta_list = [{} for _ in range(batch_size)]
    for i, (key, value) in enumerate(meta_dict.items()):
        value = detensor_maybe(value)
        if key == "offset":
            # offset is 3-tuple of batched tensors so it needs special treatment
            for k in range(batch_size):
                meta_list[k][key] = []
            for j in range(3):
                value_here = value[j]
                value_here = detensor_maybe(value_here)
                assert (
                    len(value_here) == batch_size
                ), f"Offset length mismatch for axis {j}: {len(value_here)=} != {batch_size=}"
                for k, item in enumerate(value_here):
                    meta_list[k][key].append(item)
            continue
        if len(value) == batch_size:
            for k, item in enumerate(value):
                meta_list[k][key] = item
            continue
        raise ValueError(f"{len(value)=} != {batch_size=} for key {key}")

    assert len(meta_list) == len(
        save_paths
    ), "Number of metadata entries must match number of save paths."

    for meta, save_path in zip(meta_list, save_paths, strict=True):
        save_path = Path(save_path).with_suffix(".json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(
            meta, save_path, custom_format_nan_to_none=True, indent=2, sort_keys=True, verbose=False
        )


def detensor_maybe(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return value


def split_dataframe_given_task_splitter_args(
    df: pd.DataFrame, task_splitter_args: TaskSplitterArgs, print_fn=None
):
    start, num = task_splitter_args.start, task_splitter_args.num
    return split_dataframe_for_processing(df, start, num, print_fn=print_fn)


def split_dataframe_for_processing(
    df: pd.DataFrame,
    start: int = 0,
    num: int | None = None,
    print_fn=None,
):
    len_df = len(df)
    end = start + num if num is not None else None
    df_slice = df.iloc[start:end]
    if print_fn is not None and (start > 0 or num is not None):
        print_fn(
            "Split dataframe for processing, input length "
            f"{len_df}, starting at {start}, processing max {num} "
            f"reduced to {len(df_slice)}"
        )
    return df_slice


def forward_pass(
    batch,
    model: SigLIP,
    do_image_backbone,
    do_image_projection,
    do_text_backbone,
    do_text_projection,
    expected_files,
    save_backbone_patches,
    save_sliced_backbone_patches,
    save_paths,
):
    log_debug(f"forward_pass called for {len(save_paths)} items")
    log_debug(f"save_paths: {save_paths}")
    log_debug(f"expected_files: {expected_files}")
    image_backbone = model.backbone_image
    image_feature_comb = model.feature_comb_image
    image_projection = model.projection_image
    text_backbone = model.backbone_text
    text_projection = model.projection_text

    # Check if all expected files already exist for this batch
    all_files = []
    for p in save_paths:
        all_files.extend([p.joinpath(f) for f in expected_files])
    all_exist = all(p.exists() for p in all_files)
    if all_exist:
        log_debug(f"All files already exist, skipping batch: {all_files}")
        return
    log_debug("Not all files exist, proceeding with embedding generation")

    if do_image_backbone:
        B = len(batch["image_grid_shape"])
        assert B == 1, "Batch size must be 1 for save_embeddings"
        Hp, Wp, Dp = tuple(int(v) for v in batch["image_grid_shape"][0].tolist())
        images = batch["image"]
        BN, C, H, W, D = images.shape
        N = Hp * Wp * Dp
        assert BN == B * N, f"{B=} * {N=} != {BN=}"

        image_embeddings = image_backbone(images)
        ibb_cls = image_embeddings[:, 0, :].view(B, Hp, Wp, Dp, -1)
        ibb_patch_average = image_embeddings[:, 1:, :].mean(dim=1).view(B, Hp, Wp, Dp, -1)

        save_embeddings(
            ibb_cls,
            [p / "image_backbone_cls" for p in save_paths],
        )
        save_embeddings(
            ibb_patch_average,
            [p / "image_backbone_patch_average" for p in save_paths],
        )
        if save_backbone_patches or save_sliced_backbone_patches:
            patch_tokens = image_embeddings.shape[1] - 1
            window_size = image_backbone.sliding_window_size
            patch_size = image_backbone.patch_size
            patch_grid = tuple(int(w // p) for w, p in zip(window_size, patch_size, strict=True))
            assert all(
                w % p == 0 for w, p in zip(window_size, patch_size, strict=True)
            ), f"Non-divisible window/patch sizes: {window_size=} {patch_size=}"
            assert (
                patch_tokens == patch_grid[0] * patch_grid[1] * patch_grid[2]
            ), f"Unexpected patch grid {patch_tokens=} for {patch_grid=}"
            image_backbone_patches_to_save = image_embeddings[:, 1:].view(
                B, Hp, Wp, Dp, patch_grid[0], patch_grid[1], patch_grid[2], -1
            )  # e.g. 6, 6, 4, 8, 8, 8, 1080 for 6x6x4 windows, each 8x8x8 patches
            # the patches come from applying the 16x16x8 patch embedding to 128x128x64 voxels
            if save_backbone_patches:
                save_embeddings(
                    image_backbone_patches_to_save,
                    [p / "image_backbone_patch" for p in save_paths],
                )
            if save_sliced_backbone_patches:
                axis0 = image_backbone_patches_to_save.mean(dim=(5, 6))  # (B, Hp, Wp, Dp, p0, E)
                axis1 = image_backbone_patches_to_save.mean(dim=(4, 6))  # (B, Hp, Wp, Dp, p1, E)
                axis2 = image_backbone_patches_to_save.mean(dim=(4, 5))  # (B, Hp, Wp, Dp, p2, E)
                # axis01 = image_backbone_patches_to_save.mean(dim=6)  # (B, Hp, Wp, Dp, p0, p1, E)
                save_embeddings(axis0, [p / "image_backbone_patch_axis0" for p in save_paths])
                save_embeddings(axis1, [p / "image_backbone_patch_axis1" for p in save_paths])
                save_embeddings(axis2, [p / "image_backbone_patch_axis2" for p in save_paths])
                # save_embeddings(axis01, [p / "image_backbone_patch_axis01" for p in save_paths])

        # do_image_feature_comb:
        image_embeddings = torch.cat([ibb_cls, ibb_patch_average], dim=4).view(B, N, -1)
        image_embeddings = image_feature_comb(image_embeddings, grid_size=(Hp, Wp, Dp))
        save_embeddings(
            image_embeddings[:, 0, :],
            [p / "image_feature_comb_cls" for p in save_paths],
        )
        # Reshape patch tokens to preserve spatial structure (Hp, Wp, Dp)
        image_feature_comb_patches = image_embeddings[:, 1:, :].view(B, Hp, Wp, Dp, -1)
        save_embeddings(
            image_feature_comb_patches,
            [p / "image_feature_comb_patch" for p in save_paths],
        )

        if do_image_projection:
            image_embeddings = torch.cat(
                [
                    image_embeddings[:, 0, :],  # class token
                    image_embeddings[:, 1:, :].mean(dim=1),  # mean of patch tokens
                ],
                dim=1,
            )
            image_embeddings = image_projection(image_embeddings)
            save_embeddings(
                image_embeddings,
                [p / "image_projection" for p in save_paths],
            )

    if do_text_backbone:
        text_embeddings = text_backbone(
            input_ids=batch["report_input_ids"], attention_mask=batch["report_hidden_state_mask"]
        )
        text_embeddings = last_token_pool(
            text_embeddings.last_hidden_state, batch["report_hidden_state_mask"]
        )
        save_embeddings(text_embeddings, [p / "text_backbone" for p in save_paths])
        if do_text_projection:
            text_embeddings = text_projection(text_embeddings)
            save_embeddings(text_embeddings, [p / "text_projection" for p in save_paths])
