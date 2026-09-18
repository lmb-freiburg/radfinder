"""
Extract and save images and embeddings. See README

If you run CPU OOM when dataloading try:

--workers 1
--prefetch_factor 1

or

--workers 0
"""

import gc
import resource
import time
from pathlib import Path
from timeit import default_timer

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from attr import define
from attrs import define
from loguru import logger
from radfinder.data.dataloader_args import RetrievalDatasetArgs, retrieval_dataset_args_to_dict
from radfinder.data.dataloader_retrieval import get_retrieval_dataloader
from radfinder.data.dataloader_train import get_dataset
from radfinder.models.load_model import DEFAULT_MODEL_CONFIG_FILE, FeatMode, create_siglip
from radfinder.paths import get_medv_data_dir
from radfinder.save_embeddings_lib import forward_pass, save_images
from radfinder.transforms.load_features import get_features_subdir
from radfinder.transforms.shared_utils import LoadTextMode
from radfinder.utils.config import load_config_without_types, random_seed
from torch import nn

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from packg.misc import format_exception
from packg.tqdmext import tqdm_max_ncols
from typedparser import (
    TaskSplitterArgs,
    TypedParser,
    VerboseQuietArgs,
    add_argument,
    split_list_given_task_splitter_args,
)
from visiontext.profiling.code_profiler import (
    start_pyinstrument_profiler,
    stop_pyinstrument_profiler,
)


@define
class Args(VerboseQuietArgs, TaskSplitterArgs, RetrievalDatasetArgs):
    batch_size: int = add_argument(type=int, default=1, help="Batch size for the dataloader")
    workers: int = add_argument(type=int, default=1, help="Number of workers for the dataloader")
    prefetch_factor: int = add_argument(
        type=int, default=2, help="Prefetch factor for the dataloader"
    )
    do_image_backbone: bool = add_argument(action="store_true")
    do_text_backbone: bool = add_argument(action="store_true")
    save_backbone_patches: bool = add_argument(
        action="store_true", help="Default off, too big, bigger than the input 3d images"
    )
    save_sliced_backbone_patches: bool = add_argument(
        action="store_true",
        help="Save backbone patches sliced by axis (avg over other axes) required for localization",
    )
    split: str = add_argument(type=str, default="valid", help="Dataset subset to process")
    max_memory: int | None = add_argument(type=int, default=None, help="Max memory limit in GB")
    do_profile: bool = add_argument(action="store_true")
    check: bool = add_argument(action="store_true")
    save_spaced_images: bool = add_argument(action="store_true")
    image_feat_mode: str = add_argument(default=FeatMode.FULL, help="Image feature mode")
    model_cfg: Path = add_argument(default=DEFAULT_MODEL_CONFIG_FILE)
    dataset_name: str = add_argument(default="ctrate", help="Dataset name")
    show_indices: int = add_argument(
        type=int,
        default=0,
        help="If > 0, show this many indices and their iloc",
    )
    ckpt_file: str | None = add_argument(default=None, help="Checkpoint file to load weights from")
    print_transform: bool = add_argument(action="store_true")
    max_datapoints: int | None = add_argument(
        type=int, default=None, help="Maximum number of datapoints to use"
    )
    cpu: bool = add_argument(action="store_true", help="Use CPU even if GPU is available")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    assert args.batch_size == 1, "Batch size must be 1 for save_embeddings"
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print_ = logger.info
    print_(f"Using device: {device}")

    #################### figure out image feature mode for model and dataset ####################
    image_feat_mode_model = args.image_feat_mode
    if not args.do_image_backbone:
        # no image backbone requested, disable image tower
        image_feat_mode_model = FeatMode.NONE
    image_feat_mode_dataset = FeatMode.NONE
    if args.save_spaced_images:
        # no image backbone, just save the images after spacing
        assert not args.do_image_backbone and not args.do_text_backbone, (
            f"{args.save_spaced_images=} means no forward pass, so model will not run, but got "
            f"{args.do_image_backbone=} and {args.do_text_backbone=}"
        )
        image_feat_mode_dataset = FeatMode.RETURN_SPACED_IMAGE
    elif args.do_image_backbone:
        # run image backbone as given by the args
        assert args.image_feat_mode in {FeatMode.FULL, FeatMode.FROM_SPACED_IMAGE}, (
            f"Requested to run image backbone and save features {args.image_feat_mode=}, "
            f"but trying to run model using feature mode {args.image_feat_mode=}, "
            f"only 'full' and 'from_spaced_image' supported"
        )
        image_feat_mode_dataset = args.image_feat_mode
    else:
        image_feat_mode_dataset = FeatMode.NONE
    text_feat_mode = FeatMode.FULL if args.do_text_backbone else FeatMode.NONE
    load_text = LoadTextMode.NONE if text_feat_mode == FeatMode.NONE else LoadTextMode.REPORTS

    #################### load model config and create model ####################
    model_config_file = args.model_cfg
    model_config = load_config_without_types(model_config_file)
    model = None
    if not args.save_spaced_images:
        model = create_siglip(
            model_config,
            image_feat_mode=image_feat_mode_model,
            text_feat_mode=text_feat_mode,
        )
        if args.ckpt_file is not None:
            model.load_checkpoint(args.ckpt_file)

    # Define transformations for the dataset
    do_image_backbone = args.do_image_backbone
    do_text_backbone = args.do_text_backbone
    do_image_projection = False
    do_text_projection = False

    # Determine which output files are expected based on settings
    expected_files = []
    if args.save_spaced_images:
        expected_files.append("image_spaced.safetensors.zst")
        expected_files.append("image_spaced.json")
    if do_image_backbone:
        expected_files.append("image_backbone_cls.safetensors.zst")
        if args.save_backbone_patches:
            expected_files.append("image_backbone_patch.safetensors.zst")
        if args.save_sliced_backbone_patches:
            expected_files.append("image_backbone_patch_axis0.safetensors.zst")
            expected_files.append("image_backbone_patch_axis1.safetensors.zst")
            expected_files.append("image_backbone_patch_axis2.safetensors.zst")
            # expected_files.append("image_backbone_patch_axis01.safetensors.zst")  # WARNING inconsistencies now in the data.
        expected_files.append("image_feature_comb_cls.safetensors.zst")
        expected_files.append("image_feature_comb_patch.safetensors.zst")
        expected_files.append("image_backbone_patch_average.safetensors.zst")
        if do_image_projection:
            expected_files.append("image_projection.safetensors.zst")
    if do_text_backbone:
        expected_files.append("text_backbone.safetensors.zst")
        if do_text_projection:
            expected_files.append("text_projection.safetensors.zst")
    print_(f"Expected files:\n    - {'\n    - '.join(expected_files)}")

    # Get full dataset with all indices
    full_dataset = get_dataset(
        args.dataset_name,
        split=args.split,
        transform=None,
        ctrate_filter_mode=args.ctrate_filter_mode,
        max_datapoints=args.max_datapoints,
        load_text=load_text,
        add_slices=False,
    )
    paths = [item["image"] for item in full_dataset.data]
    print_(f"Found {len(paths)} paths for dataset {args.dataset_name} and split {args.split}")

    # Create save directory
    images_subdir, features_subdir = get_features_subdir(args.dataset_name, model_config_file.stem)
    save_subdir = images_subdir if args.save_spaced_images else features_subdir
    save_dir = get_medv_data_dir() / f"embeddings/{save_subdir}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Split into sublist given args
    paths = split_list_given_task_splitter_args(paths, args, print_fn=print)
    print_(f"Processing {len(paths)} paths after splitting with {args.start=} and {args.num=}")

    # Check which datapoints already have all expected files
    print_(f"Checking existing indices in {save_dir}")
    completed_keys, remaining_keys = [], []
    for path in tqdm_max_ncols(paths, desc="Checking existing embeddings"):
        datapoint_key = full_dataset.get_datapoint_key_from_scan_path(path)
        feature_subdir = full_dataset.get_feature_subdir_from_datapoint_key(datapoint_key)
        feature_dir = save_dir / feature_subdir
        all_exist = all((feature_dir / f).is_file() for f in expected_files)
        if all_exist:
            completed_keys.append(datapoint_key)
        else:
            remaining_keys.append(datapoint_key)
    print_(f"Still todo: {len(remaining_keys)}/{len(paths)} datapoints")

    if args.show_indices != 0:
        ni = args.show_indices
        if ni < 0:  # -1 = all
            ni = len(remaining_keys)
        elif ni > len(remaining_keys):
            ni = len(remaining_keys)
        print_(f"Remaining keys (max {ni} shown): {remaining_keys[:ni]}")

    if len(remaining_keys) == 0:
        print_("All embeddings already exist, nothing to do.")
        return
    if args.check:
        print_("Check mode enabled, exiting before processing.")
        return

    # limit CPU memory
    if args.max_memory is not None:
        LIMM = args.max_memory * 1024**3
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (LIMM, hard))
        resource.setrlimit(resource.RLIMIT_DATA, (LIMM, hard))

    # Loop through the dataset and save embeddings
    profiler = None  # get_gpu_profiler()
    if args.do_profile:
        start_pyinstrument_profiler()

    # build the dataset and dataloader with the subset only
    del full_dataset
    dataloader, dataset = get_retrieval_dataloader(
        model_config=model_config,
        model_config_name=model_config_file.stem,
        dataset_name=args.dataset_name,
        split=args.split,
        max_datapoints=None,
        data_fraction=1.0,
        key_subset=remaining_keys,
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        image_feat_mode=image_feat_mode_dataset,
        text_feat_mode=text_feat_mode,
        lazy=False,
        dataset_config=retrieval_dataset_args_to_dict(args),
        load_text=load_text,
        add_slices=False,
    )
    if args.print_transform:
        print_("Dataset transform:")
        print_(dataset.transform)

    # setup accelerator
    random_seed(42)
    dataloader_config = DataLoaderConfiguration(non_blocking=True)
    accelerator = Accelerator(
        dataloader_config=dataloader_config, mixed_precision="bf16", cpu=args.cpu
    )
    device = accelerator.device
    print_(f"Using device: {device}")
    if model is not None:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model, dataloader = accelerator.prepare(model, dataloader)
    else:
        dataloader = accelerator.prepare(dataloader)

    print_(
        f"Setup dataloader iter - if it hangs here (in sbatch), maybe it's job logs being buffered"
    )
    dl_iter = iter(dataloader)
    total_batches = len(dataloader)
    print_(f"Starting to process batches, total {total_batches} batches.")
    start_time = default_timer()
    for i in range(total_batches):
        elapsed = default_timer() - start_time
        avg_time_per_batch = elapsed / (i + 1)
        remaining_batches = total_batches - (i + 1)
        remaining_time = avg_time_per_batch * remaining_batches
        print_(
            f"Processing batch {i+1}/{total_batches} - Elapsed: {elapsed/3600:.1f}h, "
            f"Remaining: {remaining_time/3600:.1f}h"
        )
        try:
            batch = next(dl_iter)
        except ValueError as e:
            logger.error(f"Corrupt item: {i}\n{format_exception(e)}\n{dataloader.dataset.data[i]}")
            continue
        # except RuntimeError as e:
        #     print_(f"CPU OOM when loading datapoint {i}, skip it. Error: {format_exception(e)}")
        #     continue
        # except KeyboardInterrupt as e:
        #     if not args.do_profile:
        #         raise e
        #     else:
        #         print_("Keyboard interrupt received, stopping for profiling...")
        #         break
        if batch is None:
            print_(f"[DEBUG] Batch {i} is None, skipping")
            continue
        print_(f"[DEBUG] Processing batch {i} with keys: {batch.keys()}")

        # figure out the save paths for the batch
        keys, save_paths = [], []
        for f in batch["filename"]:
            key = dataset.get_datapoint_key_from_scan_path(f)
            keys.append(key)
            save_paths.append(save_dir / dataset.get_feature_subdir_from_datapoint_key(key))

        if args.save_spaced_images:
            # directly save spaced images without forward pass
            spaced_images = batch["image"]  # already resampled and oriented
            save_images(spaced_images, [p / "image_spaced" for p in save_paths])
            del batch
            gc.collect()
            time.sleep(0.1)
            continue
        try:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            with accelerator.autocast(), accelerator.no_sync(model), torch.inference_mode():
                forward_pass(
                    batch,
                    model,
                    do_image_backbone,
                    do_image_projection,
                    do_text_backbone,
                    do_text_projection,
                    expected_files,
                    args.save_backbone_patches,
                    args.save_sliced_backbone_patches,
                    save_paths,
                )
        except torch.OutOfMemoryError as e:
            print_(
                f"Out of memory error for batch with filenames {batch['filename']}: "
                f"{format_exception(e)} Try to recover..."
            )
        except KeyboardInterrupt as e:
            if not args.do_profile:
                raise e
            else:
                print_("Keyboard interrupt received, stopping for profiling...")
                break
        del batch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.1)
        if profiler is not None:
            print_(f"GPU Profiler output: {profiler.profile_to_str()}")

    if args.do_profile:
        stop_pyinstrument_profiler()


if __name__ == "__main__":
    main()
