from functools import partial
from typing import Callable, List, Optional, Union

import monai.data as data
from radfinder.data.ct_rate import CTRateDataset, CTRateFilterMode
from radfinder.data.inspect import InspectDataset
from radfinder.data.merlin import MerlinDataset
from radfinder.data.rad_chestct import NpzReader, RadChestCTDataset
from radfinder.models.load_model import FeatMode
from radfinder.transforms.new_compose import ReprCompose, TimedCompose
from radfinder.transforms.shared_utils import LoadTextMode
from radfinder.transforms.train_transform import TextTransformMode, get_train_transform
from radfinder.utils.collate import extended_collate_siglip
from radfinder.utils.logging_utils import log_debug, log_info
from torch.utils.data import ConcatDataset
from transformers import Qwen2TokenizerFast

from packg.constclass import Const


class DatasetNameC(Const):
    CTRATE = "ctrate"
    MERLIN = "merlin"
    INSPECT = "inspect"
    RADCHESTCT = "radchestct"


def get_dataset(
    dataset_name: str,
    split: str = "val",
    transform: Optional[Callable] = None,
    # ways to subset the dataset
    max_datapoints: Optional[int] = None,
    data_fraction: float = 1.0,
    key_subset: list[str] | None = None,
    # dataset-specific settings
    load_text: str = LoadTextMode.REPORTS,
    ctrate_filter_mode: str = CTRateFilterMode.DUP_ALL,
    dataset_config: dict | None = None,
    add_slices: bool = True,
    sample_scan_per_dedup_report: bool = False,
):
    dataset_config = dict(dataset_config or {})
    ctrate_filter_mode = dataset_config.get("ctrate_filter_mode", ctrate_filter_mode)
    include_reports = load_text != LoadTextMode.NONE

    if dataset_name == DatasetNameC.CTRATE:
        dataset = CTRateDataset(
            include_reports=include_reports,
            filter_mode=ctrate_filter_mode,
            add_slices=add_slices,
            transform=transform,
            split=split,
            max_datapoints=max_datapoints,
            data_fraction=data_fraction,
            key_subset=key_subset,
            sample_scan_per_dedup_report=sample_scan_per_dedup_report,
        )
    elif dataset_name == DatasetNameC.MERLIN:
        dataset = MerlinDataset(
            include_reports=include_reports,
            add_slices=add_slices,
            transform=transform,
            split=split,
            max_datapoints=max_datapoints,
            data_fraction=data_fraction,
            key_subset=key_subset,
        )
    elif dataset_name == DatasetNameC.INSPECT:
        dataset = InspectDataset(
            include_reports=include_reports,
            add_slices=add_slices,
            transform=transform,
            split=split,
            max_datapoints=max_datapoints,
            data_fraction=data_fraction,
            key_subset=key_subset,
        )
    elif dataset_name == DatasetNameC.RADCHESTCT:
        dataset = RadChestCTDataset(
            include_reports=include_reports,
            transform=transform,
            split=split,
            max_datapoints=max_datapoints,
            data_fraction=data_fraction,
            key_subset=key_subset,
        )
    return dataset


def get_train_dataloader(
    datasets: Union[str, List[str], dict],
    image_feat_mode: str = FeatMode.FULL,
    text_feat_mode: str = FeatMode.FULL,
    pixdim: tuple = (0.75, 0.75, 1.5),
    img_size: tuple = (384, 384, 256),
    sliding_window_size: tuple = (128, 128, 64),
    min_area_for_padding: float = 0.0,
    features_model_name: str = None,
    text_transform_mode: str = TextTransformMode.DEFAULT,
    do_snippet_alignment: dict | None = None,
    model_settings: dict | None = None,
    language: str = "en",
    text_tokenizer: str = "Qwen/Qwen3-Embedding-0.6B",
    tokenizer_max_length: int = 4096,
    batch_size: int = 64,
    workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    shuffle: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = False,
    use_gds: bool = False,
    compose_class: type = ReprCompose,
    data_fraction: float = 1.0,
    dtype: str = "float16",
    max_num_icd10: int = 20,
    keep_original_prob: float = 0.5,
    drop_prob: float = 0.3,
    random_flip: float = 0.5,
    random_crop_features: bool | float = 0.0,
    random_flip_features: bool | float = 0.0,
    organ_text_replace_prob: float = 0.0,
    no_comp_replace_prob: float = 0.0,
    sample_scan_per_dedup_report: bool = False,
) -> data.DataLoader:
    """Create dataloader for training with multiple datasets.

    Args:
        datasets: Dataset specification. Supported formats:
            - str: Single dataset name.
            - list[str]: List of dataset names using default options.
            - dict[str, dict]: Mapping from dataset name to per-dataset override options.
        image_feat_mode: Image feature extraction mode.
        text_feat_mode: Text feature extraction mode.
        pixdim: Target voxel spacing (x, y, z) for image resampling.
        img_size: Output image size after padding or cropping.
        sliding_window_size: Sliding window patch size.
        features_model_name: Feature extractor model name.
        features_dir_overwrite: Optional path to override the default features directory.
        text_transform_mode: Text transformation mode.
        language: Language used for text processing.
        text_tokenizer: HuggingFace tokenizer name.
        tokenizer_max_length: Maximum number of tokens per text sample.
        batch_size: Batch size per iteration.
        workers: Number of DataLoader worker processes.
        prefetch_factor: Number of batches prefetched per worker.
        pin_memory: Enable pinned memory for faster GPU transfer.
        shuffle: Shuffle dataset at every epoch.
        drop_last: Drop the last incomplete batch.
        persistent_workers: Keep worker processes alive between epochs.
        use_gds: Enable GPUDirect Storage for data loading.
        compose_class: Transform composition class (default: ReprCompose).
        data_fraction: Fraction of the dataset to use (for subsampling).
        dtype: Tensor data type ("float16" or "float32").
        max_num_icd10: Maximum number of ICD-10 codes used in RandomReportTransformd.
        keep_original_prob: Probability of keeping original text in RandomReportTransformd.
        drop_prob: Probability of dropping elements in RandomReportTransformd.
        random_crop_features: Crop frozen features to training grid size with random jitter.
        random_flip_features: Flip frozen features with random jitter.
        sample_scan_per_dedup_report: Whether to sample one scan per deduplicated report.
    Returns:
        torch.utils.data.DataLoader: Configured training DataLoader.
    """
    # Normalize datasets to dict format
    if isinstance(datasets, str):
        datasets = {datasets: {}}
    elif isinstance(datasets, list):
        datasets = {ds: {} for ds in datasets}
    elif not isinstance(datasets, dict):
        raise ValueError(f"datasets must be str, list, or dict, got {type(datasets)}")

    # Create datasets
    datasets_list = []
    for dataset_name, dataset_opts in datasets.items():
        # Get default options
        split = dataset_opts.get("split", "train")
        max_datapoints = dataset_opts.get("max_datapoints", None)
        load_text = dataset_opts.get("load_text", LoadTextMode.REPORTS)
        ctrate_filter_mode = dataset_opts.get("ctrate_filter_mode", CTRateFilterMode.FIRST_ALL)

        # Use NpzReader for NPZ-based datasets
        image_reader = None
        if dataset_name == DatasetNameC.RADCHESTCT:
            image_reader = NpzReader()

        # Create transform with dataset-specific features_dataset_name
        dataset_transform = get_train_transform(
            pixdim=pixdim,
            img_size=img_size,
            sliding_window_size=sliding_window_size,
            image_feat_mode=image_feat_mode,
            text_feat_mode=text_feat_mode,
            features_dataset_name=dataset_name,
            features_model_name=features_model_name,
            do_snippet_alignment=do_snippet_alignment,
            model_settings=model_settings,
            compose_class=compose_class,
            text_transform_mode=text_transform_mode,
            language=language,
            max_num_icd10=max_num_icd10,
            keep_original_prob=keep_original_prob,
            drop_prob=drop_prob,
            dtype=dtype,
            use_gds=use_gds,
            random_flip=random_flip,
            random_crop_features=float(random_crop_features),
            random_flip_features=float(random_flip_features),
            organ_text_replace_prob=organ_text_replace_prob,
            no_comp_replace_prob=no_comp_replace_prob,
            image_reader=image_reader,
        )

        add_slices = do_snippet_alignment is not None and do_snippet_alignment.get("enabled", False)
        dataset = get_dataset(
            dataset_name=dataset_name,
            split=split,
            max_datapoints=max_datapoints,
            load_text=load_text,
            ctrate_filter_mode=ctrate_filter_mode,
            transform=dataset_transform,
            data_fraction=data_fraction,
            add_slices=add_slices,
            sample_scan_per_dedup_report=sample_scan_per_dedup_report,
        )
        extra = ""
        if dataset_name == DatasetNameC.CTRATE:
            extra = f", filter_mode={ctrate_filter_mode}"
        log_info(
            f"Dataset {dataset_name} split={split}: {len(dataset):_d} samples "
            f"(data_fraction={data_fraction}{extra})"
        )
        datasets_list.append(dataset)

    # Combine datasets if multiple
    dataset = datasets_list[0] if len(datasets_list) == 1 else ConcatDataset(datasets_list)

    # Create tokenizer
    tokenizer = Qwen2TokenizerFast.from_pretrained(text_tokenizer)

    # Create dataloader
    prefetch_factor_val = prefetch_factor if workers > 0 else None
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        prefetch_factor=prefetch_factor_val,
        shuffle=shuffle,
        collate_fn=partial(
            extended_collate_siglip,
            tokenizer=tokenizer,
            tokenizer_max_length=tokenizer_max_length,
            allow_none=True,
            sliding_window_size=sliding_window_size,
            min_area_for_padding=min_area_for_padding,
        ),
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return dataloader


def get_train_dataloader_from_config(
    features_model_name,
    text_transform_mode,
    model_config: dict,
    train_config: dict,
    timed_compose: bool = False,
    **kwargs,
) -> data.DataLoader:
    image_feat_mode = train_config["train"]["image_feat_mode"]
    text_feat_mode = train_config["train"]["text_feat_mode"]
    img_size = model_config["train_img_size"]
    pixdim = model_config["backbone_kwargs"]["pixdim"]
    sliding_window_size = model_config["backbone_kwargs"]["sliding_window_size"]
    min_area_for_padding = model_config["backbone_kwargs"]["min_area_for_padding"]
    # Calculate grid size: number of sliding windows in each dimension
    grid_size = []
    for i in range(3):
        div_here, mod_here = divmod(img_size[i], sliding_window_size[i])
        if mod_here != 0:
            raise ValueError(
                f"img_size {img_size} is not divisible by sliding_window_size {sliding_window_size}"
            )
        grid_size.append(div_here)
    grid_size = tuple(grid_size)
    log_debug(f"img_size={img_size}, sliding_window_size={sliding_window_size}")
    log_debug(f"Computed grid_size={grid_size}")
    prefetch_factor = train_config["train"]["prefetch_factor"]
    workers = train_config["train"]["workers"]
    persistent_workers = train_config["train"]["persistent_workers"]
    if workers == 0:
        prefetch_factor = None
        persistent_workers = False
    # Determine dtype for loading
    dtype = "float16" if train_config["train"]["load_fp16"] else "float32"
    log_debug(f"Loading data as {dtype}")
    compose_class = ReprCompose if not timed_compose else TimedCompose
    model_settings = train_config["train"].get("model_settings", {})
    organ_text_cfg = train_config["train"].get("organ_text", {})
    organ_text_replace_prob = (
        organ_text_cfg.get("replace_prob", 0.0) if organ_text_cfg.get("enabled", False) else 0.0
    )
    no_comp_replace_prob = train_config["train"].get("no_comp_replace_prob", 0.0)
    dataloader = get_train_dataloader(
        datasets=train_config["train"]["datasets"],
        image_feat_mode=image_feat_mode,
        text_feat_mode=text_feat_mode,
        pixdim=pixdim,
        img_size=img_size,
        sliding_window_size=sliding_window_size,
        min_area_for_padding=min_area_for_padding,
        features_model_name=features_model_name,
        text_transform_mode=text_transform_mode,
        do_snippet_alignment=train_config["train"].get("do_snippet_alignment"),
        model_settings=model_settings,
        language=train_config["train"]["language"],
        text_tokenizer=model_config["text_tokenizer"],
        tokenizer_max_length=model_config["text_backbone_kwargs"]["tokenizer_max_length"],
        batch_size=train_config["train"]["batch_size"],
        workers=workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=train_config["train"]["pin_memory"],
        drop_last=train_config["train"]["drop_last"],
        use_gds=train_config["train"]["use_gds"],
        compose_class=compose_class,
        data_fraction=train_config["train"]["data_fraction"],
        dtype=dtype,
        random_flip=train_config["train"]["random_flip"],
        random_crop_features=train_config["train"].get("random_crop_features", 0.0),
        random_flip_features=train_config["train"].get("random_flip_features", 0.0),
        organ_text_replace_prob=organ_text_replace_prob,
        no_comp_replace_prob=no_comp_replace_prob,
        drop_prob=train_config["train"].get("drop_findings_prob", 0.3),
        sample_scan_per_dedup_report=train_config["train"].get(
            "sample_scan_per_dedup_report", False
        ),
        **kwargs,
    )
    return dataloader
