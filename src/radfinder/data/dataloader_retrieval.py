from functools import partial
from typing import Any

from monai.data import DataLoader
from radfinder.data.ct_rate import CTRateFilterMode
from radfinder.data.dataloader_train import get_dataset
from radfinder.data.rad_chestct import NpzReader
from radfinder.models.load_model import FeatMode
from radfinder.transforms.eval_transform import TextTransformMode, get_eval_transform
from radfinder.transforms.new_compose import ReprCompose
from radfinder.transforms.shared_utils import Language, LoadTextMode, TransformKindC
from radfinder.utils.collate import extended_collate_siglip
from radfinder.utils.logging_utils import log_info
from transformers import AutoTokenizer


def get_retrieval_dataloader(
    model_config: dict,
    model_config_name: str,
    dataset_name: str,
    split: str,
    # ways to subset the dataset
    max_datapoints: int | None = None,
    data_fraction: float = 1.0,
    key_subset: list[str] | None = None,
    # dataloader settings
    batch_size: int = 1,
    workers: int = 1,
    prefetch_factor: int | None = 2,
    image_feat_mode: str = FeatMode.FULL,
    text_feat_mode: str = FeatMode.FULL,
    do_snippet_alignment: dict | None = None,
    model_settings: dict | None = None,
    lazy: bool = False,
    # dataset-specific settings
    language: str = Language.EN,
    compose_class=ReprCompose,
    add_slices: bool = False,
    ctrate_filter_mode: str = CTRateFilterMode.FIRST_ALL,
    load_text: str = LoadTextMode.REPORTS,
    drop_findings: bool = False,
    drop_impressions: bool = False,
    drop_prefix: bool = False,
) -> tuple[DataLoader, Any]:
    """Create evaluation transform, dataset, and dataloader for retrieval."""
    # Extract configuration parameters
    sliding_window_size = model_config["backbone_kwargs"]["sliding_window_size"]
    pixdim = model_config["backbone_kwargs"]["pixdim"]
    min_area_for_padding = model_config["backbone_kwargs"]["min_area_for_padding"]
    transform_kind = model_config.get("transform_kind", TransformKindC.SPECTRE)
    target_shape = model_config.get("target_shape")
    text_backbone_kwargs = model_config.get("text_backbone_kwargs", {}) or {}
    tokenizer_max_length = text_backbone_kwargs.get("tokenizer_max_length", 4096)
    tokenizer_padding = text_backbone_kwargs.get("tokenizer_padding", True)

    # Use NpzReader for NPZ-based datasets
    image_reader = None
    if dataset_name == "radchestct":
        image_reader = NpzReader()

    # create transform that loads features or raw data depending on settings
    ttm = (
        TextTransformMode.NONE
        if load_text == LoadTextMode.NONE
        or text_feat_mode == FeatMode.NONE
        or dataset_name == "radchest"
        else TextTransformMode.DEFAULT
    )

    transform = get_eval_transform(
        pixdim=pixdim,
        sliding_window_size=sliding_window_size,
        min_area_for_padding=min_area_for_padding,
        image_feat_mode=image_feat_mode,
        text_feat_mode=text_feat_mode,
        features_dataset_name=dataset_name,
        features_model_name=model_config_name,
        do_snippet_alignment=do_snippet_alignment,
        model_settings=model_settings,
        compose_class=compose_class,
        language=language,
        lazy=lazy,
        image_reader=image_reader,
        text_transform_mode=ttm,
        drop_findings=drop_findings,
        drop_impressions=drop_impressions,
        drop_prefix=drop_prefix,
        transform_kind=transform_kind,
        target_shape=target_shape,
    )

    # create dataset
    dataset = get_dataset(
        dataset_name,
        split=split,
        transform=transform,
        max_datapoints=max_datapoints,
        data_fraction=data_fraction,
        key_subset=key_subset,
        ctrate_filter_mode=ctrate_filter_mode,
        add_slices=add_slices,
        load_text=load_text,
        sample_scan_per_dedup_report=False,  # not needed for eval
    )
    extra = f", filter_mode={ctrate_filter_mode}" if dataset_name == "ctrate" else ""
    log_info(
        f"Dataset {dataset_name} split={split}: {len(dataset):_d} samples "
        f"(data_fraction={data_fraction}{extra})"
    )

    # Create dataloader
    prefetch_factor_final = prefetch_factor if workers > 0 else None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["text_tokenizer"],
        trust_remote_code=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        prefetch_factor=prefetch_factor_final,
        shuffle=False,
        collate_fn=partial(
            extended_collate_siglip,
            tokenizer=tokenizer,
            tokenizer_padding=tokenizer_padding,
            tokenizer_max_length=tokenizer_max_length,
            sliding_window_size=sliding_window_size,
            min_area_for_padding=min_area_for_padding,
        ),
    )

    return dataloader, dataset
