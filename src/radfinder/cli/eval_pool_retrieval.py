"""
Pool retrieval evaluation (Merlin protocol).

Usage:
    python -m radfinder.cli.eval_pool_retrieval --dataset_name merlin --split test
"""

from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from attrs import define
from radfinder.data.dataloader_retrieval import get_retrieval_dataloader
from radfinder.models.load_model import (
    DEFAULT_MODEL_CONFIG_FILE,
    FeatMode,
    create_siglip,
    resolve_arch_settings,
)
from radfinder.tasks.run_task import run_task_by_type
from radfinder.transforms.new_compose import ReprCompose, TimedCompose
from radfinder.transforms.shared_utils import Language
from radfinder.utils.config import load_config_without_types, random_seed
from radfinder.utils.logging_utils import configure_logging, log_debug, log_info
from torch import nn

from typedparser import TypedParser, VerboseQuietArgs, add_argument


@define
class Args(VerboseQuietArgs):
    model_cfg: Path = add_argument(default=DEFAULT_MODEL_CONFIG_FILE)
    train_cfg: str | None = add_argument(help="Train config file (for snippet alignment settings)")
    dataset_name: str = add_argument(default="merlin", help="Dataset name for evaluation")
    image_feat_mode: str = add_argument(default=FeatMode.FROZEN_LOCAL, help="Image feature mode")
    text_feat_mode: str = add_argument(default=FeatMode.FULL, help="Text feature mode")
    split: str = add_argument(default="test", help="Dataset split to use for evaluation")
    max_datapoints: int | None = add_argument(type=int, help="Maximum number of datapoints to use")
    batch_size: int = add_argument(type=int, default=16, help="Batch size for the dataloader")
    workers: int = add_argument(type=int, default=4, help="Number of workers for the dataloader")
    prefetch_factor: int = add_argument(
        type=int, default=2, help="Prefetch factor for the dataloader"
    )
    language: str = add_argument(
        default=Language.EN, help="Language for report generation: en, de, both"
    )
    ckpt_file: str | None = add_argument(default=None, help="Checkpoint file to load weights from")
    timed_compose: bool = add_argument(action="store_true", help="Use timed compose")
    print_transform: bool = add_argument(action="store_true", help="Print transform")
    pool_sizes: list[int] | None = add_argument(
        type=int, nargs="+", help="Pool sizes (default: 32 64 128)"
    )
    ks: list[int] | None = add_argument(type=int, nargs="+", help="Recall@k values (default: 1 8)")
    repeats: int | None = add_argument(type=int, help="Number of random repeats (default: 100)")
    seed: int = add_argument(type=int, default=42, help="Random seed for pool sampling")
    bootstrap: bool = add_argument(action="store_true", help="Also compute bootstrap CIs")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    args.image_feat_mode = FeatMode.verify_value(args.image_feat_mode)
    args.text_feat_mode = FeatMode.verify_value(args.text_feat_mode)
    configure_logging(args)
    log_info(f"{args}")
    all_metrics = main_eval_pool_retrieval(args)
    log_info("#################### pool retrieval ####################")
    for k, v in all_metrics.items():
        log_info(f"{k:>30}: {v * 100:6.2f}" if isinstance(v, float) else f"{k:>30}: {v}")


def main_eval_pool_retrieval(args: Args):
    device = "cuda"
    log_info(f"Using device: {device}")

    model_config_file = args.model_cfg
    model_config = load_config_without_types(model_config_file)
    log_debug(f"Model config: {model_config}")

    train_config = None
    if args.train_cfg is not None:
        train_config = load_config_without_types(args.train_cfg)
    do_snippet_alignment, model_settings = resolve_arch_settings(model_config, train_config)

    model = create_siglip(
        model_config, args.image_feat_mode, args.text_feat_mode, train_config=train_config
    )
    if args.ckpt_file is not None:
        model.load_checkpoint(args.ckpt_file)

    dataloader, dataset = get_retrieval_dataloader(
        model_config=model_config,
        model_config_name=model_config_file.stem,
        dataset_name=args.dataset_name,
        split=args.split,
        max_datapoints=args.max_datapoints,
        data_fraction=1.0,
        key_subset=None,
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        image_feat_mode=args.image_feat_mode,
        text_feat_mode=args.text_feat_mode,
        lazy=False,
        compose_class=TimedCompose if args.timed_compose else ReprCompose,
        language=args.language,
        do_snippet_alignment=do_snippet_alignment,
        model_settings=model_settings,
    )
    if args.print_transform:
        log_info(f"Transform: {dataset.transform}")

    random_seed(42)
    dataloader_config = DataLoaderConfiguration(non_blocking=True)
    accelerator = Accelerator(dataloader_config=dataloader_config, mixed_precision="bf16")

    device = accelerator.device
    log_info(f"Using device: {device}")
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model, dataloader = accelerator.prepare(model, dataloader)

    task_config = {
        "task_type": "pool_retrieval",
        "dataset_name": args.dataset_name,
        "split": args.split,
        "pool_sizes": args.pool_sizes,
        "ks": args.ks,
        "repeats": args.repeats,
        "seed": args.seed,
    }
    with accelerator.autocast(), accelerator.no_sync(model), torch.inference_mode():
        all_metrics, _aux, bootstrap_metrics = run_task_by_type(
            task_config=task_config,
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            model_config=model_config,
            device=device,
            bootstrap=args.bootstrap,
            verbose=args.verbose,
        )

    if bootstrap_metrics is not None:
        log_info(f"Bootstrap CIs: {bootstrap_metrics}")
    return all_metrics


if __name__ == "__main__":
    main()
