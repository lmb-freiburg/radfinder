"""
Binary zero-shot classification evaluation script.

Evaluates binary classification on CT-RATE (18 pathologies) using zero-shot
text prompts and AUROC metric. Reuses the model/dataloader infrastructure
from retrieval_task.py.

Usage:
    python -m radfinder.cli.eval_binary_zs --dataset_name ctrate -- split val
    python -m radfinder.cli.eval_binary_zs --dataset_name radchestct --split all --radchestct_label_mapping standard --eval_protocol radchestct_standard
"""

from pathlib import Path

import pandas as pd
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
from radfinder.tasks.binary_zs_ctrate_task import PromptModeC
from radfinder.tasks.run_task import run_task_by_type
from radfinder.transforms.new_compose import ReprCompose
from radfinder.transforms.shared_utils import LoadTextMode
from radfinder.utils.config import load_config_without_types, random_seed
from radfinder.utils.logging_utils import configure_logging, log_debug, log_info
from torch import nn

from typedparser import TypedParser, VerboseQuietArgs, add_argument


@define
class Args(VerboseQuietArgs):
    model_cfg: Path = add_argument(default=DEFAULT_MODEL_CONFIG_FILE)
    train_cfg: str | None = add_argument(help="Train config file (for snippet alignment settings)")
    dataset_name: str = add_argument(default="ctrate", help="Dataset name for evaluation")
    image_feat_mode: str = add_argument(default=FeatMode.FROZEN_LOCAL, help="Image feature mode")
    text_feat_mode: str = add_argument(default=FeatMode.FULL, help="Text feature mode")
    split: str = add_argument(default="val", help="Dataset split to use for evaluation")
    max_datapoints: int | None = add_argument(type=int, help="Maximum number of datapoints to use")
    batch_size: int = add_argument(type=int, default=16, help="Batch size for the dataloader")
    workers: int = add_argument(type=int, default=1, help="Number of workers for the dataloader")
    prefetch_factor: int = add_argument(
        type=int, default=2, help="Prefetch factor for the dataloader"
    )
    ckpt_file: str | None = add_argument(default=None, help="Checkpoint file to load weights from")
    radchestct_label_mapping: str = add_argument(
        default="extended", help="'standard' or 'extended'"
    )
    eval_protocol: str = add_argument(default="default", help="'default' or 'radchestct_standard'")
    prompt_mode: str = add_argument(
        default=PromptModeC.MEAN7, help="Prompt template id passed to binary_zs"
    )
    bootstrap: bool = add_argument(action="store_true", help="Also compute bootstrap CIs")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    args.image_feat_mode = FeatMode.verify_value(args.image_feat_mode)
    args.text_feat_mode = FeatMode.verify_value(args.text_feat_mode)
    configure_logging(args)
    log_info(f"{args}")
    all_metrics = main_eval_binary_zs(args)
    log_info("#################### Binary Zero-Shot Classification ####################")
    for k, v in all_metrics.items():
        if k == "mean_auroc":
            continue
        log_info(f"  {k:<45s}: {v:.4f}")
    log_info(
        f"  {'Mean AUROC':<45s}: {all_metrics['mean_auroc']:.4f} ({all_metrics['mean_auroc']:.2%})"
    )


def main_eval_binary_zs(args: Args):
    device = "cuda"
    log_info(f"Using device: {device}")

    # load model config and create model
    model_config_file = args.model_cfg
    model_config = load_config_without_types(model_config_file)
    log_debug(f"Model config: {model_config}")

    train_config = None
    if args.train_cfg is not None:
        train_config = load_config_without_types(args.train_cfg)
    do_snippet_alignment, _ = resolve_arch_settings(model_config, train_config)

    model = create_siglip(
        model_config, args.image_feat_mode, args.text_feat_mode, train_config=train_config
    )
    if args.ckpt_file is not None:
        model.load_checkpoint(args.ckpt_file)

    # Create dataloader using shared retrieval infrastructure
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
        compose_class=ReprCompose,
        do_snippet_alignment=do_snippet_alignment,
        add_slices=False,
        load_text=LoadTextMode.NONE,
    )

    # setup accelerator
    random_seed(42)
    dataloader_config = DataLoaderConfiguration(non_blocking=True)
    accelerator = Accelerator(dataloader_config=dataloader_config, mixed_precision="bf16")
    device = accelerator.device
    log_info(f"Using device: {device}")

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model, dataloader = accelerator.prepare(model, dataloader)

    # Run binary zero-shot evaluation
    task_config = {
        "task_type": "binary_zs",
        "dataset_name": args.dataset_name,
        "split": args.split,
        "prompt_mode": args.prompt_mode,
        "radchestct_label_mapping": args.radchestct_label_mapping,
        "eval_protocol": args.eval_protocol,
    }
    with accelerator.autocast(), accelerator.no_sync(model), torch.inference_mode():
        all_metrics, _aux_metrics, bootstrap_metrics = run_task_by_type(
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
