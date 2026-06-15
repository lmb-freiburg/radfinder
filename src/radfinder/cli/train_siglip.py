"""
Train RadFinder model. See README
"""

import resource
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, DataLoaderConfiguration
from attr import define
from radfinder.data.ct_rate import CTRateFilterMode
from radfinder.data.dataloader_retrieval import get_retrieval_dataloader
from radfinder.data.dataloader_train import get_train_dataloader_from_config
from radfinder.data.prompt_rate_labels import (
    PromptRateModeC,
    build_prompt_rate_labels,
    load_prompts_for_mode,
)
from radfinder.losses.prompt_rate_loss import PromptRateLoss
from radfinder.models.load_model import (
    DEFAULT_MODEL_CONFIG_FILE,
    DEFAULT_TRAINING_CONFIG_FILE,
    create_siglip_from_train_cfg,
)
from radfinder.paths import RADFINDER_REPO_DIR, get_medv_output_dir
from radfinder.tasks.binary_zs_rate_task import get_labeled_scan_keys
from radfinder.trainer_siglip import Trainer
from radfinder.transforms.shared_utils import LoadTextMode
from radfinder.transforms.train_transform import TextTransformMode
from radfinder.utils.checkpoint_paths import filter_best_last_checkpoints
from radfinder.utils.config import (
    apply_scaling_rules_to_cfg,
    load_config_without_types,
    random_seed,
)
from radfinder.utils.logging_utils import configure_logging, log_error, log_info, log_warning
from radfinder.utils.param_groups import get_param_groups_with_decay, print_optimizer_parameters
from torch.optim import AdamW
from transformers import Adafactor, Qwen2TokenizerFast

from packg import format_exception
from typedparser import TypedParser, VerboseQuietArgs, add_argument
from visiontext.profiling.code_profiler import (
    start_pyinstrument_profiler,
    stop_pyinstrument_profiler,
)


@define
class Args(VerboseQuietArgs):
    full_run_name: str | None = add_argument()
    run: str | None = add_argument()
    phase: str = add_argument(default="both", choices=["both", "train", "eval"])

    model_cfg: Path = add_argument(default=DEFAULT_MODEL_CONFIG_FILE)
    train_cfg: Path = add_argument(default=DEFAULT_TRAINING_CONFIG_FILE)
    text_transform_mode: str = add_argument(default=TextTransformMode.DEFAULT)
    wandb: bool = add_argument(action="store_true")
    options: list[str] | None = add_argument(shortcut="-o", action="append")
    max_memory: int | None = add_argument(
        type=int,
        help="Maximum CPU memory in GB (e.g., 64 for 64GB). If set, limits process memory.",
    )
    print_train_transform: bool = add_argument(action="store_true")
    print_optimizer_params: bool = add_argument(action="store_true")
    timed_compose: bool = add_argument(action="store_true")
    profile: bool = add_argument(action="store_true")
    test_tasks: list[str] = add_argument(shortcut="-t", action="append", default=[])
    test_all: bool = add_argument(
        action="store_true", help="Evaluate all checkpoints (default: best + last only)"
    )
    bootstrap: bool = add_argument(
        action="store_true", help="Compute bootstrap CIs for retrieval tasks"
    )
    assert_done_training: bool = add_argument(
        action="store_true", help="Fail if training is not already completed (for eval-only tests)"
    )
    override_checkpoint_path: Path | None = add_argument()


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logging(args)
    log_info(f"{args}")
    main_train_siglip(args)


def main_train_siglip(args: Args):
    # Limit CPU memory if requested
    if args.max_memory is not None:
        LIMM = args.max_memory * 1024**3
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (LIMM, hard))
        resource.setrlimit(resource.RLIMIT_DATA, (LIMM, hard))
        log_info(f"Limited CPU memory to {args.max_memory}GB")

    #################### Load and setup training config ####################
    train_config = load_config_without_types(args.train_cfg, merge_dotlist=args.options)
    # determine run name and set output dir
    full_run_name = args.full_run_name
    if full_run_name is None:
        full_run_name = args.train_cfg.stem
    if args.run is not None:
        full_run_name += f"_{args.run}"
    output_dir = Path(get_medv_output_dir()) / args.train_cfg.parent.name / full_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_config["train"]["output_dir"] = output_dir.as_posix()
    random_seed(train_config["train"]["seed"])
    log_info(f"Output dir: {output_dir}")

    #################### accelerator setup ####################
    dataloader_config = DataLoaderConfiguration(
        non_blocking=train_config["train"]["pin_memory"],
    )
    mixed_precision = train_config["train"]["mixed_precision"]  # "no", "fp16", "bf16", or "fp8"
    use_wandb = args.wandb
    accelerator_kwargs = dict(
        log_with="wandb" if use_wandb else None,
        dataloader_config=dataloader_config,
        mixed_precision=mixed_precision,
    )
    if mixed_precision == "fp8":
        raise NotImplementedError("FP8 training not implemented")
    accelerator = Accelerator(**accelerator_kwargs)
    log_info(f"Mixed precision training: {mixed_precision}")
    accum_steps = train_config["train"]["accum_steps"]
    log_info(f"Manual gradient accumulation steps: {accum_steps}")
    # Initialize wandb
    if use_wandb:
        accelerator.init_trackers(
            project_name="radfinder",
            config={"train_config": train_config, "args": vars(args)},
            init_kwargs={
                "wandb": {
                    "dir": (output_dir / "logs").as_posix(),
                    "name": full_run_name,
                    "resume": "allow",
                    "id": full_run_name,
                }
            },
        )

    # Apply learning rate scaling rules
    train_config = apply_scaling_rules_to_cfg(train_config)
    log_info(
        f"Applied LR scaling: base_lr={train_config["optim"]["base_lr"]:.2e} -> "
        f"lr={train_config["optim"]["lr"]:.2e}"
    )
    device = accelerator.device
    log_info(f"Using device: {device}")
    if device == "cpu":
        raise RuntimeError("Training on CPU is not supported")
    log_info("Training config:")
    log_info(f"{train_config}")

    #################### Load and setup model ####################
    model_config = load_config_without_types(args.model_cfg)
    log_info("Model config:")
    log_info(f"{model_config}")
    model, image_feat_mode, text_feat_mode = create_siglip_from_train_cfg(
        model_config, train_config
    )

    #################### Create optimizer ####################
    param_groups = get_param_groups_with_decay(
        model,
        llrd_factor=train_config["optim"]["llrd_factor"],
        patch_embed_lr_mult=train_config["optim"]["patch_embed_lr_mult"],
        lora_lr_factor=1.0,
    )
    # print_all_params()

    # sort which params to add to the optimizer
    param_groups_grad = []
    for pg in param_groups:
        if pg["requires_grad"]:
            param_groups_grad.append(pg)
            continue
        if train_config["optim"]["freeze_image_backbone_epochs"] > 0:
            # features are frozen now, but will be defrosted later, so they must be added to the opt
            new_params, new_param_names = [], []
            for param, param_name in zip(pg["params"], pg["param_names"]):
                if param_name.startswith("backbone_image."):
                    new_params.append(param)
                    new_param_names.append(param_name)
            if len(new_params) == 0:
                continue
            pg["params"] = new_params
            pg["param_names"] = new_param_names
            param_groups_grad.append(pg)

    wd = train_config["optim"]["weight_decay"]
    lr = train_config["optim"]["lr"]
    opt_name = train_config["optim"]["optimizer"]
    if opt_name == "adamw":
        optimizer = AdamW(
            param_groups_grad,
            lr=lr,
            betas=(train_config["optim"]["adamw_beta1"], train_config["optim"]["adamw_beta2"]),
            weight_decay=wd,
        )
    elif opt_name == "adafactor":
        optimizer = Adafactor(
            param_groups_grad,
            lr=lr,
            clip_threshold=99999,  # disable since trainer also does gradient clipping
            weight_decay=wd,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    if args.print_optimizer_params:
        print_optimizer_parameters(optimizer, log_fn=log_info)
        # print_optimizer_parameters_all(param_groups)

    #################### Load checkpoint if exists ####################
    start_epoch = 0
    checkpoint_path = output_dir / "checkpoint.pt"
    if args.override_checkpoint_path is not None:
        checkpoint_path = args.override_checkpoint_path
    checkpoint = None
    if checkpoint_path.exists():
        log_info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model_state_dict = checkpoint["model"]
        # old checkpoint where the criterion was still outside the model
        if "criterion" in checkpoint:
            log_warning("Deprecated checkpoint")
            criterion_state_dict = checkpoint["criterion"]
            for k, v in criterion_state_dict.items():
                model_state_dict[f"criterion.{k}"] = v
        load_result = model.load_state_dict(model_state_dict, strict=False)

        if load_result.missing_keys:
            log_info(
                f"  Missing keys (OK - frozen parameters): {len(load_result.missing_keys)} keys"
            )

        if load_result.unexpected_keys:
            error_msg = (
                f"ERROR: Unexpected keys in checkpoint:\n"
                f"  {load_result.unexpected_keys}\n"
                f"This likely means checkpoint was saved with different architecture."
            )
            log_warning(error_msg)
            raise RuntimeError(error_msg)
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as e:
            log_warning(f"Failed to load optimizer state: {format_exception(e)}")
            optimizer = None

        start_epoch = checkpoint["epoch"]
        log_info(f"Resuming from epoch {start_epoch}")

    pr_cfg = train_config["train"].get("prompt_rate", {})
    pr_loss_module = None
    pr_labels = None
    pr_tokenizer = None
    pr_pos_prompts = pr_neg_prompts = None
    snippet_cfg = train_config["train"].get("do_snippet_alignment", {})
    val_snippet_alignment = snippet_cfg or None
    if start_epoch >= train_config["optim"]["epochs"]:
        log_info(f"Training already completed for {train_config['optim']['epochs']} epochs")
        unwrapped_model = model
        dataloader = None
        val_dict = {}
    else:
        if args.assert_done_training:
            raise RuntimeError(
                f"assert_done_training: expected training to be done but "
                f"start_epoch={start_epoch} < epochs={train_config['optim']['epochs']}. "
                f"Checkpoint dir: {output_dir}"
            )
        #################### Create training dataloader ####################
        # for now always need the dataloader because we need it to infer global step
        dataloader = get_train_dataloader_from_config(
            features_model_name=args.model_cfg.stem,
            text_transform_mode=args.text_transform_mode,
            model_config=model_config,
            train_config=train_config,
            timed_compose=args.timed_compose,
        )
        log_info(f"Train dataloader: {len(dataloader.dataset):_d} samples")
        if args.print_train_transform:
            print_transform_from_dataset(dataloader.dataset)

        #################### create validation tasks ####################
        val_tasks = train_config["train"]["validation_tasks"]
        # axis2_global needs axis2 features loaded for val retrieval
        model_settings = train_config["train"].get("model_settings")
        val_dict = get_val_tasks(
            val_tasks=val_tasks,
            model_config=model_config,
            model_config_name=args.model_cfg.stem,
            image_feat_mode=image_feat_mode,
            text_feat_mode=text_feat_mode,
            batch_size=train_config["train"]["batch_size"],
            val_workers=train_config["train"]["val_workers"],
            prefetch_factor=train_config["train"]["prefetch_factor"],
            do_snippet_alignment=val_snippet_alignment,
            model_settings=model_settings,
        )
        log_info(f"Validation tasks: {', '.join(val_dict.keys())}")

        #################### Initialize prompt-rate auxiliary loss ####################
        # TODO refactor this loss creation out of the training script
        if pr_cfg.get("enabled", False):
            pr_mode = pr_cfg.get("mode", PromptRateModeC.RATE)
            dataset_names = list(train_config["train"].get("datasets", {}).keys())
            pr_labels, labels_dict = build_prompt_rate_labels(
                dataset_names,
                split="train",
                mode=pr_mode,
            )
            assert len(pr_labels) > 0, (
                f"Prompt-rate: 0 labeled scans for datasets={dataset_names}, mode={pr_mode}. "
                f"Check that label files exist and scan keys match."
            )

            # Compute pos_weight per question from training labels
            all_labels = np.stack(list(labels_dict.values()))
            num_q = all_labels.shape[1]
            n_valid_per_q = (all_labels != -1).sum(axis=0)
            n_dead_questions = (n_valid_per_q == 0).sum()
            if n_dead_questions > num_q * 0.5:
                log_error(
                    f"Prompt-rate: {n_dead_questions}/{num_q} questions have zero valid labels. "
                    f"This likely means label files are empty or keys don't match. "
                    f"Sample label key: {next(iter(labels_dict))}"
                )
            pos_weight_per_q = torch.ones(num_q)
            cap = pr_cfg.get("pos_weight_cap", 20.0)
            label_smoothing = pr_cfg.get("label_smoothing", 0.0)
            for qi in range(num_q):
                mask = all_labels[:, qi] != -1
                if mask.sum() == 0:
                    continue
                n_pos = (all_labels[mask, qi] == 1).sum()
                n_neg = (all_labels[mask, qi] == 0).sum()
                if n_pos > 0:
                    pos_weight_per_q[qi] = min(float(n_neg) / n_pos, cap)

            # Build per-question group-importance weights
            question_weight_per_q = None
            qgw_cfg = pr_cfg.get("question_group_weights")
            if qgw_cfg is not None:
                question_weight_per_q = torch.ones(num_q)
                if pr_mode == PromptRateModeC.RATE:
                    question_weight_per_q[:226] = qgw_cfg.get("abdomen", 1.0)
                    question_weight_per_q[226:319] = qgw_cfg.get("chest", 1.0)
                elif pr_mode == PromptRateModeC.CTRATE:
                    question_weight_per_q[:18] = qgw_cfg.get("ctrate", 1.0)
                elif pr_mode == PromptRateModeC.BOTH:
                    question_weight_per_q[:226] = qgw_cfg.get("abdomen", 1.0)
                    question_weight_per_q[226:319] = qgw_cfg.get("chest", 1.0)
                    question_weight_per_q[319:337] = qgw_cfg.get("ctrate", 1.0)
                else:
                    raise ValueError(
                        f"Unknown prompt-rate mode for question_group_weights: {pr_mode}"
                    )
                group_info = []
                if pr_mode in (PromptRateModeC.RATE, PromptRateModeC.BOTH):
                    eff_abd = 226 * qgw_cfg.get("abdomen", 1.0)
                    eff_ch = 93 * qgw_cfg.get("chest", 1.0)
                    group_info.append(f"abdomen: 226x{qgw_cfg.get('abdomen', 1.0)}={eff_abd:.0f}")
                    group_info.append(f"chest: 93x{qgw_cfg.get('chest', 1.0)}={eff_ch:.0f}")
                if pr_mode in (PromptRateModeC.CTRATE, PromptRateModeC.BOTH):
                    eff_ct = 18 * qgw_cfg.get("ctrate", 1.0)
                    group_info.append(f"ctrate: 18x{qgw_cfg.get('ctrate', 1.0)}={eff_ct:.0f}")
                total_eff = question_weight_per_q.sum().item()
                pcts = ", ".join(group_info)
                log_info(f"Question group weights: {pcts} (total effective: {total_eff:.0f})")

            pr_loss_module = PromptRateLoss(
                pos_weight_per_q, label_smoothing, question_weight_per_q
            ).to(device)

            # Load prompts based on mode
            pr_pos_prompts, pr_neg_prompts = load_prompts_for_mode(pr_mode)
            assert (
                len(pr_pos_prompts) == num_q
            ), f"Prompt count {len(pr_pos_prompts)} != label dim {num_q} for mode={pr_mode}"
            pr_tokenizer = Qwen2TokenizerFast.from_pretrained(model_config["text_tokenizer"])
            log_info(
                f"Prompt-rate: {len(pr_labels)} labeled scans, "
                f"weight={pr_cfg['weight']}, {num_q} questions, mode={pr_mode}"
            )

        #################### Prepare model, data, and optimizer with accelerator ####################
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)
        unwrapped_model = accelerator.unwrap_model(model)

        #################### Training setup ####################
        # Calculate steps accounting for gradient accumulation
        batches_per_epoch = len(dataloader)
        steps_per_epoch = batches_per_epoch // accum_steps
        total_num_steps = train_config["optim"]["epochs"] * steps_per_epoch
        warmup_num_steps = train_config["optim"]["warmup_epochs"] * steps_per_epoch

        log_info("Training setup:")
        log_info(f"  Epochs: {train_config['optim']['epochs']}")
        log_info(f"  Batches per epoch: {batches_per_epoch}")
        log_info(f"  Gradient accumulation steps: {accum_steps}")
        log_info(f"  Optimizer steps per epoch: {steps_per_epoch}")
        log_info(f"  Total optimizer steps: {total_num_steps}")
        log_info(f"  Warmup steps: {warmup_num_steps}")
        log_info(f"  Base LR: {train_config['optim']['lr']}")
        log_info(f"  Min LR: {train_config['optim']['min_lr']}")

    # Create trainer and start training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        accelerator=accelerator,
        train_config=train_config,
        model_config=model_config,
        val_dict=val_dict,
        prompt_rate_loss=pr_loss_module,
        prompt_rate_labels=pr_labels,
        prompt_rate_tokenizer=pr_tokenizer,
        prompt_rate_pos_prompts=pr_pos_prompts,
        prompt_rate_neg_prompts=pr_neg_prompts,
        bootstrap=args.bootstrap,
    )
    if args.phase == "train" or args.phase == "both":
        if args.profile:
            start_pyinstrument_profiler()
        trainer.start_training(start_epoch_num=start_epoch)
        if args.profile:
            stop_pyinstrument_profiler(open_in_browser=False)

    #################### tests ####################
    # disable wandb for evaluation
    accelerator = Accelerator(
        log_with=None, dataloader_config=dataloader_config, mixed_precision=mixed_precision
    )
    model = accelerator.prepare(unwrapped_model)
    trainer.model = model
    trainer.accelerator = accelerator

    if args.phase == "eval" or args.phase == "both":
        if args.test_tasks != []:
            all_test_tasks = args.test_tasks
        else:
            val_tasks_here = train_config["train"]["validation_tasks"]
            test_tasks_here = train_config["train"].get("test_tasks", [])
            if isinstance(val_tasks_here, str):
                val_tasks_here = [val_tasks_here]
            if val_tasks_here is None:
                val_tasks_here = []
            if isinstance(test_tasks_here, str):
                test_tasks_here = [test_tasks_here]
            if test_tasks_here is None:
                test_tasks_here = []
            all_test_tasks = val_tasks_here + test_tasks_here

        log_info(f"Evaluating tasks: {all_test_tasks}")
        test_tasks_new = [t for t in all_test_tasks if t not in trainer.val_dict.keys()]
        if len(test_tasks_new) > 0:
            # add only the missing tasks
            test_dict_new = get_val_tasks(
                val_tasks=test_tasks_new,
                model_config=model_config,
                model_config_name=args.model_cfg.stem,
                image_feat_mode=image_feat_mode,
                text_feat_mode=text_feat_mode,
                batch_size=train_config["train"]["batch_size"],
                val_workers=train_config["train"]["val_workers"],
                prefetch_factor=train_config["train"]["prefetch_factor"],
                do_snippet_alignment=val_snippet_alignment,
                model_settings=train_config["train"].get("model_settings"),
            )
            trainer.val_dict.update(test_dict_new)

        # eval all epochs or only best + last
        checkpoint_dir = Path(train_config["train"]["output_dir"])
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not args.test_all:
            checkpoints = filter_best_last_checkpoints(checkpoints)
        for ckpt_file in checkpoints:
            _, _, epoch, _, val_score = ckpt_file.name.removesuffix(".pt").split("_")
            epoch = int(epoch)
            val_score = float(val_score)
            log_info(f"Evaluating checkpoint: {ckpt_file} at epoch {epoch} score: {val_score:.2f}")
            if not trainer.does_epoch_need_validating(epoch):
                log_info(f"  Skipping epoch {epoch} because it has already been validated")
                continue
            model.load_checkpoint(ckpt_file)
            val_meanr = trainer.validate_epoch(epoch, accelerator_log=False)
            log_info(f"    Validation meanr: {val_meanr}")

        if len(checkpoints) == 0:
            log_info(
                f"Evaluating zero-shot model "
                f"(no training checkpoints found, OK if this is an eval run)"
            )
            trainer.validate_epoch(0, accelerator_log=False)
    log_info("DONE")
    return {"start_epoch": start_epoch}


def get_val_tasks(
    val_tasks: list[str],
    model_config: dict,
    model_config_name: str,
    image_feat_mode: str,
    text_feat_mode: str,
    batch_size: int,
    val_workers: int,
    prefetch_factor: int,
    do_snippet_alignment: dict | None = None,
    model_settings: dict | None = None,
) -> dict:
    val_dict = {}
    val_list = []

    # expand task list into list of single tasks

    def _expand_task(val_task_local: str):
        val_task_file = Path(val_task_local)
        if not val_task_file.is_absolute():
            val_task_file = RADFINDER_REPO_DIR / val_task_file
        val_task_config = load_config_without_types(val_task_file)
        return val_task_file, val_task_config

    for val_task in val_tasks:
        val_task_file, val_task_config = _expand_task(val_task)
        if val_task_config.get("is_list", False):
            for file in val_task_config["files"]:
                listed_val_task_file, listed_val_task_config = _expand_task(file)
                val_list.append((val_task, listed_val_task_file, listed_val_task_config))
        else:
            val_list.append((val_task, val_task_file, val_task_config))

    for val_task, val_task_file, val_task_config in val_list:
        task_type = val_task_config.get("task_type", "retrieval")
        needs_slices = task_type == "localization"
        task_snippet_alignment = do_snippet_alignment
        if needs_slices and (
            task_snippet_alignment is None or not task_snippet_alignment.get("enabled", False)
        ):
            task_snippet_alignment = {
                "enabled": True,
                "snippet_mode": "axis_localization",
            }

        key_subset = None
        if task_type == "binary_zs_rate":
            key_subset = get_labeled_scan_keys(
                dataset_name=val_task_config["dataset_name"],
                split=val_task_config["split"],
                modality=val_task_config.get("modality", "abdomen_chest"),
                language=val_task_config.get("language", "en"),
            )

        load_text = LoadTextMode.REPORTS
        if task_type in {"binary_zs"}:
            load_text = LoadTextMode.NONE

        val_dataloader, val_dataset = get_retrieval_dataloader(
            model_config=model_config,
            model_config_name=model_config_name,
            dataset_name=val_task_config["dataset_name"],
            split=val_task_config["split"],
            max_datapoints=val_task_config.get("max_datapoints", None),
            data_fraction=1.0,
            key_subset=key_subset,
            batch_size=batch_size,
            workers=val_workers,
            prefetch_factor=prefetch_factor,
            image_feat_mode=image_feat_mode,
            text_feat_mode=text_feat_mode,
            lazy=False,
            do_snippet_alignment=task_snippet_alignment,
            model_settings=model_settings,
            language=val_task_config.get("language", "en"),
            add_slices=needs_slices,
            ctrate_filter_mode=val_task_config.get(
                "ctrate_filter_mode", CTRateFilterMode.FIRST_ALL
            ),
            load_text=load_text,
        )
        val_dict[val_task_file.stem] = (val_task_config, val_dataloader, val_dataset)
    return val_dict


def print_transform_from_dataset(dataset):
    # helper function to print the transform from the dataset (which is wrapped in a dataloader)
    if hasattr(dataset, "transform"):
        log_info("Transform of dataset:")
        log_info(f"Transform: {dataset.transform}")
        return
    if hasattr(dataset, "datasets"):
        log_info("ConcatDataset detected (multiple datasets):")
        for i, ds in enumerate(dataset.datasets):
            if hasattr(ds, "transform"):
                log_info(f"Transform for dataset {i}: {ds.transform}")
                break
            else:
                log_warning(f"Could not find transform in dataset {i}")
        return
    log_error("Could not find transform in dataset")


if __name__ == "__main__":
    main()
