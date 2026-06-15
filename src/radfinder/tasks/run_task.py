"""
Shared task-dispatch helper.

`run_task_by_type` consolidates the `if/elif task_type ==` ladder that lived in
both `trainer_siglip.Trainer.validate_epoch` and every standalone eval CLI. Each
caller builds a `task_config` dict (from YAML in the trainer, from argparse in
the CLIs) and the dispatcher selects the matching `run_*` function — or its
`run_*_with_bootstrap` variant when `bootstrap=True`.

Callers stay responsible for everything around the run call: model construction,
dataloader creation, accelerator / autocast wrapping, output file naming,
dumping JSON, wandb logging.
"""

from typing import Any

from monai.data import DataLoader
from radfinder.models.vision_language import SigLIP
from radfinder.tasks.binary_zs_ctrate_task import run_binary_zs
from radfinder.tasks.binary_zs_rate_task import run_binary_zs_rate
from radfinder.tasks.bootstrap_retrieval import (
    run_binary_zs_with_bootstrap,
    run_localization_with_bootstrap,
    run_pool_retrieval_with_bootstrap,
    run_retrieval_with_bootstrap,
    run_volume_retrieval_with_bootstrap,
)
from radfinder.tasks.localization_task import filter_one_slice_per_scan, run_localization
from radfinder.tasks.pool_retrieval_task import run_pool_retrieval
from radfinder.tasks.retrieval_task import run_retrieval
from radfinder.tasks.volume_retrieval_task import run_volume_retrieval

SUPPORTED_TASK_TYPES = (
    "retrieval",
    "pool_retrieval",
    "volume_retrieval",
    "binary_zs",
    "binary_zs_rate",
    "localization",
)

BOOTSTRAPPABLE_TASK_TYPES = (
    "retrieval",
    "pool_retrieval",
    "volume_retrieval",
    "binary_zs",
    "localization",
)


def run_task_by_type(
    task_config: dict,
    model: SigLIP,
    dataloader: DataLoader,
    dataset: Any,
    *,
    model_config: dict | None = None,
    device: str = "cuda",
    bootstrap: bool = False,
    verbose: bool = False,
) -> tuple[dict, dict | None, dict | None]:
    """
    Dispatch `task_config["task_type"]` to the right `run_*` function.

    Returns `(main_metrics, aux_metrics_or_None, bootstrap_metrics_or_None)`.

    `bootstrap=True` runs the corresponding `run_*_with_bootstrap` variant; only
    the task types listed in `BOOTSTRAPPABLE_TASK_TYPES` accept it.
    """
    task_type = task_config["task_type"]
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unknown task_type {task_type!r}; expected one of {SUPPORTED_TASK_TYPES}")
    if bootstrap and task_type not in BOOTSTRAPPABLE_TASK_TYPES:
        raise ValueError(
            f"bootstrap is not implemented for task_type={task_type!r}; "
            f"supported with bootstrap: {BOOTSTRAPPABLE_TASK_TYPES}"
        )

    if task_type == "retrieval":
        if bootstrap:
            main, boots = run_retrieval_with_bootstrap(
                model=model,
                dataloader=dataloader,
                dataset=dataset,
                device=device,
                verbose=verbose,
            )
            return main, None, boots
        main = run_retrieval(
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            device=device,
            verbose=verbose,
        )
        return main, None, None

    if task_type == "binary_zs":
        kwargs = dict(
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            device=device,
            dataset_name=task_config["dataset_name"],
            model_config=model_config,
            prompt_mode=task_config.get("prompt_mode", "t3"),
            radchestct_label_mapping=task_config.get("radchestct_label_mapping", "extended"),
            eval_protocol=task_config.get("eval_protocol", "default"),
        )
        if bootstrap:
            main, boots, aux = run_binary_zs_with_bootstrap(**kwargs)
            return main, aux, boots
        main, aux = run_binary_zs(**kwargs, verbose=verbose)
        return main, aux, None

    if task_type == "binary_zs_rate":
        main, aux = run_binary_zs_rate(
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            device=device,
            dataset_name=task_config["dataset_name"],
            model_config=model_config,
            modality=task_config.get("modality", "abdomen_chest"),
            split=task_config["split"],
            verbose=verbose,
        )
        return main, aux, None

    if task_type == "localization":
        filter_one_slice_per_scan(dataset)
        if bootstrap:
            main, boots = run_localization_with_bootstrap(
                model=model, dataloader=dataloader, dataset=dataset, device=device
            )
            return main, None, boots
        main, aux = run_localization(
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            device=device,
            verbose=verbose,
        )
        return main, aux, None

    if task_type == "volume_retrieval":
        kwargs = dict(
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            device=device,
            dataset_name=task_config["dataset_name"],
        )
        if bootstrap:
            main, boots = run_volume_retrieval_with_bootstrap(**kwargs)
            return main, None, boots
        main = run_volume_retrieval(**kwargs, verbose=verbose)
        return main, None, None

    if task_type == "pool_retrieval":
        kwargs = dict(
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            device=device,
            model_config=model_config,
            pool_sizes=task_config.get("pool_sizes"),
            ks=task_config.get("ks"),
            repeats=task_config.get("repeats"),
            seed=task_config.get("seed", 42),
            verbose=verbose,
        )
        if bootstrap:
            main, boots = run_pool_retrieval_with_bootstrap(**kwargs)
            return main, None, boots
        main = run_pool_retrieval(**kwargs)
        return main, None, None

    raise RuntimeError(f"unreachable: task_type={task_type!r}")
