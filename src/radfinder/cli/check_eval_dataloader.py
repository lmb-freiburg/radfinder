"""
Check dataloader
"""

from pathlib import Path

from attr import define
from radfinder.cli.train_siglip import get_val_tasks
from radfinder.data.dataloader_train import get_train_dataloader_from_config
from radfinder.models.load_model import (
    DEFAULT_MODEL_CONFIG_FILE,
    DEFAULT_TRAINING_CONFIG_FILE,
    FeatMode,
)
from radfinder.transforms.train_transform import TextTransformMode
from radfinder.utils.config import load_config_without_types, random_seed
from radfinder.utils.logging_utils import log_info

from packg.log import configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument
from typedparser.objects import repr_value


@define
class Args(VerboseQuietArgs):
    model_cfg: Path = add_argument(default=DEFAULT_MODEL_CONFIG_FILE)
    train_cfg: Path = add_argument(default=DEFAULT_TRAINING_CONFIG_FILE)
    text_transform_mode: str = add_argument(default=TextTransformMode.DEFAULT)
    options: list[str] | None = add_argument(shortcut="-o", action="append")
    image_feat_mode: str = add_argument(default=FeatMode.FROZEN_LOCAL, help="Image feature mode")
    text_feat_mode: str = add_argument(default=FeatMode.FULL, help="Text feature mode")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(get_logger_level_from_args(args))
    log_info(args)

    #################### Load and setup config ####################
    model_config = load_config_without_types(args.model_cfg)
    train_config = load_config_without_types(args.train_cfg, merge_dotlist=args.options)
    # determine run name and set output dir
    train_config["train"]["output_dir"] = "UNUSED"
    random_seed(train_config["train"]["seed"])
    train_config["train"]["workers"] = 0
    log_info("Training config:")
    log_info(train_config)

    #################### Create eval dataloader ####################
    val_tasks_here = train_config["train"]["validation_tasks"]
    test_tasks_here = train_config["train"]["test_tasks"]
    all_test_tasks = val_tasks_here + test_tasks_here

    log_info(f"Evaluating tasks: {all_test_tasks}")
    test_dict_new = get_val_tasks(
        val_tasks=all_test_tasks,
        model_config=model_config,
        model_config_name=args.model_cfg.stem,
        image_feat_mode=args.image_feat_mode,
        text_feat_mode=args.text_feat_mode,
        batch_size=train_config["train"]["batch_size"],
        val_workers=train_config["train"]["val_workers"],
        prefetch_factor=train_config["train"]["prefetch_factor"],
        do_snippet_alignment=None,
        model_settings=train_config["train"].get("model_settings"),
    )

    for task_name, (_task_config, dataloader, dataset) in test_dict_new.items():
        log_info(f"\nTask: {task_name}")
        for i, batch in enumerate(dataloader):
            if i >= 1:
                break
            log_info(f"\nBatch {i}:")
            print(repr_value(batch))
        log_info("*" * 80)


if __name__ == "__main__":
    main()
