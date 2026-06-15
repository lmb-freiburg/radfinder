"""
Check dataloader
"""

from pathlib import Path

from attr import define
from radfinder.cli.train_siglip import print_transform_from_dataset
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
    # overwrite image feat mode
    train_config["train"]["image_feat_mode"] = args.image_feat_mode
    train_config["train"]["text_feat_mode"] = args.text_feat_mode
    # determine run name and set output dir
    train_config["train"]["output_dir"] = "UNUSED"
    random_seed(train_config["train"]["seed"])
    train_config["train"]["workers"] = 0
    log_info("Training config:")
    log_info(train_config)

    #################### Create training dataloader ####################
    dataloader = get_train_dataloader_from_config(
        features_model_name=args.model_cfg.stem,
        text_transform_mode=args.text_transform_mode,
        model_config=model_config,
        train_config=train_config,
    )
    log_info(f"\nDataloader: {len(dataloader.dataset)} samples")
    log_info(f"\nTransform:")
    print_transform_from_dataset(dataloader.dataset)

    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        log_info(f"\nBatch {i}:")
        print(repr_value(batch))


if __name__ == "__main__":
    main()
