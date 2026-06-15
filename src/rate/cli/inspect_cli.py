"""
Pipeline structure inspired by YalaLab/rate@79b23df src/cli.py (rewritten, MIT).

CLI for running RATE report structuring on the Inspect dataset.

Inspect is a CTPA (chest) dataset with ~23k scans.
Reports only have impressions (no findings section).

Stages:
- remove_comparisons_impressions: Remove comparison language from impressions
- map_categories: Map impressions to organ categories using chest_ct config
- questions: Answer binary questions per report using impression text
"""

from pathlib import Path
from typing import Optional

from attrs import define
from loguru import logger
from radfinder.paths import RATE_CONFIG_DIR
from rate.engine_from_data import (
    NO_COMPARISONS_IMPRESSIONS_FILE,
    run_categories_stage,
    run_questions_stage,
    run_remove_comparisons_stage,
)
from rate.misc_rate import ok
from rate.rate_common_utils import RateStructuringArgs
from rate.rate_inspect_utils import (
    INSPECT_DEFAULT_DATA_DIR,
    INSPECT_DEFAULT_OUTPUT_DIR,
    InspectRateOutputLoader,
    build_reports_inspect,
)

from packg.iotools.yamlext import load_yaml
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument


@define
class Args(VerboseQuietArgs, RateStructuringArgs):
    base_dir: Optional[Path] = add_argument(help="Source dir", default=INSPECT_DEFAULT_DATA_DIR)
    save_dir: str = add_argument(help="Saved results to check", default=INSPECT_DEFAULT_OUTPUT_DIR)
    stage: str = add_argument(
        choices=["remove_comparisons_impressions", "map_categories", "questions"],
        default="map_categories",
    )
    split: str = add_argument(help="train, valid, test, trainval, all", default="valid")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    # Inspect is all chest / en
    bodypart = "chest"
    language = "en"
    save_dir = Path(args.save_dir)

    # ---------- load config
    config_file = args.config
    if config_file is None:
        config_file = RATE_CONFIG_DIR / f"default_config_{language}.yaml"
    config = load_yaml(config_file)

    config["model"] = {
        "name": args.model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    if args.host:
        config["server"]["base_url"] = args.host
    if args.port:
        config["server"]["port"] = args.port
    if args.autohost:
        config["server"]["autohost"] = args.autohost
        print(f"Using {args.autohost= } configuration which will override host/port settings")

    modality_config_file = (
        RATE_CONFIG_DIR / f"modalities_{language}" / f"{bodypart}_ct.yaml"
    ).as_posix()
    config_dict = {
        "modality": bodypart,
        "modality-config": modality_config_file,
        "processing": {
            "save-dir": save_dir.as_posix(),
            "num-workers": args.num_workers,
        },
        "check_only": args.check,
    }
    config.update(config_dict)

    # ---------- load input data
    print("Loading input data...")
    loader = InspectRateOutputLoader(args.save_dir, args.base_dir, verbose=True)

    split_arg = args.split
    if split_arg == "trainval":
        splits = ["valid", "train"]
    elif split_arg == "all":
        splits = ["valid", "train", "test"]
    else:
        splits = [split_arg]
    print(f"Running splits: {splits}")

    for split in splits:
        rdf, no_comp_imprs_dict, catdict, quesdict, mod_cfg = loader.load_data(
            language, bodypart, split
        )

        impressions_dict = {k: v for k, v in rdf["impressions"].to_dict().items() if ok(v)}

        if args.stage == "remove_comparisons_impressions":
            print(
                f"Using {len(impressions_dict)} impressions.\n"
                f"Example: {next(iter(impressions_dict.items()))}\n"
            )
            no_comp_imprs_dict = run_remove_comparisons_stage(
                save_dir,
                mod_cfg,
                config,
                impressions_dict,
                no_comp_imprs_dict,
                bodypart,
                split,
                NO_COMPARISONS_IMPRESSIONS_FILE,
                "no_comparison_impressions",
                "remove_comparisons_impressions",
                chunk_size=args.chunk_size,
            )
        elif args.stage == "map_categories":
            # Inspect has no findings — use impressions for category mapping
            if no_comp_imprs_dict:
                text_for_cats = no_comp_imprs_dict
                print(f"Using {len(text_for_cats)} impressions with removed comparisons.")
            else:
                text_for_cats = impressions_dict
                print(f"Using {len(text_for_cats)} raw impressions (no comparison removal).")
            print(f"Example: {next(iter(text_for_cats.items()))}\n")
            catdict = run_categories_stage(
                save_dir,
                mod_cfg,
                config,
                text_for_cats,
                catdict,
                bodypart,
                split,
                chunk_size=args.chunk_size,
            )
        elif args.stage == "questions":
            reports_dict = build_reports_inspect(rdf)
            print(
                f"Using {len(reports_dict)} total reports.\n"
                f"Example: {next(iter(reports_dict.items()))}\n"
            )
            quesdict = run_questions_stage(
                save_dir,
                mod_cfg,
                config,
                reports_dict,
                quesdict,
                bodypart,
                split,
                chunk_size=args.chunk_size,
            )


if __name__ == "__main__":
    main()
