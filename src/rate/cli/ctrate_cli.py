"""
Pipeline structure inspired by YalaLab/rate@79b23df src/cli.py (rewritten, MIT).

Run the RATE processing of the CT-RATE dataset.

CT-RATE is a chest dataset.
Reports only have findings and impressions.

Stages:
- remove_comparisons_findings: Remove comparison language from findings
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
    NO_COMPARISONS_FINDINGS_FILE,
    NO_COMPARISONS_IMPRESSIONS_FILE,
    run_categories_stage,
    run_questions_stage,
    run_remove_comparisons_stage,
)
from rate.misc_rate import ok
from rate.rate_common_utils import RateStructuringArgs
from rate.rate_ctrate_utils import (
    CTRATE_DEFAULT_DATA_DIR,
    CTRATE_DEFAULT_OUTPUT_DIR,
    CTRateRateOutputLoader,
    build_reports_ctrate,
)

from packg.iotools.yamlext import load_yaml
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument


@define
class Args(VerboseQuietArgs, RateStructuringArgs):
    base_dir: Optional[Path] = add_argument(help="Source base dir", default=CTRATE_DEFAULT_DATA_DIR)
    save_dir: str = add_argument(help="Saved results to check", default=CTRATE_DEFAULT_OUTPUT_DIR)
    stage: str = add_argument(
        choices=[
            "remove_comparisons_findings",
            "remove_comparisons_impressions",
            "map_categories",
            "questions",
        ],
        default="remove_comparisons_findings",
    )
    split: str = add_argument(help="train, val, test, trainval, all", default="val")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    # CT-RATE is all chest / en
    bodypart = "chest"
    language = "en"
    save_dir = Path(args.save_dir)

    # ---------- load config
    config_file = args.config
    if config_file is None:
        config_file = RATE_CONFIG_DIR / f"default_config_{language}.yaml"
    config = load_yaml(config_file)

    # Set model config from args
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
    # Construct the complete config dictionary first
    modality_config_file = (
        RATE_CONFIG_DIR / f"modalities_{language}" / f"{bodypart}_ct.yaml"
    ).as_posix()
    config_dict = {
        # "input-files": args.input_files or [],
        "modality": bodypart,  # save outputs per bodypart
        "modality-config": modality_config_file,
        "processing": {
            "save-dir": save_dir.as_posix(),
            "num-workers": args.num_workers,
        },
        "check_only": args.check,
    }
    # Merge with default config
    config.update(config_dict)

    print(f"Loading input data...")
    loader = CTRateRateOutputLoader(args.save_dir, args.base_dir, verbose=True)

    split_arg = args.split
    if split_arg == "trainval":
        splits = ["valid", "train"]
    elif split_arg == "all":
        splits = ["valid", "train"]
    else:
        splits = [split_arg]
    print(f"Running splits : {splits}")
    for split in splits:
        if split == "val":
            split = "valid"

        rdf, no_comp_finds_dict, no_comp_imprs_dict, catdict, quesdict, mod_cfg = loader.load_data(
            language, bodypart, split
        )

        findings_with_history = rdf[f"Findings_EN"]
        findings_with_history_dict = {
            k: v for k, v in findings_with_history.to_dict().items() if ok(v)
        }

        imprs_with_history = rdf[f"Impressions_EN"]
        imprs_with_history_dict = {k: v for k, v in imprs_with_history.to_dict().items() if ok(v)}

        reports_dict = build_reports_ctrate(rdf)

        # ---------- Processing
        # remove-comparisons: necessary for CT-RATE
        # extract-findings: is not needed because in our data the findings are a separate field
        # this leaves us with: "map-categories", "process-questions"
        if args.stage == "remove_comparisons_findings":
            print(
                f"Using {len(findings_with_history_dict)} findings with history.\n"
                f"Example: {next(iter(findings_with_history_dict.items()))}\n"
            )
            no_comp_finds_dict = run_remove_comparisons_stage(
                save_dir,
                mod_cfg,  # modality specific config
                config,  # global config
                findings_with_history_dict,  # reports to process
                no_comp_finds_dict,  # existing results
                bodypart,
                split,
                NO_COMPARISONS_FINDINGS_FILE,
                "no_comparison_findings",
                "remove_comparisons_findings",
                chunk_size=args.chunk_size,
            )
        elif args.stage == "remove_comparisons_impressions":
            print(
                f"Using {len(imprs_with_history_dict)} impressions with history.\n"
                f"Example: {next(iter(imprs_with_history_dict.items()))}\n"
            )
            no_comp_imprs_dict = run_remove_comparisons_stage(
                save_dir,
                mod_cfg,  # modality specific config
                config,  # global config
                imprs_with_history_dict,  # reports to process
                no_comp_imprs_dict,  # existing results
                bodypart,
                split,
                NO_COMPARISONS_IMPRESSIONS_FILE,
                "no_comparison_impressions",
                "remove_comparisons_impressions",
                chunk_size=args.chunk_size,
            )
        elif args.stage == "map_categories":
            print(
                f"Using {len(no_comp_finds_dict)} findings with removed history.\n"
                f"Example: {next(iter(no_comp_finds_dict.items()))}\n"
            )
            catdict = run_categories_stage(
                save_dir,
                mod_cfg,  # modality specific config
                config,  # global config
                no_comp_finds_dict,  # reports to process
                catdict,  # existing results
                bodypart,
                split,
                chunk_size=args.chunk_size,
            )
        elif args.stage == "questions":
            print(
                f"Using {len(reports_dict)} total reports.\n"
                f"Example: {next(iter(reports_dict.items()))}\n"
            )

            quesdict = run_questions_stage(
                save_dir,
                mod_cfg,  # modality specific config
                config,  # global config
                reports_dict,  # reports to process
                quesdict,  # existing results
                bodypart,
                split,
                chunk_size=args.chunk_size,
            )


if __name__ == "__main__":
    main()
