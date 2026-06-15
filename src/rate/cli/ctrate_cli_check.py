"""
Utility to check progress of the RATE processing of the CT-RATE dataset.
"""

from pathlib import Path
from typing import Optional

from attrs import define
from loguru import logger
from rate.rate_common_utils import RateStructuringArgs
from rate.rate_ctrate_utils import (
    CTRATE_DEFAULT_DATA_DIR,
    CTRATE_DEFAULT_OUTPUT_DIR,
    CTRateRateOutputLoader,
    build_reports_ctrate,
)

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument


@define
class Args(VerboseQuietArgs, RateStructuringArgs):
    base_dir: Optional[Path] = add_argument(help="Source base dir", default=CTRATE_DEFAULT_DATA_DIR)
    save_dir: str = add_argument(help="Saved results to check", default=CTRATE_DEFAULT_OUTPUT_DIR)


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    # CT-RATE is all chest / en
    bodypart = "chest"
    language = "en"
    loader = CTRateRateOutputLoader(args.save_dir, args.base_dir, verbose=True)

    splits = ["train", "valid"]
    split_outputs = {}
    for split in splits:
        rdf, no_comp_finds_dict, no_comp_imprs_dict, catdict, quesdict, mod_cfg = loader.load_data(
            language, bodypart, split
        )
        reports_dict = build_reports_ctrate(rdf)
        split_outputs[split] = (
            rdf,
            no_comp_finds_dict,
            no_comp_imprs_dict,
            catdict,
            quesdict,
            mod_cfg,
            reports_dict,
        )


if __name__ == "__main__":
    main()
