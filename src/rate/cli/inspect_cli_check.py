"""
Utility to check progress of the RATE processing of the Inspect dataset.
"""

from pathlib import Path
from typing import Optional

from attrs import define
from radfinder.paths import RATE_CONFIG_DIR
from rate.rate_common_utils import RateStructuringArgs
from rate.rate_inspect_utils import (
    INSPECT_DEFAULT_DATA_DIR,
    INSPECT_DEFAULT_OUTPUT_DIR,
    InspectRateOutputLoader,
)

from packg.iotools.yamlext import load_yaml
from typedparser import TypedParser, VerboseQuietArgs, add_argument


@define
class Args(VerboseQuietArgs, RateStructuringArgs):
    base_dir: Optional[Path] = add_argument(help="Source dir", default=INSPECT_DEFAULT_DATA_DIR)
    save_dir: str = add_argument(help="Saved results to check", default=INSPECT_DEFAULT_OUTPUT_DIR)


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    loader = InspectRateOutputLoader(args.save_dir, args.base_dir, verbose=True)

    # Inspect is all chest / en
    language = "en"
    bodypart = "chest"

    # Check for duplicate questions in config
    modality = load_yaml(RATE_CONFIG_DIR / f"modalities_{language}" / f"{bodypart}_ct.yaml")
    for cat, catdata in modality["categories"].items():
        questions_here = set()
        for questionitem in catdata["questions"]:
            question = questionitem["question"]
            if question in questions_here:
                print(f"Duplicate question found: {cat} {question}")
            questions_here.add(question)

    # Load and validate all splits
    for split in ["train", "valid", "test"]:
        loader.load_data(language, bodypart, split)


if __name__ == "__main__":
    main()
