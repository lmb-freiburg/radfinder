import attrs

from attrs import define
from radfinder.data.ct_rate import CTRateFilterMode
from radfinder.transforms.shared_utils import Language

from typedparser import add_argument

LANGUAGE_DEFAULT = Language.EN
CTRATE_FILTER_MODE_DEFAULT = CTRateFilterMode.DUP_ALL


@define
class RetrievalDatasetArgs:
    language: str = add_argument(
        default=LANGUAGE_DEFAULT, help="Language for report generation: en, de, both"
    )
    ctrate_filter_mode: str = add_argument(
        default=CTRATE_FILTER_MODE_DEFAULT,
        help=(
            f"CT-RATE volume filter + per-report dedup. One of: {CTRateFilterMode.values_list()}"
        ),
    )


def retrieval_dataset_args_to_dict(args: RetrievalDatasetArgs) -> dict[str, str]:
    return {field.name: getattr(args, field.name) for field in attrs.fields(RetrievalDatasetArgs)}
