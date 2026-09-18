import re
from pathlib import Path
from typing import Optional

import pandas as pd
from attrs import define
from radfinder.paths import RATE_CONFIG_DIR
from rate.engine_from_data import (
    DEFAULT_CHUNK_SIZE,
    MAP_CATEGORIES_FILE,
    NO_COMPARISONS_FINDINGS_FILE,
    NO_COMPARISONS_IMPRESSIONS_FILE,
    QUESTIONS_FILE,
)

from packg.iotools.yamlext import load_yaml
from typedparser import add_argument

RE_MULTI_SPACE = re.compile(r"\s+")
RE_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@define
class RateStructuringArgs:
    """
    If the LLM gets stuck on a few of the outputs with the default arguments try setting
    p=0.9 temp=0.7
    """

    num_workers: int = add_argument(
        help="Number of worker threads for processing",
        default=256,
        type=int,
    )
    config: Optional[str] = add_argument(help="Path to configuration file", default=None)
    host: Optional[str] = add_argument(help="Overwrite config host", default=None)
    port: Optional[int] = add_argument(help="Overwrite config port", default=None, type=int)
    autohost: Optional[str] = add_argument(
        help="Automatically read host and port from ~/llmhost/<autohostname>_hostname_* files",
        default=None,
    )
    model_name: str = add_argument(
        help="Model name to use",
        default="Qwen/Qwen3-30B-A3B-FP8",
    )
    temperature: float = add_argument(
        help="Temperature for model inference",
        default=0.1,
        type=float,
    )
    top_p: float = add_argument(
        help="Top-p sampling parameter",
        default=0.1,
        type=float,
    )
    max_tokens: int = add_argument(
        help="Maximum tokens for model output (reasoning + output)",
        default=4096,
        type=int,
    )
    check: bool = add_argument(
        action="store_true", help="Check how much work remains without actually processing anything"
    )
    chunk_size: int = add_argument(
        help="Number of LLM inputs to process between saving", default=DEFAULT_CHUNK_SIZE, type=int
    )


def _normalize_text_key(text: str) -> str:
    text = text.strip().lower()
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_NON_ALNUM.sub("", text)
    return text


def _strip_location_suffixes(category: str) -> str:
    # Remove explicit location suffixes to support category renames like
    # "X in the thorax" / "Y in the abdomen" and German equivalents.
    patterns = [
        r"\s+in the (thorax|chest|abdomen|pelvis)\s*$",
        r"\s+in (the )?visible (neck|abdomen)\s*$",
        r"\s+im (thorax|abdomen|bauch|becken)\s*$",
        r"\s+im sichtbaren (hals|abdomen)\s*$",
    ]
    out = category.strip()
    for pattern in patterns:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE).strip()
    return out


def _match_category_to_expected(
    source_category: str,
    expected_categories: list[str],
) -> tuple[str, str]:
    if source_category in expected_categories:
        return source_category, "exact"

    expected_set = set(expected_categories)
    source_stripped = _strip_location_suffixes(source_category)
    if source_stripped in expected_set:
        return source_stripped, "strip_suffix_source"

    stripped_to_expected: dict[str, list[str]] = {}
    for exp in expected_categories:
        stripped_to_expected.setdefault(_strip_location_suffixes(exp), []).append(exp)

    stripped_candidates = stripped_to_expected.get(source_stripped, [])
    if len(stripped_candidates) == 1:
        return stripped_candidates[0], "strip_suffix_expected"

    norm_source = _normalize_text_key(source_stripped)
    norm_map: dict[str, list[str]] = {}
    for exp in expected_categories:
        norm_map.setdefault(_normalize_text_key(_strip_location_suffixes(exp)), []).append(exp)
    norm_candidates = norm_map.get(norm_source, [])
    if len(norm_candidates) == 1:
        return norm_candidates[0], "normalized"

    return source_category, "unmatched"


def _build_expected_questions(categories: dict) -> dict[str, list[str]]:
    expected_questions = {}
    for category, category_config in categories.items():
        expected_questions[category] = [q["question"] for q in category_config["questions"]]
    return expected_questions


def _build_question_to_categories(expected_questions: dict[str, list[str]]) -> dict[str, set[str]]:
    q2cat: dict[str, set[str]] = {}
    for category, questions in expected_questions.items():
        for question in questions:
            q2cat.setdefault(question, set()).add(category)
    return q2cat


def _resolve_question_category(
    source_category: str,
    question: str,
    expected_categories: list[str],
    expected_questions: dict[str, list[str]],
    question_to_categories: dict[str, set[str]],
) -> tuple[str, str]:
    if source_category in expected_questions and question in expected_questions[source_category]:
        return source_category, "exact"

    candidates = question_to_categories.get(question, set())
    if len(candidates) == 1:
        return next(iter(candidates)), "by_question_unique"
    if len(candidates) > 1:
        mapped_cat, mapped_reason = _match_category_to_expected(source_category, sorted(candidates))
        if mapped_reason != "unmatched" and mapped_cat in candidates:
            return mapped_cat, f"by_question_ambiguous_{mapped_reason}"

    mapped_cat, mapped_reason = _match_category_to_expected(source_category, expected_categories)
    if mapped_reason != "unmatched" and question in expected_questions.get(mapped_cat, []):
        return mapped_cat, f"by_category_{mapped_reason}"

    return source_category, "unmatched"


def resolve_question_category_compat(
    source_category: str,
    question: str,
    categories: dict,
) -> tuple[str, str]:
    expected_categories = list(categories.keys())
    expected_questions = _build_expected_questions(categories)
    question_to_categories = _build_question_to_categories(expected_questions)
    return _resolve_question_category(
        source_category=source_category,
        question=question,
        expected_categories=expected_categories,
        expected_questions=expected_questions,
        question_to_categories=question_to_categories,
    )


def resolve_category_compat(
    source_category: str,
    categories: dict,
) -> tuple[str, str]:
    expected_categories = list(categories.keys())
    return _match_category_to_expected(source_category, expected_categories)


def load_outputs(
    save_dir_lang: Path,
    language: str,
    bodypart: str,
    split: str,
    reports_index: list[str],
    verbose: bool = True,
    do_validate_findings: bool = True,
    do_validate_impressions: bool = True,
    allow_category_compat_remap: bool = True,
):
    reports_index_set = set(reports_index)
    # load config of what to expect
    modality_config_file = RATE_CONFIG_DIR / f"modalities_{language}" / f"{bodypart}_ct.yaml"
    mod_cfg = load_yaml(modality_config_file)
    categories = mod_cfg["categories"]

    # load outputs for this language and bodypart and split
    # Load no_comparisons_findings
    no_comparisons_findings_file = save_dir_lang / NO_COMPARISONS_FINDINGS_FILE.format(
        bodypart=bodypart, split=split
    )
    if not no_comparisons_findings_file.is_file():
        no_comp_finds_dict = {}
    else:
        no_comparisons_findings_csv = pd.read_csv(no_comparisons_findings_file)
        for col in no_comparisons_findings_csv.columns:
            n_na = no_comparisons_findings_csv[col].isna().sum()
            if n_na > 0:
                raise ValueError(f"  {n_na:7_d} N/A in no_comparisons_findings csv column {col}")
        no_comp_finds_dict = unstack_no_comparisons(
            no_comparisons_findings_csv, reports_index_set, "no_comparison_findings"
        )

    # Load no_comparisons_impressions
    no_comp_imprs_file = save_dir_lang / NO_COMPARISONS_IMPRESSIONS_FILE.format(
        bodypart=bodypart, split=split
    )
    if not no_comp_imprs_file.is_file():
        no_comp_imprs_dict = {}
    else:
        no_comp_imprs_csv = pd.read_csv(no_comp_imprs_file)
        print(f"Loaded {no_comp_imprs_file} - {len(no_comp_imprs_csv)} rows")

        n_rows_with_nas = no_comp_imprs_csv.isna().any(axis=1).sum()
        if n_rows_with_nas > 0:
            print(f"Warning: {n_rows_with_nas=}. This is fine if input data also has NaNs.")
        # # instead skip all rows that have a na
        # len_before = len(no_comp_imprs_csv)
        # no_comp_imprs_csv = no_comp_imprs_csv.dropna(axis="index", how="any")
        # n_dropped_na = len_before - len(no_comp_imprs_csv)

        no_comp_imprs_dict = unstack_no_comparisons(
            no_comp_imprs_csv, reports_index_set, "no_comparison_impressions"
        )

    # Load categories
    cat_file = save_dir_lang / MAP_CATEGORIES_FILE.format(bodypart=bodypart, split=split)
    if not cat_file.is_file():
        catdict = {}
    else:
        cat_csv = pd.read_csv(cat_file)
        for col in cat_csv.columns:
            n_na = cat_csv[col].isna().sum()
            if n_na > 0:
                raise ValueError(f"  {n_na:7_d} N/A in categories csv column {col}")
        catdict = unstack_categories(
            cat_csv,
            reports_index_set,
            expected_categories=list(categories.keys()),
            do_category_compat_remap=allow_category_compat_remap,
            verbose=verbose,
        )
    ques_file = save_dir_lang / QUESTIONS_FILE.format(bodypart=bodypart, split=split)
    if not ques_file.is_file():
        quesdict = {}
    else:
        ques_csv = pd.read_csv(ques_file)
        for col in ques_csv.columns:
            n_na = ques_csv[col].isna().sum()
            if n_na > 0:
                raise ValueError(f"  {n_na:7_d} N/A in questions csv column {col}")
        quesdict = unstack_questions(
            ques_csv,
            reports_index_set,
            categories=categories,
            do_category_compat_remap=allow_category_compat_remap,
            verbose=verbose,
        )

    if do_validate_findings:
        validate_remove_comparisons(
            reports_index,
            no_comp_finds_dict,
            language,
            bodypart,
            split,
            "no_comp_findings:",
            verbose,
        )

    if do_validate_impressions:
        validate_remove_comparisons(
            reports_index,
            no_comp_imprs_dict,
            language,
            bodypart,
            split,
            "no_comp_impress:",
            verbose,
        )

    validate_categories(
        reports_index,
        catdict,
        categories,
        language,
        bodypart,
        split,
        verbose,
        strict_category_match=not allow_category_compat_remap,
    )

    validate_questions(
        reports_index,
        quesdict,
        categories,
        language,
        bodypart,
        split,
        verbose,
        strict_category_match=not allow_category_compat_remap,
    )
    return no_comp_finds_dict, no_comp_imprs_dict, catdict, quesdict, mod_cfg


def validate_categories(
    report_index: list[str],
    catdict,
    categories,
    language: str,
    bodypart: str,
    split: str,
    verbose: bool = True,
    strict_category_match: bool = True,
):
    expected_categories = list(categories.keys())
    # validate outputs and count outputs of categories
    reports_incomplete_cat, reports_complete_cat = [], []
    n_missing_categories_total = 0
    for reportrserid in report_index:
        if reportrserid not in catdict:
            reports_incomplete_cat.append(reportrserid)
            n_missing_categories_total += len(expected_categories)
            continue
        reportdata = catdict[reportrserid]
        found_cats = set(reportdata.keys())
        unexpected_cats = found_cats - set(expected_categories)
        if strict_category_match and unexpected_cats:
            raise AssertionError(
                f"Unexpected categories: {reportrserid} {bodypart} {language}: {unexpected_cats}"
            )
        missing_cats = set(expected_categories) - found_cats
        n_missing_categories = len(missing_cats)

        if n_missing_categories > 0:
            reports_incomplete_cat.append(reportrserid)
        else:
            reports_complete_cat.append(reportrserid)
        n_missing_categories_total += n_missing_categories
    complete_cat_idx = set(reports_complete_cat)
    n_total_categories = len(report_index) * len(expected_categories)
    n_complete_categories_total = n_total_categories - n_missing_categories_total
    n_done = len(set(report_index) & complete_cat_idx)
    n_reports = len(report_index)
    if verbose:
        print(
            f"{language} {bodypart:15s} {split:5s} categories:       "
            f"{n_done:7_d} / {n_reports:7_d} ({n_done/n_reports:7.2%}) "
            f"requests: {n_complete_categories_total:10_d} / {n_total_categories:10_d} "
            f"({n_complete_categories_total/n_total_categories:7.2%})"
        )


def validate_questions(
    report_index: list[str],
    quesdict,
    categories,
    language: str,
    bodypart: str,
    split: str,
    verbose: bool = True,
    strict_category_match: bool = True,
):
    """validate outputs and counts outputs of questions"""
    expected_categories = list(categories.keys())
    expected_questions = {}
    for category, category_config in categories.items():
        expected_questions[category] = [q["question"] for q in category_config["questions"]]
    n_reports = len(report_index)
    reports_incomplete, reports_complete = [], []
    n_missing_questions_total = 0
    for reportrserid in report_index:
        n_missing_questions = 0

        if reportrserid not in quesdict:
            # report is missing completely, all questions for all categories are missing
            reports_incomplete.append(reportrserid)
            for cat in expected_categories:
                n_missing_questions += len(expected_questions[cat])
            n_missing_questions_total += n_missing_questions
            continue

        reportdata = quesdict[reportrserid]
        n_missing_questions = 0
        found_cats = set(reportdata.keys())
        unexpected_cats = found_cats - set(expected_categories)
        if strict_category_match and unexpected_cats:
            raise AssertionError(
                f"Unexpected categories: {reportrserid} {bodypart} {language}: {unexpected_cats}"
            )
        missing_cats = set(expected_categories) - found_cats
        for cat in missing_cats:
            # entire category is missing, all questions for that category are missing
            n_missing_questions += len(expected_questions[cat])

        for category, qadata in reportdata.items():
            if category not in expected_questions:
                # Unknown category (e.g., legacy naming) is ignored in lenient mode.
                continue
            found_questions = set(qadata.keys())
            assert found_questions.issubset(
                set(expected_questions[category])
            ), f"Unexpected {reportrserid} {bodypart} {language} {category}: {found_questions=}"
            # only a few questions for this category are missing
            missing_questions = set(expected_questions[category]) - found_questions
            n_missing_questions += len(missing_questions)

        if n_missing_questions > 0:
            reports_incomplete.append(reportrserid)
        else:
            reports_complete.append(reportrserid)
        n_missing_questions_total += n_missing_questions
    complete_idx = set(reports_complete)
    n_total_questions_per_report = sum(len(qs) for qs in expected_questions.values())
    n_total_questions = n_reports * n_total_questions_per_report
    n_complete_questions_total = n_total_questions - n_missing_questions_total
    n_done = len(set(report_index) & complete_idx)
    n_reports = len(report_index)
    if verbose:
        print(
            f"{language} {bodypart:15s} {split:5s} questions:        "
            f"{n_done:7_d} / {n_reports:7_d} ({n_done/n_reports:7.2%}) "
            f"requests: {n_complete_questions_total:10_d} / {n_total_questions:10_d} "
            f"({n_complete_questions_total/n_total_questions:7.2%})"
        )


def validate_remove_comparisons(
    report_index: list[str],
    results_dict,
    language: str,
    bodypart: str,
    split: str,
    stage_label: str,
    verbose: bool = True,
):
    """Validate no_comparisons outputs and count completion.

    Args:
        report_index: List of report IDs to validate
        results_dict: Dictionary of report_id -> processed text
        language: Language identifier
        bodypart: Body part identifier
        split: Data split
        stage_label: Label for display (e.g., "no_comp_findings", "no_comp_impress")
        verbose: Whether to print statistics
    """
    n_reports = len(report_index)
    n_complete = 0
    reports_complete = []
    reports_incomplete = []

    for reportrserid in report_index:
        if reportrserid in results_dict:
            n_complete += 1
            reports_complete.append(reportrserid)
        else:
            reports_incomplete.append(reportrserid)

    if verbose:
        print(
            f"{language} {bodypart:15s} {split:5s} {stage_label:17s} "
            f"{n_complete:7_d} / {n_reports:7_d} ({n_complete/n_reports:7.2%})"
        )


def unstack_no_comparisons(data_df, reports_index_set, column_name: str):
    """Generic function to unstack no_comparisons data from CSV to dict.

    Args:
        data_df: DataFrame with 'report_id' and a data column
        reports_index_set: Set of report IDs to include
        column_name: Name of the column containing the text data

    Input CSV:
        report_id                         <column_name>
        c038c5353716b60047b2d1407878a5a9  The pancreas appears normal. The liver...
        d123e4567890f12345678901234567890  No acute findings in the chest...
    ...

    Output dict:
        {
            "c038c5353716b60047b2d1407878a5a9": "The pancreas appears normal. The liver...",
            "d123e4567890f12345678901234567890": "No acute findings in the chest...",
            ...
        }
    """
    result = {}
    for report_id, text_data in zip(data_df["report_id"], data_df[column_name]):
        if report_id not in reports_index_set:
            continue
        result[report_id] = text_data
    return result


def unstack_categories(
    cat,
    reports_index_set,
    expected_categories: Optional[list[str]] = None,
    do_category_compat_remap: bool = False,
    verbose: bool = True,
):
    """
    Input CSV:
        report_id                         category       findings
        c038c5353716b60047b2d1407878a5a9  Pancreas       Pancreas normally lobulated with ...
        c038c5353716b60047b2d1407878a5a9  Adrenal gland  Adrenal glands bilaterally slender.
    ...

    Output dict:
        {
            "c038c5353716b60047b2d1407878a5a9": {
                "Pancreas": "Pancreas normally lobulated with ...",
                "Adrenal gland": "Adrenal glands bilaterally slender.",
                ...
            },
            ...
        }

    """
    result = {}
    remap_reason_counts: dict[str, int] = {}
    n_rows_kept = 0
    for report_id, category, findings in zip(cat["report_id"], cat["category"], cat["findings"]):
        if report_id not in reports_index_set:
            continue
        n_rows_kept += 1
        category_out = category
        if do_category_compat_remap and expected_categories is not None:
            category_out, remap_reason = _match_category_to_expected(category, expected_categories)
            remap_reason_counts[remap_reason] = remap_reason_counts.get(remap_reason, 0) + 1
        if report_id not in result:
            result[report_id] = {}
        result[report_id][category_out] = findings
    if do_category_compat_remap and verbose:
        print(
            f"unstack_categories remap stats: rows={n_rows_kept} " f"reasons={remap_reason_counts}"
        )
    return result


def unstack_questions(
    ques,
    reports_index_set,
    categories: Optional[dict] = None,
    do_category_compat_remap: bool = False,
    verbose: bool = True,
):
    """
    (question_id is optional and will be dropped here)
    Input CSV:
        report_id                         category  question_id  question                    answer
        c038c5353716b60047b2d1407878a5a9  Pancreas  58c9dc9d     Is chronic pancreatitis...  No
        c038c5353716b60047b2d1407878a5a9  Pancreas  f7b31c40     ...pancreatic atrophy?      No
    ...

    Output dict:
        {
            "c038c5353716b60047b2d1407878a5a9": {
                "Pancreas": {
                    'Is chronic pancreatitis suspected based on imaging findings?': 'No',
                    'Is there pancreatic atrophy?': 'No',
                    ...
                }
                ...
            }
            ...
        }
    """
    result = {}
    remap_reason_counts: dict[str, int] = {}
    n_rows_kept = 0
    if categories is not None:
        expected_categories = list(categories.keys())
        expected_questions = _build_expected_questions(categories)
        question_to_categories = _build_question_to_categories(expected_questions)
    else:
        expected_categories = []
        expected_questions = {}
        question_to_categories = {}

    for report_id, category, question, answer in zip(
        ques["report_id"], ques["category"], ques["question"], ques["answer"]
    ):
        if report_id not in reports_index_set:
            continue
        n_rows_kept += 1
        category_out = category
        if do_category_compat_remap and categories is not None:
            category_out, remap_reason = _resolve_question_category(
                source_category=category,
                question=question,
                expected_categories=expected_categories,
                expected_questions=expected_questions,
                question_to_categories=question_to_categories,
            )
            remap_reason_counts[remap_reason] = remap_reason_counts.get(remap_reason, 0) + 1
        if report_id not in result:
            result[report_id] = {}
        if category_out not in result[report_id]:
            result[report_id][category_out] = {}
        result[report_id][category_out][question] = answer
    if do_category_compat_remap and verbose:
        print(
            f"unstack_questions remap stats: rows={n_rows_kept} " f"reasons={remap_reason_counts}"
        )
    return result
