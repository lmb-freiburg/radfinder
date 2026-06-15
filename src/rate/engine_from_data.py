"""
Pipeline structure inspired by YalaLab/rate@79b23df src/core/engine.py (rewritten, MIT).

Simpler version of the engine with pure functions
"""

import os
from copy import deepcopy
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger

from packg.tqdmext import tqdm_max_ncols

from .batch_processor_fix_sglang import BatchProcessorFixSglang
from .validators_with_reasoning import ResultValidatorWithReasoning

NO_COMPARISONS_FINDINGS_FILE = "no_comparisons_findings_{bodypart}_{split}.csv"
NO_COMPARISONS_IMPRESSIONS_FILE = "no_comparisons_impressions_{bodypart}_{split}.csv"
FINDINGS_FILE = "findings_{bodypart}_{split}.csv"
MAP_CATEGORIES_FILE = "category_findings_{bodypart}_{split}.csv"
QUESTIONS_FILE = "questions_{bodypart}_{split}.csv"
DEFAULT_CHUNK_SIZE = 10240

#################### remove comparisons stages ####################


def build_remove_comparisons_requests(
    reports_data: Dict[str, str], prompt_template: str
) -> List[Tuple[str, str]]:
    """Build requests for remove-comparisons stage."""
    requests = []
    for report_id, text in reports_data.items():
        prompt = prompt_template.format(report=text)
        requests.append((report_id, prompt))
    return requests


def run_remove_comparisons_stage(
    save_dir,
    modality_config,  # modality specific config (not used in this stage)
    config,  # global config
    input_dict,  # text data to process (findings or impressions)
    existing_results_dict,  # existing results
    bodypart,
    split,
    output_file_template,  # e.g., NO_COMPARISONS_FINDINGS_FILE
    output_column_name,  # e.g., "no_comparison_findings"
    stage_name,  # e.g., "remove_comparisons_findings" for progress bar
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    """Generic function to remove comparisons from any text input.

    Args:
        save_dir: Directory to save results
        modality_config: Modality specific config (not used in this stage)
        config: Global config
        input_dict: Dictionary of report_id -> text to process
        existing_results_dict: Dictionary of report_id -> already processed text
        bodypart: Body part identifier
        split: Data split (train/val/test)
        output_file_template: File template string with {bodypart} and {split} placeholders
        output_column_name: Name of the column in output CSV
        stage_name: Name for progress bar and logging
    """
    os.makedirs(save_dir, exist_ok=True)

    # build requests for remove comparisons stage
    prompt_template = config["prompt_templates"]["remove_comparison"]
    requests = build_remove_comparisons_requests(input_dict, prompt_template)

    # filter out already existing results
    requests_todo = []
    for report_id, prompt in requests:
        if report_id not in existing_results_dict:
            requests_todo.append((report_id, prompt))
    print(f"{stage_name}: {len(requests_todo)} requests to process out of {len(requests)}")
    if len(requests_todo) == 0:
        print(f"All {stage_name} already processed, skipping stage")
        return deepcopy(existing_results_dict)

    # Process requests in batches
    validator = ResultValidatorWithReasoning()
    batch_processor = BatchProcessorFixSglang(config, validator)
    target_filename = save_dir / output_file_template.format(bodypart=bodypart, split=split)
    all_results = deepcopy(existing_results_dict)

    pbar = tqdm_max_ncols(desc=stage_name, total=len(requests_todo), smoothing=0)
    pos = 0
    while True:
        chunk = requests_todo[pos : pos + chunk_size]
        if len(chunk) == 0:
            break
        pos += len(chunk)

        # get results as dict of report_id -> (processed_text, reasoning_content)
        res_ok, res_fail = batch_processor.process_batch(chunk, "no_comparisons")
        pbar.update(len(res_ok))

        # add the failed results to the bottom of the todo list
        redo = []
        for report_id, prompt in chunk:
            if report_id in res_fail:
                redo.append((report_id, prompt))
        if len(redo) > 0:
            requests_todo.extend(redo)
            pbar.write(f"Re-queued {len(redo)} failed requests")

        # process successful results
        if len(res_ok) == 0:
            logger.warning(f"No valid results from position {pos}/{len(requests_todo)}")
            continue

        # update all_results
        for report_id, (processed_text, reasoning_content) in res_ok.items():
            assert report_id not in all_results, f"Duplicate result for report {report_id}"
            all_results[report_id] = processed_text

        # convert it to table
        records = []
        for report_id, processed_text in all_results.items():
            records.append({"report_id": report_id, output_column_name: processed_text})
        all_results_df = pd.DataFrame(records)
        all_results_df.to_csv(target_filename, index=False)
        print(f"Saved intermediate results shape {all_results_df.shape} to {target_filename}")
    print(f"Successfully completed {stage_name} stage with {len(all_results)} reports")
    return all_results


#################### map categories stage ####################


def build_map_categories_requests(
    findings_data: Dict[str, str], findings_categories: Dict[str, Dict], prompt_template
) -> Tuple[List[Tuple[str, str]], Dict[str, Tuple[str, str]]]:
    """Build requests for map-categories stage."""
    requests = []
    id_mapping = {}
    for report_id, findings in findings_data.items():
        # Skip empty findings or "No such section" responses
        if not (
            findings
            and findings.strip()
            and findings.strip().lower() not in ["no such section", "no findings section"]
        ):
            raise ValueError(f"Empty or invalid findings for report {report_id}: '{findings}'")
        for category_name, category in findings_categories.items():
            custom_id = f"{report_id}_{category_name}"
            prompt = prompt_template.format(
                findings=findings,
                category=category_name,
                description=category["description"],
            )
            requests.append((custom_id, prompt))
            id_mapping[custom_id] = (report_id, category_name)

    return requests, id_mapping


def process_map_categories_results(
    chunk_results: Dict[str, Tuple[str, str]], id_mapping: Dict[str, Tuple[str, str]]
) -> Dict[str, Dict[str, str]]:
    """Process map-categories results into nested structure."""
    chunk_nested_results = {}
    for custom_id, (content, reasoning_content) in chunk_results.items():
        report_id, category_name = id_mapping[custom_id]
        if report_id not in chunk_nested_results:
            chunk_nested_results[report_id] = {}
        chunk_nested_results[report_id][category_name] = content
    return chunk_nested_results


def run_categories_stage(
    save_dir,
    modality_config,  # modality specific config
    config,  # global config
    findings_dict,  # reports to process
    catdict,  # existing results
    bodypart,
    split,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    os.makedirs(save_dir, exist_ok=True)

    # build requests for category findings stage
    categories = modality_config["categories"]
    prompt_template = config["prompt_templates"]["map_category_findings"]
    requests, id_mapping = build_map_categories_requests(findings_dict, categories, prompt_template)

    # filter out already existing results
    requests_todo = []
    for custom_id, prompt in requests:
        report_id, category_name = id_mapping[custom_id]
        try:
            result = catdict[report_id][category_name]
        except KeyError:
            requests_todo.append((custom_id, prompt))
    print(f"Category mapping: {len(requests_todo)} requests to process out of {len(requests)}")
    if len(requests_todo) == 0:
        print("All category findings already processed, skipping stage")
        return

    # Process requests in batches with improved error handling
    # processor = EngineFromData(config)

    validator = ResultValidatorWithReasoning()
    batch_processor = BatchProcessorFixSglang(config, validator)
    target_filename = save_dir / MAP_CATEGORIES_FILE.format(bodypart=bodypart, split=split)
    all_results = deepcopy(catdict)
    # chunks = [requests_todo[i : i + batch_size] for i in range(0, len(requests_todo), batch_size)]

    pbar = tqdm_max_ncols(desc="map_categories", total=len(requests_todo), smoothing=0)
    pos = 0
    while True:
        chunk = requests_todo[pos : pos + chunk_size]
        if len(chunk) == 0:
            break
        pos += len(chunk)

        # get results as dict of reportrserid_category -> (finding, reasoning_content)
        res_ok, res_fail = batch_processor.process_batch(chunk, "category_findings")
        pbar.update(len(res_ok))

        # add the failed results to the bottom of the todo list
        redo = []
        for custom_id, prompt in chunk:
            if custom_id in res_fail:
                redo.append((custom_id, prompt))
        if len(redo) > 0:
            requests_todo.extend(redo)
            pbar.write(f"Re-queued {len(redo)} failed requests")

        # process successful results
        if len(res_ok) == 0:
            logger.warning(f"No valid results from position {pos}/{len(requests_todo)}")
            continue

        # convert it back to reportrserid -> category -> finding
        chunk_results = process_map_categories_results(res_ok, id_mapping)

        # update all_results
        for reportrserid, reportdata in chunk_results.items():
            if reportrserid not in all_results:
                all_results[reportrserid] = {}
            for category, finding in reportdata.items():
                assert (
                    category not in all_results[reportrserid]
                ), f"Duplicate result for report {reportrserid}, category {category}"
                all_results[reportrserid][category] = finding

        # convert it to table
        records = []
        for report_id, category_findings in all_results.items():
            for category, findings in category_findings.items():
                records.append({"report_id": report_id, "category": category, "findings": findings})
        all_results_df = pd.DataFrame(records)
        all_results_df.to_csv(target_filename, index=False)
        print(f"Saved intermediate results shape {all_results_df.shape} to {target_filename}")
    print(f"Successfully completed category_findings stage with {len(all_results)} reports")
    return all_results


#################### questions stage ####################


def build_questions_requests(
    reports_data: Dict[str, str], categories, prompt_template
) -> Tuple[List[Tuple[str, str]], Dict[str, Tuple[str, str, str]]]:
    """Build requests for process-questions stage."""
    requests = []
    id_mapping = {}
    for report_id, raw_text in reports_data.items():
        for category_name, category in categories.items():
            for question in category["questions"]:
                question_text = question["question"]
                # question_id = hashlib.md5(question_text.encode("utf-8")).hexdigest()[:8]
                request_id = f"{report_id}_{category_name}_{question_text}"
                prompt = prompt_template.format(report=raw_text, question=question_text)
                requests.append((request_id, prompt))
                id_mapping[request_id] = (report_id, category_name, question_text)
    all_ids = [r[0] for r in requests]
    assert len(all_ids) == len(set(all_ids)), "Duplicate request IDs found in questions requests"
    return requests, id_mapping


def process_questions_results(
    chunk_results: Dict[str, Tuple[str, str]], id_mapping: Dict[str, Tuple[str, str, str]]
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Process question results into nested structure."""
    chunk_nested_results = {}
    for request_id, (answer, reasoning_content) in chunk_results.items():
        report_id, category_name, question_text = id_mapping[request_id]
        if report_id not in chunk_nested_results:
            chunk_nested_results[report_id] = {}
        if category_name not in chunk_nested_results[report_id]:
            chunk_nested_results[report_id][category_name] = {}
        chunk_nested_results[report_id][category_name][question_text] = answer
    return chunk_nested_results


def run_questions_stage(
    save_dir,
    modality_config,  # modality specific config
    config,  # global config
    reports_dict,  # reports to process
    quesdict,  # existing results
    bodypart,
    split,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    os.makedirs(save_dir, exist_ok=True)

    # build requests for "questions" stage: reportid_category_questiontext -> prompt
    categories = modality_config["categories"]
    prompt_template = config["prompt_templates"]["findings_question"]
    requests, id_mapping = build_questions_requests(reports_dict, categories, prompt_template)

    # filter out already existing results
    requests_todo = []
    for custom_id, prompt in requests:
        report_id, category_name, question_text = id_mapping[custom_id]
        try:
            _ = quesdict[report_id][category_name][question_text]
        except KeyError:
            requests_todo.append((custom_id, prompt))
    print(f"Questions: {len(requests_todo):_d} requests to process out of {len(requests):_d}")
    if len(requests_todo) == 0:
        print("All questions already processed, skipping stage")
        return deepcopy(quesdict)

    validator = ResultValidatorWithReasoning()
    batch_processor = BatchProcessorFixSglang(config, validator)
    target_filename = save_dir / QUESTIONS_FILE.format(bodypart=bodypart, split=split)
    all_results = deepcopy(quesdict)
    all_results_df = convert_questions_results_to_df(all_results)
    print(f"INPUT: {all_results_df.shape=} from {target_filename}")

    pbar = tqdm_max_ncols(desc="questions", total=len(requests_todo), smoothing=0)
    pos = 0
    while True:
        chunk = requests_todo[pos : pos + chunk_size]
        if len(chunk) == 0:
            break
        pos += len(chunk)

        # get results as dict of reportrserid_category_questiontext -> (finding, reasoning_content)
        res_ok, res_fail = batch_processor.process_batch(chunk, "questions")
        pbar.update(len(res_ok))

        # add the failed results to the bottom of the todo list
        redo = []
        for custom_id, prompt in chunk:
            if custom_id in res_fail:
                redo.append((custom_id, prompt))
        if len(redo) > 0:
            requests_todo.extend(redo)
            pbar.write(f"Re-queued {len(redo)} failed requests")

        # process successful results
        if len(res_ok) == 0:
            logger.warning(f"No valid results from position {pos}/{len(requests_todo)}")
            continue

        # convert it back to reportrserid -> category -> question -> finding
        chunk_results = process_questions_results(res_ok, id_mapping)

        # update all_results
        for reportrserid, reportdata in chunk_results.items():
            if reportrserid not in all_results:
                all_results[reportrserid] = {}
            for category, questions_answers in reportdata.items():
                if category not in all_results[reportrserid]:
                    all_results[reportrserid][category] = {}
                for question_text, answer in questions_answers.items():
                    assert question_text not in all_results[reportrserid][category], (
                        f"Duplicate result for report {reportrserid}, category {category}, "
                        f"question {question_text}"
                    )
                    all_results[reportrserid][category][question_text] = answer

        # convert it to table
        all_results_df = convert_questions_results_to_df(all_results)
        all_results_df.to_csv(target_filename, index=False)
        print(f"Saved intermediate results shape {all_results_df.shape} to {target_filename}")
    print(f"Successfully completed questions stage with {len(all_results)} reports")
    return all_results


def convert_questions_results_to_df(all_results):
    records = []
    for report_id, categories_qa in all_results.items():
        for category, questions_answers in categories_qa.items():
            for question, answer in questions_answers.items():
                records.append(
                    {
                        "report_id": report_id,
                        "category": category,
                        "question": question,
                        "answer": answer,
                    }
                )
    all_results_df = pd.DataFrame(records)
    return all_results_df
