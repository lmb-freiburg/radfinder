import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from openai import OpenAI

from packg.tqdmext import tqdm_max_ncols

from .batch_processor import BatchProcessor
from .exceptions import BatchProcessingError, ValidationError
from .logging_config import setup_logging
from .storage import StorageManager
from .validators import ResultValidator


class Engine:
    def __init__(self, config: Union[str, Dict]):
        """Initialize the report processor with configuration."""
        self.config = self._load_config(config)
        self.modality_config_path = self.config["modality-config"]
        self._findings_categories = None
        self._batch_size = self.config.get("processing", {}).get("batch-size", 1024)

        # Get save directory and input filename for logging/storage
        save_dir = self.config.get("processing", {}).get("save-dir", ".")
        input_files = self.config.get("input-files", [])
        input_filename = input_files[0] if input_files else None

        # Setup logging and storage with proper directories
        self.logger = setup_logging("Engine", save_dir=save_dir, filename=input_filename)
        self.storage = StorageManager(save_dir=save_dir)

        # Initialize OpenAI client for sglang
        self.client = OpenAI(
            base_url=f"{self.config['server']['base_url']}:{self.config['server']['port']}/v1",
            api_key="None",
        )

        # Initialize validation and batch processing components
        self.validator = ResultValidator(self.logger)
        self.batch_processor = BatchProcessor(self.client, self.config, self.logger, self.validator)

        # Apply debug subsampling if enabled
        if self.config.get("debug", {}).get("enabled", False):
            self._apply_debug_subsampling()

    def _load_config(self, config: Union[str, Dict]) -> Dict:
        """Load configuration from file or dictionary.

        Args:
            config: Dictionary containing config
        Returns:
            Dictionary containing validated configuration
        """
        config_dict = config

        # Validate required fields
        required_fields = ["input-files", "modality-config", "accession-col", "report-col"]
        missing_fields = [field for field in required_fields if field not in config_dict]
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing_fields)}")

        # Set default values for optional fields
        if "processing" not in config_dict:
            config_dict["processing"] = {}
        if "batch-size" not in config_dict["processing"]:
            config_dict["processing"]["batch-size"] = 1024
        if "save-dir" not in config_dict["processing"]:
            config_dict["processing"]["save-dir"] = "results"

        return config_dict

    def run_remove_comparisons(self, reports: Dict[str, str]) -> Dict[str, str]:
        """Run remove comparisons stage."""
        return self._process_stage(
            stage_name="no_comparisons",
            input_data=reports,
            request_builder=self._build_remove_comparisons_requests,
        )

    def run_extract_findings(self, input_data: Dict[str, str]) -> Dict[str, str]:
        """Run extract findings stage.

        Args:
            input_data: Either list of (report_id, text) tuples or dict of {report_id: no_comparison_text}
        """
        return self._process_stage(
            stage_name="findings",
            input_data=input_data,
            request_builder=self._build_extract_findings_requests,
        )

    def run_map_categories(self, findings_data: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Run map categories stage."""
        return self._process_stage(
            stage_name="category_findings",
            input_data=findings_data,
            request_builder=self._build_map_categories_requests,
            result_processor=self._process_category_results,
        )

    def run_process_questions(self, reports: Dict[str, str]) -> Dict[str, Dict]:
        """Run process questions stage."""
        return self._process_stage(
            stage_name="questions",
            input_data=reports,
            request_builder=self._build_process_questions_requests,
            result_processor=self._process_question_results,
        )

    def run_stages(
        self,
        stages: List[str],
        reports: Optional[Dict[str, str]] = None,
        input_files: Dict[str, str] = None,
    ) -> Dict[str, Dict]:
        """Run specified stages with proper data flow and dependencies.

        Args:
            stages: List of stage names to run
            reports: Raw report data as list of (report_id, text) tuples
            input_files: Dict mapping stage names to input file paths
        """
        start_time = time.time()

        results = {"no_comparisons": {}, "findings": {}, "category_findings": {}, "questions": {}}

        try:
            # Load all existing results first to ensure proper data flow
            for stage_name in ["no_comparisons", "findings", "category_findings", "questions"]:
                existing_results = self._load_existing_results(stage_name)
                if existing_results:
                    results[stage_name] = existing_results
                    self.logger.info(
                        f"Loaded {len(existing_results)} existing results for {stage_name}"
                    )

            # Process stages in dependency order
            for stage in [
                "remove-comparisons",
                "extract-findings",
                "map-categories",
                "process-questions",
            ]:
                if stage not in stages:
                    continue

                if stage == "remove-comparisons":
                    if reports is None:
                        raise ValueError("Raw reports required for remove-comparisons stage")
                    stage_results = self.run_remove_comparisons(reports)
                    if stage_results:
                        results["no_comparisons"].update(stage_results)

                elif stage == "extract-findings":
                    # Determine input data source
                    input_data = None
                    if "input-no-comparisons" in input_files:
                        # Load from previous stage file
                        input_data = self.storage.load_no_comparisons(
                            Path(input_files["input-no-comparisons"])
                        )
                    elif results["no_comparisons"]:
                        # Use results from current run
                        input_data = results["no_comparisons"]
                    else:
                        raise ValueError("No input data available for extract-findings stage")

                    stage_results = self.run_extract_findings(input_data)
                    if stage_results:
                        results["findings"].update(stage_results)

                elif stage == "map-categories":
                    # Determine input data source
                    findings_data = None
                    if "input-findings" in input_files:
                        # Load from previous stage file
                        findings_data = self.storage.load_findings(
                            Path(input_files["input-findings"])
                        )
                    elif results["findings"]:
                        # Use results from current run/existing data
                        findings_data = results["findings"]
                    else:
                        raise ValueError("No findings data available for map-categories stage")

                    stage_results = self.run_map_categories(findings_data)
                    if stage_results:
                        results["category_findings"].update(stage_results)

                elif stage == "process-questions":
                    if reports is None:
                        raise ValueError("Raw reports required for process-questions stage")
                    stage_results = self.run_process_questions(reports)
                    if stage_results:
                        results["questions"].update(stage_results)

            final_results = self._combine_results(
                reports,
                results.get("no_comparisons", {}),
                results.get("findings", {}),
                results.get("category_findings", {}),
                results.get("questions", {}),
                start_time,
            )

            self.storage.save_final_results(final_results)
            return final_results

        except Exception as e:
            self.logger.error(f"Stage processing error: {e}")
            raise

    def _combine_results(
        self,
        reports,
        no_comparisons,
        findings_results,
        category_findings,
        question_results,
        start_time,
    ):
        """Combine all processing results."""
        results = {}
        processing_time = time.time() - start_time

        # If reports is None (e.g., when running just intermediate stages),
        # get report IDs from other result dictionaries
        if reports is None:
            # Collect all report IDs from available results
            all_report_ids = set()
            for result_dict in [
                no_comparisons,
                findings_results,
                category_findings,
                question_results,
            ]:
                if result_dict:
                    all_report_ids.update(result_dict.keys())

            # Create results for all known report IDs
            for report_id in all_report_ids:
                results[report_id] = {
                    "raw_text": "",  # Not available when running intermediate stages
                    "no_comparison_text": no_comparisons.get(report_id, ""),
                    "findings": findings_results.get(report_id, ""),
                    "category_findings": category_findings.get(report_id, {}),
                    "answers": question_results.get(report_id, {}),
                    "processing_time": processing_time,
                }
        else:
            # Original logic when reports are available
            for report_id, raw_text in reports.items():
                results[report_id] = {
                    "raw_text": raw_text,
                    "no_comparison_text": no_comparisons.get(report_id, ""),
                    "findings": findings_results.get(report_id, ""),
                    "category_findings": category_findings.get(report_id, {}),
                    "answers": question_results.get(report_id, {}),
                    "processing_time": processing_time,
                }

        return results

    @property
    def findings_categories(self) -> Dict:
        """Get the findings categories, loading them if necessary."""
        if self._findings_categories is None:
            self._findings_categories = self._load_findings_categories()
        return self._findings_categories

    def _load_findings_categories(self) -> Dict:
        """Load findings categories from modality configuration.

        Returns:
            Dictionary mapping category names to category objects
        """
        with open(self.modality_config_path, "r") as f:
            modality_config = yaml.safe_load(f)

        if "categories" not in modality_config:
            raise ValueError(f"No categories found in modality config: {self.modality_config_path}")

        return modality_config["categories"]

    def _load_existing_results(self, stage_name: str) -> Dict:
        """Load existing results for a stage."""
        file_mapping = {
            "no_comparisons": ("no_comparisons.csv", self.storage.load_no_comparisons),
            "findings": ("findings.csv", self.storage.load_findings),
            "category_findings": ("category_findings.csv", self.storage.load_category_findings),
            "questions": ("questions.csv", self.storage.load_questions),
        }

        if stage_name in file_mapping:
            filename, loader = file_mapping[stage_name]
            file_path = self.storage.save_dir / filename
            if file_path.exists():
                return loader(file_path)
        return {}

    def _build_remove_comparisons_requests(
        self, reports_data: Dict[str, str]
    ) -> List[Tuple[str, str]]:
        """Build requests for remove-comparisons stage."""
        return [
            (rid, self.config["prompt_templates"]["remove_comparison"].format(report=text))
            for rid, text in reports_data.items()
        ]

    def _build_extract_findings_requests(
        self, no_comparisons_data: Dict[str, str]
    ) -> List[Tuple[str, str]]:
        """Build requests for extract-findings stage."""
        return [
            (rid, self.config["prompt_templates"]["extract_findings"].format(report=text))
            for rid, text in no_comparisons_data.items()
        ]

    def _build_map_categories_requests(
        self, findings_data: Dict[str, str]
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Tuple[str, str]]]:
        """Build requests for map-categories stage."""
        if not self.findings_categories:
            self.findings_categories = self._load_findings_categories()

        requests = []
        id_mapping = {}

        for report_id, findings in findings_data.items():
            # Skip empty findings or "No such section" responses
            if (
                findings
                and findings.strip()
                and findings.strip().lower() not in ["no such section", "no findings section"]
            ):
                for category_name, category in self.findings_categories.items():
                    custom_id = f"{report_id}_{category_name}"
                    prompt = self.config["prompt_templates"]["map_category_findings"].format(
                        findings=findings,
                        category=category_name,
                        description=category["description"],
                    )
                    requests.append((custom_id, prompt))
                    id_mapping[custom_id] = (report_id, category_name)

        return requests, id_mapping

    def _build_process_questions_requests(
        self, reports_data: Dict[str, str]
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Tuple[str, str, str]]]:
        """Build requests for process-questions stage."""
        if not self.findings_categories:
            self.findings_categories = self._load_findings_categories()

        requests = []
        id_mapping = {}

        for report_id, raw_text in reports_data.items():
            for category_name, category in self.findings_categories.items():
                for question in category["questions"]:
                    question_text = question["question"]
                    question_id = self.storage._generate_question_id(question_text)
                    request_id = f"{report_id}_{category_name}_{question_id}"
                    prompt = self.config["prompt_templates"]["findings_question"].format(
                        report=raw_text, question=question_text
                    )
                    requests.append((request_id, prompt))
                    id_mapping[request_id] = (report_id, category_name, question_text)

        return requests, id_mapping

    def _process_category_results(
        self, chunk_results: Dict[str, str], id_mapping: Dict[str, Tuple[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """Process map-categories results into nested structure."""
        chunk_nested_results = {}
        for custom_id, content in chunk_results.items():
            report_id, category_name = id_mapping[custom_id]
            chunk_nested_results.setdefault(report_id, {})[category_name] = content
        return chunk_nested_results

    def _process_question_results(
        self, chunk_results: Dict[str, str], id_mapping: Dict[str, Tuple[str, str, str]]
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """Process question results into nested structure."""
        chunk_nested_results = {}
        for request_id, answer in chunk_results.items():
            report_id, category_name, question_text = id_mapping[request_id]
            chunk_nested_results.setdefault(report_id, {}).setdefault(category_name, []).append(
                {question_text: answer}
            )
        return chunk_nested_results

    def _process_stage(
        self,
        stage_name: str,
        input_data: Dict[str, str],
        request_builder: callable,
        result_processor: callable = None,
    ) -> Dict:
        """Generic stage processing with incremental saving and resume logic."""

        self.logger.info(f"Processing {stage_name}")

        # Build all requests first
        builder_result = request_builder(input_data)
        if isinstance(builder_result, tuple):
            requests, id_mapping = builder_result
        else:
            requests, id_mapping = builder_result, None

        # Filter out already processed requests
        if stage_name in ["no_comparisons", "findings"]:
            # Simple stages: filter by report IDs
            processed_ids = self.storage.get_processed_report_ids(stage_name)
            if processed_ids:
                self.logger.info(
                    f"Found {len(processed_ids)} processed reports, filtering them out"
                )
                requests = [
                    (req_id, prompt) for req_id, prompt in requests if req_id not in processed_ids
                ]
        else:
            # Complex stages: filter by request IDs
            processed_request_ids = self.storage.get_processed_request_ids(stage_name)
            if processed_request_ids:
                self.logger.info(
                    f"Found {len(processed_request_ids)} processed requests, filtering them out"
                )
                requests = [
                    (req_id, prompt)
                    for req_id, prompt in requests
                    if req_id not in processed_request_ids
                ]

        # Early return if all work is done
        if not requests:
            self.logger.info(f"All requests already processed for {stage_name}")
            return self._load_existing_results(stage_name)

        # Process requests in batches with improved error handling
        all_results = {}
        chunks = [
            requests[i : i + self._batch_size] for i in range(0, len(requests), self._batch_size)
        ]

        for i, chunk in enumerate(
            tqdm_max_ncols(
                chunks, desc=f"Processing {stage_name} batches", unit="batch", smoothing=0
            )
        ):
            try:
                # Use new batch processor with validation
                chunk_results = self.batch_processor.process_batch(chunk, stage_name)

                # Apply result processor if provided
                if result_processor:
                    chunk_results = result_processor(chunk_results, id_mapping)

                # Skip redundant validation - batch processor already validated individual results
                # Additional validation here was causing 92% data loss by re-filtering valid responses

                # Only save and accumulate if we have valid results
                if chunk_results:
                    all_results.update(chunk_results)
                    self.storage.save_incremental_results(stage_name, chunk_results)
                    self.logger.info(
                        f"Saved {len(chunk_results)} results from batch {i+1}/{len(chunks)}"
                    )
                else:
                    self.logger.warning(f"No valid results from batch {i+1}/{len(chunks)}")

            except BatchProcessingError as e:
                self.logger.error(f"Batch {i+1}/{len(chunks)} failed: {str(e)}")
                # Continue with next batch rather than failing entire stage
                continue
            except ValidationError as e:
                self.logger.error(f"Validation failed for batch {i+1}/{len(chunks)}: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error in batch {i+1}/{len(chunks)}: {str(e)}")
                # For unexpected errors, we might want to fail the entire stage
                raise

        # Validate CSV file integrity after processing
        if not self.storage.validate_csv_file(stage_name):
            self.logger.warning(f"CSV file validation failed for {stage_name}")

        return all_results

    def _apply_debug_subsampling(self):
        """Apply debug mode subsampling to reduce data for faster iteration."""
        debug_config = self.config.get("debug", {})
        seed = debug_config.get("seed", 42)
        np.random.seed(seed)

        self.logger.info("Applying debug mode subsampling...")

        # 1. Subsample categories
        original_categories = self._load_findings_categories()
        category_names = list(original_categories.keys())

        debug_categories_count = min(debug_config.get("categories", 4), len(category_names))
        selected_categories = np.random.choice(
            category_names, size=debug_categories_count, replace=False
        )

        # 2. Subsample questions per category
        debug_num_questions = debug_config.get("num_questions", 10)
        filtered_categories = {}

        for category_name in selected_categories:
            category = original_categories[category_name].copy()
            questions = category["questions"]

            # Sample questions if there are more than the debug limit
            if len(questions) > debug_num_questions:
                selected_indices = np.random.choice(
                    len(questions), size=debug_num_questions, replace=False
                )
                category["questions"] = [questions[i] for i in selected_indices]

            filtered_categories[category_name] = category

        # 3. Override the findings categories with filtered version
        self._findings_categories = filtered_categories

        # 4. Log the subsampling results
        total_questions = sum(len(cat["questions"]) for cat in filtered_categories.values())
        self.logger.info(
            f"Debug mode: selected {len(filtered_categories)} categories: {list(filtered_categories.keys())}"
        )
        self.logger.info(f"Debug mode: total questions after filtering: {total_questions}")

        # Note: Report subsampling will be handled at the CLI level when loading CSVs
