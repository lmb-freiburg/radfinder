"""
Result validation utilities for the Rad Text Engine (RaTE).

Vendored from YalaLab/rate@79b23df src/core/validators.py (ECL 2.0). Changes:
- accept `(content, reasoning_content)` tuples instead of bare content strings,
  so qwen3-style `<think>...</think>` reasoning is kept alongside the answer
- return `(valid_results, invalid_results)` so the caller can retry failures
- keep retrying when len(requests) < 100 even if all results are invalid
- switch from injected stdlib logger to module-level loguru logger
"""

from typing import Dict, List, Set, Tuple

from loguru import logger

from .exceptions import ValidationError


class ResultValidatorWithReasoning:
    """Validates processing results for different stages."""

    def __init__(self):
        self.invalid_findings = {"", "No such section", "No findings section"}
        # For category findings, only filter out truly empty responses or clear errors
        # "No findings" is a valid medical assessment and should NOT be filtered
        self.invalid_category_findings = {"", "error", "failed", "unable to process"}

    def validate_batch_results(
        self, results: Dict[str, Tuple[str, str]], requests: List[Tuple[str, str]], stage_name: str
    ) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]]]:
        """Validate batch results and filter out invalid ones."""
        if not results:
            raise ValidationError(f"No results returned for batch in {stage_name}")

        # Check if we got results for all requests
        request_ids = {req_id for req_id, _ in requests}
        result_ids = set(results.keys())

        missing_ids = request_ids - result_ids
        if missing_ids:
            logger.warning(
                f"Missing results for {len(missing_ids)} requests in {stage_name}: {list(missing_ids)[:5]}..."
            )

        # Validate individual results
        valid_results, invalid_results = {}, {}
        invalid_count = 0
        invalid_patterns = {}

        for request_id, (content, reasoning_content) in results.items():
            if self._is_valid_result(content, stage_name):
                valid_results[request_id] = (content, reasoning_content)
            else:
                invalid_results[request_id] = (content, reasoning_content)
                invalid_count += 1
                # Track patterns of invalid results
                content_lower = content.strip().lower()
                if not content_lower:
                    pattern = "EMPTY"
                elif len(content_lower) < 20:
                    pattern = f"SHORT: '{content_lower}'"
                else:
                    pattern = f"LONG: '{content_lower[:30]}...'"

                invalid_patterns[pattern] = invalid_patterns.get(pattern, 0) + 1

                if invalid_count <= 10:  # Log first 10 invalid results in detail
                    logger.warning(
                        f"Invalid result for {request_id} in {stage_name}:\nCONTENT: {content}\n"
                        f"REASONING: {reasoning_content}\n"
                    )

        if invalid_count > 0:
            logger.info(f"Filtered out {invalid_count} invalid results from {stage_name}")
            logger.info(f"Invalid result patterns: {invalid_patterns}")

        if not valid_results and len(requests) > 100:
            # for small batch sizes it can happen that all are invalid and should be retried
            raise ValidationError(f"All results were invalid for batch in {stage_name}")

        return valid_results, invalid_results

    def _is_valid_result(self, content: str, stage_name: str) -> bool:
        """Check if a result is valid for the given stage."""
        if not content or not content.strip():
            return False

        content_lower = content.strip().lower()

        if stage_name == "findings":
            return content_lower not in {s.lower() for s in self.invalid_findings}
        elif stage_name == "category_findings":
            # For category findings, accept any non-empty response including "no findings"
            # Only filter out clear errors or empty responses
            return content_lower not in {s.lower() for s in self.invalid_category_findings}
        elif stage_name in ["questions", "no_comparisons"]:
            # For questions, we expect "Yes" or "No" answers
            # For no_comparisons, any non-empty text is valid
            return len(content.strip()) > 0

        return True

    def validate_findings(self, results: Dict[str, str]) -> Dict[str, str]:
        """Validate findings results and filter out invalid ones."""
        valid_results = {}
        for report_id, findings in results.items():
            if findings and findings.strip().lower() not in {
                s.lower() for s in self.invalid_findings
            }:
                valid_results[report_id] = findings
            else:
                logger.warning(f"Invalid findings for report {report_id}: '{findings}'")
        return valid_results

    def validate_category_findings(
        self, results: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """Validate category findings and filter out invalid ones."""
        valid_results = {}
        for report_id, categories in results.items():
            valid_categories = {}
            for category, findings in categories.items():
                # Accept any non-empty response, including "No findings" which is valid
                if (
                    findings
                    and findings.strip()
                    and findings.strip().lower()
                    not in {s.lower() for s in self.invalid_category_findings}
                ):
                    valid_categories[category] = findings
                else:
                    logger.warning(
                        f"Invalid category findings for report {report_id}, category {category}: '{findings}'"
                    )

            if valid_categories:
                valid_results[report_id] = valid_categories

        return valid_results

    def validate_csv_integrity(self, file_path, expected_columns: List[str]) -> bool:
        """Validate CSV file integrity."""
        try:
            import pandas as pd

            df = pd.read_csv(file_path)

            # Check columns
            if not set(expected_columns).issubset(set(df.columns)):
                logger.error(
                    f"CSV missing required columns. Expected: {expected_columns}, Got: {list(df.columns)}"
                )
                return False

            # Check for empty rows
            empty_rows = df.isnull().all(axis=1).sum()
            if empty_rows > 0:
                logger.warning(f"Found {empty_rows} empty rows in {file_path}")

            # Check for duplicate entries (if report_id column exists)
            if "report_id" in df.columns:
                duplicates = df.duplicated(subset=["report_id"]).sum()
                if duplicates > 0:
                    logger.warning(f"Found {duplicates} duplicate report_ids in {file_path}")

            return True

        except Exception as e:
            logger.error(f"Error validating CSV {file_path}: {e}")
            return False
