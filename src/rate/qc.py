import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QCGenerator:
    """Quality Control file generator for radiology report processing results."""

    def __init__(self, results_path: str, qc_dir: str):
        """
        Initialize QC generator with results and output directory.

        Args:
            results_path: Path to the JSON file containing processing results
            qc_dir: Directory to save QC files
        """
        self.results_path = Path(results_path)
        self.qc_dir = Path(qc_dir)
        self.qc_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        with open(self.results_path, "r") as f:
            self.results = json.load(f)

        logger.info(f"Loaded {len(self.results)} reports from {results_path}")

    def generate_combined_findings_qc(self, budget: int = 40) -> str:
        """
        Generate combined QC file for both no-comparison text extraction and findings extraction.

        Args:
            budget: Number of samples to include

        Returns:
            Path to generated CSV file
        """
        report_ids = list(self.results.keys())
        if len(report_ids) < budget:
            budget = len(report_ids)
            logger.warning(f"Budget reduced to {budget} due to insufficient samples")

        query_ids = np.random.choice(report_ids, size=budget, replace=False)

        results = []
        for query_id in query_ids:
            report = self.results[query_id]
            raw_text = report.get("raw_text", "")
            no_comparison_text = report.get("no_comparison_text", "")

            # Extract findings - use the correct key 'findings_text'
            findings_text = report.get("findings_text", "")

            results.append(
                {
                    "query_id": query_id,
                    "raw_text": raw_text,
                    "no_comparison_text": no_comparison_text,
                    "findings": findings_text,
                    "no_comparison_correct": "",  # For human annotation
                    "findings_correct": "",  # For human annotation
                }
            )

        df = pd.DataFrame(results)
        output_file = self.qc_dir / "combined_findings_qc.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"Generated combined findings QC file: {output_file}")
        logger.info(f"  Samples: {len(df)}")

        return str(output_file)

    def generate_category_qc(self, budget_per_category: int = 10) -> str:
        """
        Generate a single consolidated QC file for all category findings.

        Args:
            budget_per_category: Number of samples per category

        Returns:
            Path to generated CSV file
        """
        # Get all unique categories from all reports
        all_categories = set()
        for report in self.results.values():
            if "category_findings" in report and isinstance(report["category_findings"], dict):
                all_categories.update(report["category_findings"].keys())

        all_samples = []

        for category in all_categories:
            # Separate reports with "No relevant findings" vs actual findings
            relevant_reports = []
            no_findings_reports = []

            for query_id, report in self.results.items():
                if "category_findings" not in report:
                    continue

                category_findings = report["category_findings"]
                if category in category_findings:
                    finding_text = str(category_findings[category])
                    if "No relevant findings" in finding_text:
                        no_findings_reports.append((query_id, report, finding_text))
                    else:
                        relevant_reports.append((query_id, report, finding_text))

            # Sample with overweighting towards relevant findings
            samples = []

            # Take more samples from relevant findings (if available)
            relevant_budget = min(int(budget_per_category * 0.7), len(relevant_reports))
            no_findings_budget = budget_per_category - relevant_budget

            if relevant_reports:
                relevant_sample_ids = np.random.choice(
                    len(relevant_reports), size=relevant_budget, replace=False
                )
                samples.extend([relevant_reports[i] for i in relevant_sample_ids])

            if no_findings_reports and no_findings_budget > 0:
                no_findings_sample_size = min(no_findings_budget, len(no_findings_reports))
                no_findings_sample_ids = np.random.choice(
                    len(no_findings_reports), size=no_findings_sample_size, replace=False
                )
                samples.extend([no_findings_reports[i] for i in no_findings_sample_ids])

            # Add samples to consolidated list
            for query_id, report, finding_text in samples:
                all_samples.append(
                    {
                        "query_id": query_id,
                        "category": category,
                        "raw_text": report.get("raw_text", ""),
                        "category_finding": finding_text,
                        "correct": "",  # For human annotation
                    }
                )

        if all_samples:
            # Shuffle all samples together
            np.random.shuffle(all_samples)

            df = pd.DataFrame(all_samples)
            output_path = self.qc_dir / "categories_qc.csv"
            df.to_csv(output_path, index=False)

            categories_count = len(all_categories)
            logger.info(
                f"Generated consolidated categories QC file with {len(all_samples)} samples from {categories_count} categories: {output_path}"
            )
            return str(output_path)

        logger.warning("No category samples found for QC generation")
        return ""

    def generate_questions_qc(self, budget: int = 50) -> str:
        """
        Generate a consolidated QC file for all questions with mixed positive/negative examples.
        Uses stratified sampling: first sample a question, then sample a pos/neg example from that question.
        This creates a single CSV with random questions and labels, making annotation more efficient.

        Args:
            budget: Total number of samples (half positive, half negative)

        Returns:
            Path to generated CSV file
        """
        # Get all question-answer pairs from all reports
        question_examples = {}  # question -> {"positive": [...], "negative": [...]}

        for query_id, report in self.results.items():
            if "qa_results" not in report:
                continue

            for category_qa in report["qa_results"].values():
                if isinstance(category_qa, list):
                    for qa_dict in category_qa:
                        if isinstance(qa_dict, dict):
                            for question, answer in qa_dict.items():
                                # Convert answer to binary
                                if isinstance(answer, str):
                                    answer_lower = answer.lower().strip()
                                    if answer_lower in ["yes", "true", "1"]:
                                        binary_label = 1
                                    elif answer_lower in ["no", "false", "0"]:
                                        binary_label = 0
                                    else:
                                        continue
                                elif isinstance(answer, (int, float)):
                                    binary_label = 1 if answer > 0 else 0
                                else:
                                    continue

                                if question not in question_examples:
                                    question_examples[question] = {"positive": [], "negative": []}

                                example = {
                                    "query_id": query_id,
                                    "question": question,
                                    "raw_text": report.get("raw_text", ""),
                                    "predicted_label": binary_label,
                                }

                                if binary_label == 1:
                                    question_examples[question]["positive"].append(example)
                                else:
                                    question_examples[question]["negative"].append(example)

        # Filter questions that have both positive and negative examples
        questions_with_pos = [
            q for q, examples in question_examples.items() if examples["positive"]
        ]
        questions_with_neg = [
            q for q, examples in question_examples.items() if examples["negative"]
        ]

        if not questions_with_pos and not questions_with_neg:
            logger.warning("No valid question examples found for QC generation")
            return ""

        # Sample examples using stratified sampling: half positive, half negative
        positive_budget = budget // 2
        negative_budget = budget - positive_budget

        samples = []

        # Sample positive examples using stratified sampling
        if questions_with_pos and positive_budget > 0:
            for _ in range(positive_budget):
                # First, randomly select a question that has positive examples
                selected_question = np.random.choice(questions_with_pos)
                # Then, randomly select a positive example from that question
                positive_examples = question_examples[selected_question]["positive"]
                selected_example = np.random.choice(positive_examples)
                samples.append(selected_example)

        # Sample negative examples using stratified sampling
        if questions_with_neg and negative_budget > 0:
            for _ in range(negative_budget):
                # First, randomly select a question that has negative examples
                selected_question = np.random.choice(questions_with_neg)
                # Then, randomly select a negative example from that question
                negative_examples = question_examples[selected_question]["negative"]
                selected_example = np.random.choice(negative_examples)
                samples.append(selected_example)

        # Shuffle the samples to mix positive and negative examples
        np.random.shuffle(samples)

        # Create DataFrame
        results = []
        for sample in samples:
            results.append(
                {
                    "query_id": sample["query_id"],
                    "question": sample["question"],
                    "raw_text": sample["raw_text"],
                    "predicted_label": sample["predicted_label"],
                    "human_label": "",  # For human annotation
                }
            )

        if results:
            df = pd.DataFrame(results)
            output_path = self.qc_dir / "questions_qc.csv"
            df.to_csv(output_path, index=False)

            # Log statistics
            pos_count = sum(1 for r in results if r["predicted_label"] == 1)
            neg_count = len(results) - pos_count
            unique_questions = len(set(r["question"] for r in results))
            total_available_questions = len(question_examples)
            questions_with_both = len(set(questions_with_pos) & set(questions_with_neg))

            logger.info(
                f"Generated questions QC file with {len(results)} samples "
                f"({pos_count} positive, {neg_count} negative, {unique_questions} unique questions): {output_path}"
            )
            logger.info(
                f"Stratified sampling from {total_available_questions} total questions "
                f"({len(questions_with_pos)} with positives, {len(questions_with_neg)} with negatives, "
                f"{questions_with_both} with both)"
            )
            return str(output_path)

        logger.warning("No valid question examples found for QC generation")
        return ""

    def generate_all_qc(
        self,
        combined_findings_budget: int = 40,
        category_budget: int = 10,
        questions_budget: int = 50,
    ) -> Dict[str, List[str]]:
        """
        Generate all QC files with simplified 3-CSV structure.

        Args:
            combined_findings_budget: Budget for combined findings+no-comparison QC
            category_budget: Budget per category for consolidated category QC
            questions_budget: Budget for questions QC

        Returns:
            Dictionary mapping QC type to list of generated file paths
        """
        results = {}

        logger.info("Generating simplified QC files (3 CSVs)...")

        # Generate combined findings QC (CSV 1: Findings+no comparison)
        try:
            combined_path = self.generate_combined_findings_qc(budget=combined_findings_budget)
            results["combined_findings"] = [combined_path] if combined_path else []
        except Exception as e:
            logger.error(f"Failed to generate combined findings QC: {e}")
            results["combined_findings"] = []

        # Generate consolidated category QC (CSV 2: QA style for categories)
        try:
            category_path = self.generate_category_qc(budget_per_category=category_budget)
            results["categories"] = [category_path] if category_path else []
        except Exception as e:
            logger.error(f"Failed to generate category QC: {e}")
            results["categories"] = []

        # Generate questions QC (CSV 3: QA as is, no changes)
        try:
            questions_path = self.generate_questions_qc(budget=questions_budget)
            results["questions"] = [questions_path] if questions_path else []
        except Exception as e:
            logger.error(f"Failed to generate questions QC: {e}")
            results["questions"] = []

        total_files = sum(len(paths) for paths in results.values())
        logger.info(f"Simplified QC generation complete. Generated {total_files} files total.")

        return results
