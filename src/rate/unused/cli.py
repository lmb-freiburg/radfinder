"""Main CLI for processing radiology reports with AI-powered analysis.

This module provides the command-line interface for the Rad Text Engine (RaTE),
supporting stage-based processing, debug mode, and flexible input/output options.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from rate.engine import Engine
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process medical reports with configurable settings."
    )

    # Required settings
    parser.add_argument(
        "--save-dir", type=str, required=True, help="Directory to save processed results"
    )
    parser.add_argument(
        "--modality-config",
        type=str,
        required=True,
        help="Path to modality-specific configuration file",
    )

    # Stage selection
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[
            "remove-comparisons",
            "extract-findings",
            "map-categories",
            "process-questions",
            "all",
        ],
        default=["all"],
        help="Stages to execute (default: all)",
    )

    # Input data options
    parser.add_argument(
        "--input-files",
        nargs="+",
        help="Input CSV or JSON files containing reports (required for remove-comparisons and process-questions stages)",
    )

    # Intermediate file inputs
    parser.add_argument(
        "--input-no-comparisons", type=str, help="Path to no_comparisons.csv from previous run"
    )
    parser.add_argument("--input-findings", type=str, help="Path to findings.csv from previous run")
    parser.add_argument(
        "--input-category-findings",
        type=str,
        help="Path to category_findings.csv from previous run",
    )

    # Debug mode options
    parser.add_argument(
        "--debug-mode", action="store_true", help="Enable debug mode with subsampling"
    )
    parser.add_argument(
        "--debug-sample",
        type=int,
        default=5000,
        help="Max number of reports to sample in debug mode (default: 5000)",
    )
    parser.add_argument(
        "--debug-categories",
        type=int,
        default=4,
        help="Number of categories to sample in debug mode (default: 4)",
    )
    parser.add_argument(
        "--debug-num-questions",
        type=int,
        default=10,
        help="Number of questions per category in debug mode (default: 10)",
    )
    parser.add_argument(
        "--debug-seed", type=int, default=42, help="Random seed for debug sampling (default: 42)"
    )

    # Optional settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for processing reports (default: 1024)",
    )
    parser.add_argument(
        "--accession-col",
        default="Accession",
        help="Column name for accession numbers in CSV (default: Accession)",
    )
    parser.add_argument(
        "--report-col",
        default="Report Text",
        help="Column name for report text in CSV (default: Report Text)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help='Path to default configuration file (default: "config/default_config.yaml")',
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def validate_stage_inputs(stages: list, args) -> dict:
    """Validate that required inputs are available for selected stages."""
    input_files = {}
    reports_needed = False

    # Check which inputs are needed
    if "remove-comparisons" in stages or "process-questions" in stages or args.stages == ["all"]:
        reports_needed = True

    if "extract-findings" in stages and not (
        "remove-comparisons" in stages or args.stages == ["all"]
    ):
        if not args.input_no_comparisons:
            raise ValueError(
                "extract-findings stage requires --input-no-comparisons when not running remove-comparisons"
            )
        input_files["input-no-comparisons"] = args.input_no_comparisons

    if "map-categories" in stages and not ("extract-findings" in stages or args.stages == ["all"]):
        if not args.input_findings:
            raise ValueError(
                "map-categories stage requires --input-findings when not running extract-findings"
            )
        input_files["input-findings"] = args.input_findings

    # Check if we need input files for reports
    if reports_needed and not args.input_files:
        raise ValueError(
            "Raw reports (--input-files) required for remove-comparisons or process-questions stages"
        )

    return input_files


def load_reports_from_files(
    input_files: list, accession_col: str, report_col: str, debug_config: Optional[dict] = None
) -> dict:
    """Load reports from CSV or JSON files."""
    all_reports = []

    for input_file in input_files:
        file_path = Path(input_file)
        file_extension = file_path.suffix.lower()

        try:
            if file_extension == ".csv":
                df = pd.read_csv(input_file)
                print(f"Loading {len(df)} reports from CSV file: {input_file}")

                # Collect all (accession_number, report) tuples
                report_tuples = [
                    (str(row[accession_col]), row[report_col]) for _, row in df.iterrows()
                ]
                all_reports.extend(report_tuples)

            elif file_extension == ".json":
                with open(input_file, "r") as f:
                    data = json.load(f)

                # Handle different JSON structures
                if isinstance(data, list):
                    # Array of objects: [{"accession": "123", "report": "text"}, ...]
                    print(f"Loading {len(data)} reports from JSON file: {input_file}")
                    report_tuples = [
                        (str(item[accession_col]), item[report_col])
                        for item in data
                        if accession_col in item and report_col in item
                    ]
                    all_reports.extend(report_tuples)

                elif isinstance(data, dict):
                    # Check if it's a mapping of accession -> report
                    if all(isinstance(v, str) for v in data.values()):
                        print(
                            f"Loading {len(data)} reports from JSON file (key-value format): {input_file}"
                        )
                        report_tuples = [(str(k), v) for k, v in data.items()]
                        all_reports.extend(report_tuples)
                    else:
                        # Assume it's an object with array of reports
                        reports_array = data.get("reports", data.get("data", []))
                        if isinstance(reports_array, list):
                            print(
                                f"Loading {len(reports_array)} reports from JSON file (nested format): {input_file}"
                            )
                            report_tuples = [
                                (str(item[accession_col]), item[report_col])
                                for item in reports_array
                                if accession_col in item and report_col in item
                            ]
                            all_reports.extend(report_tuples)
                        else:
                            raise ValueError(f"Unsupported JSON structure in {input_file}")
                else:
                    raise ValueError(f"Unsupported JSON structure in {input_file}")

            else:
                raise ValueError(
                    f"Unsupported file format: {file_extension}. Only .csv and .json files are supported."
                )

        except Exception as e:
            print(f"Error reading file {input_file}: {str(e)}")
            raise

    reports_dict = {str(rid): text for rid, text in all_reports}

    # Apply debug subsampling if enabled
    if debug_config and debug_config.get("enabled", False):
        debug_sample = debug_config.get("sample", 5000)
        seed = debug_config.get("seed", 42)
        np.random.seed(seed)

        if len(reports_dict) > debug_sample:
            print(
                f"Debug mode: subsampling {debug_sample} reports from {len(reports_dict)} total reports"
            )

            # Sample report IDs
            report_ids = list(reports_dict.keys())
            selected_ids = np.random.choice(report_ids, size=debug_sample, replace=False)

            # Create subsampled dictionary
            reports_dict = {rid: reports_dict[rid] for rid in selected_ids}
            print(f"Debug mode: using {len(reports_dict)} sampled reports")

    return reports_dict


def main():
    args = parse_args()

    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Handle 'all' stages
    if args.stages == ["all"]:
        stages = ["remove-comparisons", "extract-findings", "map-categories", "process-questions"]
    else:
        # Ensure all stages are in order of dependencies
        stages = [
            stage
            for stage in [
                "remove-comparisons",
                "extract-findings",
                "map-categories",
                "process-questions",
            ]
            if stage in args.stages
        ]

    print(f"Running stages: {', '.join(stages)}")

    # Validate inputs
    try:
        input_files = validate_stage_inputs(stages, args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Load default config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Add intermediate input files to input_files dict
    if args.input_no_comparisons:
        input_files["input-no-comparisons"] = args.input_no_comparisons
    if args.input_findings:
        input_files["input-findings"] = args.input_findings
    if args.input_category_findings:
        input_files["input-category-findings"] = args.input_category_findings

    reports = None

    # Construct the complete config dictionary first
    config_dict = {
        "input-files": args.input_files or [],
        "modality-config": args.modality_config,
        "accession-col": args.accession_col,
        "report-col": args.report_col,
        "processing": {"batch-size": args.batch_size, "save-dir": args.save_dir},
        "debug": {
            "enabled": args.debug_mode,
            "sample": args.debug_sample,
            "categories": args.debug_categories,
            "num_questions": args.debug_num_questions,
            "seed": args.debug_seed,
        },
    }

    # Load reports if needed
    if args.input_files:
        reports = load_reports_from_files(
            args.input_files, args.accession_col, args.report_col, config_dict.get("debug")
        )
        print(f"Loaded {len(reports)} total reports")

    # Merge with default config
    config.update(config_dict)

    # Set debug logging if requested
    if args.debug:
        import logging

        # Only set debug level for our application loggers, not all loggers
        logging.getLogger("Engine").setLevel(logging.DEBUG)
        logging.getLogger("BatchProcessor").setLevel(logging.DEBUG)
        logging.getLogger("StorageManager").setLevel(logging.DEBUG)
        logging.getLogger("QCGenerator").setLevel(logging.DEBUG)
        print("Debug logging enabled for rad-report-engine modules")

    # Initialize processor
    processor = Engine(config)

    try:
        # Run the specified stages
        results = processor.run_stages(stages=stages, reports=reports, input_files=input_files)

        print(f"Successfully completed stages: {', '.join(stages)}")
        print(f"Results contain {len(results)} items")

    except Exception as e:
        print(f"Error during processing: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
