"""
Show validation results for SPECTRE experiments.

Usage:
    python -m radfinder.cli.show_results -s evals -g default
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from attrs import define
from loguru import logger
from radfinder.utils.show_results_lib import (
    METRIC_GROUPS,
    describe_metric_columns,
    get_results_dataframe,
)

from packg.log import configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument


@define
class Args(VerboseQuietArgs):
    subfolder: str | None = add_argument(
        shortcut="-s",
        help="Comma-separated subfolders under MEDV_OUTPUT_DIR (e.g. train1,train2)",
        default=None,
    )
    group: str = add_argument(
        shortcut="-g",
        default="default",
        choices=list(METRIC_GROUPS.keys()),
        help="Metric group to display",
    )
    all_epochs: bool = add_argument(
        shortcut="-a",
        default=False,
        help="Show all epochs instead of only best and last",
    )


def main() -> None:
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(get_logger_level_from_args(args))
    logger.info(args)

    if args.subfolder is None:
        raise ValueError("Set -s to one or more subfolders, e.g. -s train1,train2")

    df = get_results_dataframe(args.subfolder, args.group, all_epochs=args.all_epochs)
    if len(df) > 0:
        _print_single_space(df)
        _log_column_legend(df)
        _dump_results_dataframe(args.subfolder, df, args.group)


def _print_single_space(df: pd.DataFrame) -> None:
    """Print the dataframe with a single space between columns (right-justified)."""
    columns = list(df.columns)
    # Reuse pandas' per-column value formatting (e.g. consistent float decimals).
    col_cells = {
        col: [cell.strip() for cell in df[[col]].to_string(index=False, header=False).split("\n")]
        for col in columns
    }
    index_strs = [str(idx) for idx in df.index]
    index_width = max((len(s) for s in index_strs), default=0)
    widths = {col: max(len(str(col)), *(len(c) for c in col_cells[col])) for col in columns}

    header = " ".join([" " * index_width] + [str(col).rjust(widths[col]) for col in columns])
    print(header)
    for row in range(len(df)):
        cells = [index_strs[row].rjust(index_width)] + [
            col_cells[col][row].rjust(widths[col]) for col in columns
        ]
        print(" ".join(cells))


def _log_column_legend(df: pd.DataFrame) -> None:
    """Log, for each metric column, the full dataset name and metric key."""
    described = describe_metric_columns(list(df.columns))
    if not described:
        return
    col_width = max(len(column) for column, _, _ in described)
    lines = [
        f"  {column:<{col_width}s}  {datasets} : {metric_key}"
        for column, datasets, metric_key in described
    ]
    logger.info("Columns (short -> dataset name : metric):\n" + "\n".join(lines))


def _dump_results_dataframe(subfolders_raw: str, df: pd.DataFrame, metric_group: str) -> None:
    subfolders = sorted({s.strip() for s in subfolders_raw.split(",") if s.strip()})
    target_dir = Path.home() / "temp_results_output" / ",".join(subfolders)
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"metrics_{metric_group}.csv" if metric_group != "default" else "metrics.csv"
    out_path = target_dir / filename
    df.to_csv(out_path.as_posix(), index=False)
    logger.info(f"Saved results to {out_path.as_posix()}")


if __name__ == "__main__":
    main()
