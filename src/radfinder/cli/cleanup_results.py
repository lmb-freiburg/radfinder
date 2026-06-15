"""
Delete experiment folders that have no checkpoints and are older than a given age.

Usage:
    python -m radfinder.cli.cleanup_results
    python -m radfinder.cli.cleanup_results -s subfolder1,subfolder2
    python -m radfinder.cli.cleanup_results --age 48
    python -m radfinder.cli.cleanup_results --write
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from attrs import define
from loguru import logger
from radfinder.paths import get_medv_output_dir

from packg.log import configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument


@define
class Args(VerboseQuietArgs):
    subfolder: str = add_argument(
        positional=True, help="Comma-separated subfolders under MEDV_OUTPUT_DIR"
    )
    age: float = add_argument(
        help="Minimum age in hours for a folder to be considered for deletion.",
        default=24.0,
    )
    write: bool = add_argument(
        help="Actually delete folders. Default is dry-run.",
        action="store_true",
    )


def main() -> None:
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(get_logger_level_from_args(args))
    logger.info(args)

    subfolders = [s.strip() for s in args.subfolder.split(",") if s.strip()]
    if len(subfolders) == 0:
        logger.warning("No subfolders found")
        return

    logger.info(f"Scanning subfolders: {subfolders}")
    now = time.time()
    min_age_seconds = args.age * 3600
    candidates: list[tuple[Path, float, float]] = []
    too_recent: list[tuple[Path, float]] = []

    for subfolder in subfolders:
        base_dir = get_medv_output_dir() / subfolder
        if not base_dir.is_dir():
            logger.warning(f"Skipping missing directory: {base_dir}")
            continue
        for exp_dir in sorted(base_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if len(list(exp_dir.glob("checkpoint*.pt"))) > 0:
                continue
            oldest_mtime = _get_oldest_file_mtime(exp_dir)
            if oldest_mtime is None:
                oldest_mtime = exp_dir.stat().st_mtime
            age_seconds = now - oldest_mtime
            if age_seconds < min_age_seconds:
                too_recent.append((exp_dir, age_seconds))
                continue
            dir_size = sum(f.stat().st_size for f in exp_dir.rglob("*") if f.is_file())
            candidates.append((exp_dir, age_seconds, dir_size))

    if len(too_recent) > 0:
        for exp_dir, age_seconds in too_recent:
            logger.info(f"TOO RECENT: {exp_dir.as_posix()} ({age_seconds / 3600:.1f}h old)")

    if len(candidates) == 0:
        logger.info("No experiment folders to clean up")
        return

    total_size = sum(size for _, _, size in candidates)
    for exp_dir, age_seconds, dir_size in candidates:
        age_hours = age_seconds / 3600
        action = "DELETE" if args.write else "WOULD DELETE"
        logger.info(
            f"{action}: {exp_dir.as_posix()} "
            f"({dir_size / 1024 / 1024:.1f} MB, {age_hours:.0f}h old)"
        )
        if args.write:
            shutil.rmtree(exp_dir)

    logger.info(
        f"Total: {len(candidates)} folders, {total_size / 1024 / 1024:.1f} MB"
        f"{' deleted' if args.write else ' (dry run)'}"
    )


def _get_oldest_file_mtime(directory: Path) -> float | None:
    oldest = None
    for f in directory.rglob("*"):
        if not f.is_file():
            continue
        if f.name == "cached_val_output.csv":
            continue
        mtime = f.stat().st_mtime
        if oldest is None or mtime < oldest:
            oldest = mtime
    return oldest


if __name__ == "__main__":
    main()
