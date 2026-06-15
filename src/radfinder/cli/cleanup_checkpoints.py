"""Delete intermediate checkpoints, keeping only best (lowest meanr) and last (highest epoch).

Usage:
    python -m radfinder.cli.clean_checkpoints -s train7
    python -m radfinder.cli.clean_checkpoints -s train7 --write
"""

import re
from pathlib import Path

from attr import define
from loguru import logger
from radfinder.paths import get_medv_output_dir

from packg.log import configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument

CKPT_PATTERN = re.compile(r"checkpoint_epoch_(\d+)_meanr_([\d.]+)\.pt$")


@define
class Args(VerboseQuietArgs):
    subfolder: str = add_argument(
        shortcut="-s", required=True, help="Subfolder under MEDV_OUTPUT_DIR"
    )
    write: bool = add_argument(action="store_true", help="Actually delete files (default: dry run)")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(get_logger_level_from_args(args))
    clean_checkpoints(args)


def clean_checkpoints(args: Args):
    base_dir = get_medv_output_dir() / args.subfolder
    assert base_dir.is_dir(), f"Directory does not exist: {base_dir}"

    # Find experiment dirs by locating checkpoint.pt
    exp_dirs = sorted({p.parent for p in base_dir.rglob("checkpoint.pt") if ".rsync" not in str(p)})
    print(f"Found {len(exp_dirs)} experiments in {base_dir}")

    total_delete = 0
    total_bytes = 0

    for exp_dir in exp_dirs:
        ckpts = sorted(exp_dir.glob("checkpoint_epoch_*.pt"))
        parsed = []
        for p in ckpts:
            m = CKPT_PATTERN.match(p.name)
            if m is None:
                logger.warning(f"Cannot parse: {p.name}")
                continue
            parsed.append((int(m.group(1)), float(m.group(2)), p))

        if len(parsed) <= 2:
            continue

        best = min(parsed, key=lambda x: x[1])
        last = max(parsed, key=lambda x: x[0])
        keep = {best[2], last[2]}

        to_delete = [p for _, _, p in parsed if p not in keep]
        if not to_delete:
            continue

        delete_bytes = sum(p.stat().st_size for p in to_delete)
        total_delete += len(to_delete)
        total_bytes += delete_bytes

        label = "DELETE" if args.write else "would delete"
        print(
            f"\n{exp_dir.name}: {len(parsed)} checkpoints, "
            f"keep best=epoch {best[0]} (meanr {best[1]:.2f}), last=epoch {last[0]}"
        )
        for p in to_delete:
            size_mb = p.stat().st_size / 1024**2
            print(f"  {label}: {p.name} ({size_mb:.0f} MB)")
            if args.write:
                p.unlink()

    print(f"\nTotal: {total_delete} files, {total_bytes / 1024**3:.1f} GB", end="")
    if not args.write:
        print(" (dry run, use --write to delete)")
    else:
        print(" deleted")


if __name__ == "__main__":
    main()
