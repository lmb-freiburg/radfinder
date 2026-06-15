from pathlib import Path

from radfinder.paths import get_medv_output_dir


def get_ckpt_from_train_cfg(train_cfg: Path) -> str:
    """Derive the default checkpoint path for a training config."""
    train_cfg = Path(train_cfg)
    ckpt = Path(get_medv_output_dir()) / train_cfg.parent.name / train_cfg.stem / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt.as_posix()


def parse_checkpoint(p: Path) -> tuple[int, float]:
    """Parse `checkpoint_epoch_<N>_meanr_<M>.pt` filenames into (epoch, meanr)."""
    parts = p.name.removesuffix(".pt").split("_")
    return int(parts[2]), float(parts[4])


def find_best_checkpoint(
    experiment_dir: Path, epoch_override: int | None = None
) -> tuple[int, Path]:
    """
    Return the (epoch, path) of the checkpoint with the lowest meanr.

    With `epoch_override`, returns the checkpoint for that specific epoch instead.
    """
    checkpoints = sorted(experiment_dir.glob("checkpoint_epoch_*.pt"))
    assert len(checkpoints) > 0, f"No checkpoint_epoch_*.pt files found in {experiment_dir}"

    if epoch_override is not None:
        candidates = [c for c in checkpoints if parse_checkpoint(c)[0] == epoch_override]
        assert (
            len(candidates) > 0
        ), f"No checkpoint found for epoch {epoch_override} in {experiment_dir}"
        epoch, meanr = parse_checkpoint(candidates[0])
        print(f"Using epoch {epoch} (meanr={meanr:.2f})")
        return epoch, candidates[0]

    best_ckpt = min(checkpoints, key=lambda p: parse_checkpoint(p)[1])
    epoch, meanr = parse_checkpoint(best_ckpt)
    print(f"Best epoch {epoch} (meanr={meanr:.2f})")
    return epoch, best_ckpt


def filter_best_last_checkpoints(checkpoints: list[Path]) -> list[Path]:
    """Filter to just the best (lowest-meanr) and last (highest-epoch) checkpoints."""
    if not checkpoints:
        return checkpoints
    parsed = [(p, *parse_checkpoint(p)) for p in checkpoints]
    best = min(parsed, key=lambda x: x[2])
    last = max(parsed, key=lambda x: x[1])
    keep = {best[0], last[0]}
    filtered = [p for p in checkpoints if p in keep]
    skipped = len(checkpoints) - len(filtered)
    print(
        f"Evaluating best (epoch {best[1]}, meanr={best[2]:.2f}) + last (epoch {last[1]}), "
        f"skipping {skipped} checkpoints"
    )
    return filtered
