from collections.abc import Callable

import numpy as np

# Bootstrap defaults used across every `run_*_with_bootstrap` entry point and the
# trainer/eval CLIs. Override at the function-arg level if a one-off needs it.
DEFAULT_N_BOOTSTRAP = 10000
DEFAULT_BOOTSTRAP_CI = 0.95
DEFAULT_BOOTSTRAP_SEED = 42


def bootstrap_ci(
    per_sample_scores: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(per_sample_scores)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_metrics = np.array([metric_fn(per_sample_scores[idx[i]]) for i in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(boot_metrics, 100 * alpha))
    ci_high = float(np.percentile(boot_metrics, 100 * (1 - alpha)))
    return float(metric_fn(per_sample_scores)), ci_low, ci_high


def bootstrap_ci_multi(
    per_sample_data: np.ndarray,
    metric_fns: dict[str, Callable[[np.ndarray], float]],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, tuple[float, float, float]]:
    """
    Bootstrap multiple metrics from the same resampled indices.

    Returns {metric_name: (point_estimate, ci_low, ci_high)}.
    """
    rng = np.random.default_rng(seed)
    n = len(per_sample_data)
    idx = rng.integers(0, n, size=(n_bootstrap, n))

    boot_values: dict[str, np.ndarray] = {name: np.empty(n_bootstrap) for name in metric_fns}
    for i in range(n_bootstrap):
        sample = per_sample_data[idx[i]]
        for name, fn in metric_fns.items():
            boot_values[name][i] = fn(sample)

    alpha = (1 - ci) / 2
    results = {}
    for name, fn in metric_fns.items():
        point = float(fn(per_sample_data))
        lo = float(np.percentile(boot_values[name], 100 * alpha))
        hi = float(np.percentile(boot_values[name], 100 * (1 - alpha)))
        results[name] = (point, lo, hi)
    return results


def ci_from_repeats(
    values: list[float] | np.ndarray,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> tuple[float, float, float]:
    """Compute CI directly from repeated measurements (e.g. pool retrieval repeats)."""
    arr = np.asarray(values)
    alpha = (1 - ci) / 2
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 100 * alpha)),
        float(np.percentile(arr, 100 * (1 - alpha))),
    )
