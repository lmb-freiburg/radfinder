import numpy as np
import pytest
from radfinder.tasks.bootstrap_retrieval import (
    bootstrap_binary_zs,
    bootstrap_localization,
    bootstrap_standard_retrieval,
    bootstrap_volume_retrieval,
    compute_map_at_k_per_sample,
    pool_retrieval_ci,
)
from radfinder.utils.bootstrap import bootstrap_ci, bootstrap_ci_multi, ci_from_repeats


def test_bootstrap_ci_mean_known_distribution():
    rng = np.random.default_rng(123)
    data = rng.normal(loc=5.0, scale=1.0, size=1000)
    point, lo, hi = bootstrap_ci(data, np.mean, n_bootstrap=5000, seed=0)
    assert abs(point - 5.0) < 0.1
    assert lo < point < hi
    assert hi - lo < 0.3  # CI should be tight for N=1000


def test_bootstrap_ci_binary():
    data = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 1], dtype=np.float64)
    point, lo, hi = bootstrap_ci(data, np.mean, n_bootstrap=5000, seed=42)
    assert point == pytest.approx(0.6)
    assert 0.2 <= lo <= 0.6
    assert 0.6 <= hi <= 1.0


def test_bootstrap_ci_multi_shared_resampling():
    rng = np.random.default_rng(99)
    data = rng.normal(loc=10.0, scale=2.0, size=500)
    results = bootstrap_ci_multi(
        data,
        {"mean": np.mean, "median": np.median},
        n_bootstrap=3000,
        seed=42,
    )
    assert "mean" in results and "median" in results
    for name in ("mean", "median"):
        point, lo, hi = results[name]
        assert lo < point < hi
        assert abs(point - 10.0) < 0.5


def test_bootstrap_ci_reproducibility():
    data = np.random.default_rng(0).normal(size=100)
    r1 = bootstrap_ci(data, np.mean, seed=42)
    r2 = bootstrap_ci(data, np.mean, seed=42)
    assert r1 == r2


def test_ci_from_repeats():
    values = list(range(100))
    mean, lo, hi = ci_from_repeats(values, ci=0.95)
    assert mean == pytest.approx(49.5)
    assert lo < mean < hi
    assert lo == pytest.approx(2.475, abs=0.01)
    assert hi == pytest.approx(96.525, abs=0.01)


def test_bootstrap_standard_retrieval():
    rng = np.random.default_rng(7)
    ranks = rng.integers(0, 200, size=500).astype(np.float64)
    result = bootstrap_standard_retrieval(ranks, n_bootstrap=2000, seed=42)
    assert result["n"] == 500
    for k in (1, 5, 10, 50, 100):
        key = f"r{k}"
        assert key in result
        assert f"{key}_ci_lo" in result
        assert f"{key}_ci_hi" in result
        assert f"{key}_ci_half" in result
        assert result[f"{key}_ci_lo"] <= result[key] <= result[f"{key}_ci_hi"]
        assert result[f"{key}_ci_half"] >= 0
    assert result["medr_ci_lo"] <= result["medr"] <= result["medr_ci_hi"]
    assert result["meanr_ci_lo"] <= result["meanr"] <= result["meanr_ci_hi"]


def test_bootstrap_standard_retrieval_perfect():
    ranks = np.zeros(100, dtype=np.float64)
    result = bootstrap_standard_retrieval(ranks, n_bootstrap=1000, seed=42)
    assert result["r1"] == 1.0
    assert result["r1_ci_lo"] == 1.0
    assert result["r1_ci_hi"] == 1.0


def test_compute_map_at_k_per_sample():
    n = 10
    sim = np.eye(n, dtype=np.float32)
    labels = np.eye(n, dtype=np.int32)
    ap_scores = compute_map_at_k_per_sample(sim, labels, k=5)
    assert ap_scores.shape == (n,)
    assert all(s > 0 for s in ap_scores)


def test_pool_retrieval_ci_smoke():
    rng = np.random.default_rng(42)
    n = 64
    image_emb = rng.standard_normal((n, 32)).astype(np.float32)
    text_emb = rng.standard_normal((n, 32)).astype(np.float32)
    result = pool_retrieval_ci(
        image_emb,
        text_emb,
        pool_sizes=[32],
        ks=[1],
        repeats=20,
        seed=42,
        ci=0.95,
    )
    key = "pool32_r1"
    assert key in result
    assert f"{key}_ci_lo" in result
    assert f"{key}_ci_hi" in result
    assert result[f"{key}_ci_lo"] <= result[key] <= result[f"{key}_ci_hi"]


def test_bootstrap_binary_zs():
    rng = np.random.default_rng(42)
    n = 200
    n_classes = 5
    labels = rng.integers(0, 2, size=(n, n_classes)).astype(np.int32)
    for c in range(n_classes):
        if labels[:, c].sum() == 0:
            labels[0, c] = 1
        if labels[:, c].sum() == n:
            labels[0, c] = 0
    predictions = rng.random((n, n_classes)).astype(np.float32)
    pathologies = [f"class_{i}" for i in range(n_classes)]
    result = bootstrap_binary_zs(
        predictions,
        labels,
        pathologies=pathologies,
        n_bootstrap=500,
        seed=42,
    )
    assert result["n"] == n
    assert result["n_bootstrap"] == 500
    for m in ["mean_auroc", "mean_prec", "mean_f1", "mean_acc"]:
        assert m in result
        assert f"{m}_ci_lo" in result
        assert f"{m}_ci_hi" in result
        assert f"{m}_ci_half" in result
        assert result[f"{m}_ci_lo"] <= result[m] <= result[f"{m}_ci_hi"]
        assert result[f"{m}_ci_half"] >= 0


def test_bootstrap_binary_zs_reproducibility():
    rng = np.random.default_rng(7)
    n, n_classes = 100, 3
    labels = rng.integers(0, 2, size=(n, n_classes)).astype(np.int32)
    for c in range(n_classes):
        if labels[:, c].sum() == 0:
            labels[0, c] = 1
        if labels[:, c].sum() == n:
            labels[0, c] = 0
    predictions = rng.random((n, n_classes)).astype(np.float32)
    pathologies = [f"c{i}" for i in range(n_classes)]
    r1 = bootstrap_binary_zs(predictions, labels, pathologies=pathologies, n_bootstrap=200, seed=99)
    r2 = bootstrap_binary_zs(predictions, labels, pathologies=pathologies, n_bootstrap=200, seed=99)
    for key in r1:
        assert r1[key] == r2[key], f"Mismatch for {key}: {r1[key]} != {r2[key]}"


def test_bootstrap_localization():
    rng = np.random.default_rng(42)
    n = 500
    pred = rng.integers(0, 32, size=n)
    target = rng.integers(0, 32, size=n)
    result = bootstrap_localization(pred, target, n_bootstrap=2000, seed=42)
    assert result["n_snippets"] == n
    assert result["n_bootstrap"] == 2000
    for m in [
        "loc_mae_mm",
        "loc_median_mm",
        "loc_mae_positions",
        "loc_acc_exact",
        "loc_acc_within_12mm",
        "loc_acc_within_24mm",
    ]:
        assert m in result
        assert f"{m}_ci_lo" in result
        assert f"{m}_ci_hi" in result
        assert f"{m}_ci_half" in result
        assert result[f"{m}_ci_lo"] <= result[m] <= result[f"{m}_ci_hi"]
        assert result[f"{m}_ci_half"] >= 0


def test_bootstrap_localization_reproducibility():
    rng = np.random.default_rng(7)
    n = 200
    pred = rng.integers(0, 20, size=n)
    target = rng.integers(0, 20, size=n)
    r1 = bootstrap_localization(pred, target, n_bootstrap=500, seed=99)
    r2 = bootstrap_localization(pred, target, n_bootstrap=500, seed=99)
    for key in r1:
        assert r1[key] == r2[key], f"Mismatch for {key}: {r1[key]} != {r2[key]}"


def test_bootstrap_volume_retrieval_keys():
    rng = np.random.default_rng(10)
    n = 30
    sim = rng.standard_normal((n, n)).astype(np.float32)
    np.fill_diagonal(sim, -np.inf)
    labels = rng.integers(0, 2, size=(n, 8)).astype(np.int32)
    result = bootstrap_volume_retrieval(sim, labels, ks=(5, 10), n_bootstrap=200, seed=42)
    assert result["n"] == n
    for k in (5, 10):
        key = f"vol_map{k}"
        assert key in result
        assert f"{key}_ci_lo" in result
        assert f"{key}_ci_hi" in result
        assert f"{key}_ci_half" in result
        assert result[f"{key}_ci_lo"] <= result[key] <= result[f"{key}_ci_hi"]
