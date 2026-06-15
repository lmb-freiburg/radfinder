"""Tests for `radfinder.data.prompt_rate_labels`.

Hit the real disk: these tests require the RaTE question-map CSVs and the
`ctrate18_prompts.yaml` config. They lock down the public-API behavior of the
prompt-loading helpers and the empty-dataset code path of
`build_prompt_rate_labels`.
"""

from __future__ import annotations

import pytest
from radfinder.data.prompt_rate_labels import (
    PromptRateModeC,
    build_prompt_rate_labels,
    load_ctrate18_prompts,
    load_prompts_for_mode,
)
from radfinder.tasks.binary_zs_ctrate_task import PATHOLOGIES

# Use val split for speed (~10k scans vs ~78k for train).
VAL = "val"

# ─────────────────────────── load_ctrate18_prompts ─────────────────────────


def test_load_ctrate18_prompts_shape_and_content():
    names, pos, neg = load_ctrate18_prompts()
    # 18 CT-RATE pathologies, 3 variants per side.
    assert len(names) == 18
    assert names == list(PATHOLOGIES)
    assert len(pos) == 18
    assert len(neg) == 18
    for variants in pos + neg:
        assert len(variants) == 3
        assert all(isinstance(v, str) and len(v) > 0 for v in variants)
    # Spot-check: first pathology's first positive prompt mentions the pathology.
    assert "Medical material" in pos[0][0] or "thorax" in pos[0][0].lower()


# ─────────────────────────── load_prompts_for_mode ─────────────────────────


def test_load_prompts_for_mode_rate_returns_319_prompts():
    pos, neg = load_prompts_for_mode(PromptRateModeC.RATE)
    assert len(pos) == 319
    assert len(neg) == 319


def test_load_prompts_for_mode_ctrate_returns_18_prompts():
    pos, neg = load_prompts_for_mode(PromptRateModeC.CTRATE)
    assert len(pos) == 18
    assert len(neg) == 18


def test_load_prompts_for_mode_both_concatenates_to_337():
    pos_both, neg_both = load_prompts_for_mode(PromptRateModeC.BOTH)
    assert len(pos_both) == 319 + 18
    assert len(neg_both) == 319 + 18
    # Order: rate prompts first, ctrate prompts after.
    pos_rate, _ = load_prompts_for_mode(PromptRateModeC.RATE)
    pos_ctrate, _ = load_prompts_for_mode(PromptRateModeC.CTRATE)
    assert pos_both[:319] == pos_rate
    assert pos_both[319:] == pos_ctrate


def test_load_prompts_for_mode_unknown_raises():
    with pytest.raises(ValueError):
        load_prompts_for_mode("garbage")


# ─────────────────────────── build_prompt_rate_labels ──────────────────────


def test_build_prompt_rate_labels_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_prompt_rate_labels(dataset_names=[], split=VAL, mode="bogus")


def test_build_prompt_rate_labels_empty_dataset_names_returns_empty():
    """With no dataset_names, nothing should be loaded — empty result, no crashes."""
    for mode in (PromptRateModeC.RATE, PromptRateModeC.CTRATE, PromptRateModeC.BOTH):
        pr_labels, _ = build_prompt_rate_labels(dataset_names=[], split=VAL, mode=mode)
        assert pr_labels == {}, f"expected empty result for mode={mode}, got {len(pr_labels)} scans"
