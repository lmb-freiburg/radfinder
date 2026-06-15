from types import SimpleNamespace

import pytest
from radfinder.utils import logging_utils


def _clear_rank_env(monkeypatch):
    for key in ("RANK", "LOCAL_RANK", "SLURM_JOB_NAME", "SLURM_NTASKS", "SLURM_PROCID"):
        monkeypatch.delenv(key, raising=False)


def test_configure_spectre_logging_uses_main_level_on_rank_zero(monkeypatch):
    _clear_rank_env(monkeypatch)

    config = logging_utils.configure_logging(args=None, main_level="DEBUG")

    assert config["handlers"][0]["level"] == "DEBUG"


def test_configure_spectre_logging_uses_error_level_on_non_main_rank(monkeypatch):
    _clear_rank_env(monkeypatch)
    monkeypatch.setenv("RANK", "1")

    config = logging_utils.configure_logging(args=None, main_level="DEBUG")

    assert config["handlers"][0]["level"] == "ERROR"


def test_configure_spectre_logging_uses_error_level_on_slurm_non_main_rank(monkeypatch):
    _clear_rank_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NAME", "spectre_train")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("SLURM_PROCID", "1")

    config = logging_utils.configure_logging(args=None, main_level="DEBUG")

    assert config["handlers"][0]["level"] == "ERROR"


def test_configure_spectre_logging_uses_cli_level_on_main_rank(monkeypatch):
    _clear_rank_env(monkeypatch)
    args = SimpleNamespace(verbose=True, quiet=False, loglevel=None)

    config = logging_utils.configure_logging(args=args, main_level="ERROR")

    assert config["handlers"][0]["level"] == "DEBUG"


def test_log_once_emits_once(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        logging_utils.logger,
        "log",
        lambda level, message: emitted.append((level, message)),
    )
    logging_utils._LOG_ONCE_KEYS.clear()

    logging_utils.log_once("same-key", "first", level="WARNING")
    logging_utils.log_once("same-key", "second", level="WARNING")

    assert emitted == [("WARNING", "first")]


def test_log_wrappers_reject_extra_positional_args():
    with pytest.raises(TypeError):
        logging_utils.log_info("first", "silently-dropped-by-loguru")  # type: ignore[call-arg]
