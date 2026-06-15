"""
Filesystem paths. Set the env vars or hardcode the paths into the functions.
"""

import os
from pathlib import Path

__version__ = "0.1.0"

RADFINDER_REPO_DIR: Path = Path(__file__).parent.parent.parent
RATE_CONFIG_DIR: Path = RADFINDER_REPO_DIR / "configs/rate"
PROMPTS_DIR: Path = RATE_CONFIG_DIR / "modalities_en/prompts_abdomen_chest_ct"


def _get_env_var_or_fail(var_name: str) -> Path:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(f"{var_name} env var is not set")
    return value


def get_medv_data_dir() -> Path:
    value = _get_env_var_or_fail("MEDV_DATA_DIR")
    return Path(value)


def get_medv_output_dir() -> Path:
    value = _get_env_var_or_fail("MEDV_OUTPUT_DIR")
    return Path(value)
