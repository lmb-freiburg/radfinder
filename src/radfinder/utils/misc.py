import json
import random
from itertools import repeat
from pathlib import Path
from typing import Iterable

import monai
import numpy as np
import torch
from safetensors.torch import load_file

from packg.typext import PathType


def load_safetensors_state(path: Path) -> dict:
    """
    Load a safetensors state dict from a single file or an exported HF directory.

    A directory may hold one `model.safetensors` or a sharded set described by
    `model.safetensors.index.json`.
    """
    if path.is_file():
        return load_file(path.as_posix())

    index_file = path / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text())["weight_map"]
        state: dict = {}
        for shard in sorted(set(weight_map.values())):
            state.update(load_file((path / shard).as_posix()))
        return state

    single = path / "model.safetensors"
    if single.exists():
        return load_file(single.as_posix())

    raise FileNotFoundError(f"No model.safetensors[.index.json] under {path}")


def simple_decap(in_str: str) -> str:
    if not in_str.isupper():
        return in_str
    sents = in_str.split(". ")
    return ". ".join(s.capitalize() for s in sents)


def equal_state_dicts(sd1, sd2) -> str | None:
    if not sd1.keys() == sd2.keys():
        return "keys mismatch"
    for k1, v1 in sd1.items():
        v2 = sd2[k1]
        if not torch.equal(v1, v2):
            return "values mismatch"
    return None


def find_path_rel_or_abs(path: PathType, base_dir: PathType):
    path = Path(path)
    if path.exists():
        return path
    if path.is_absolute():
        # file is absolute and was not found
        raise FileNotFoundError(f"File {path} not found")
    abs_path = Path(base_dir) / path
    if abs_path.exists():
        return abs_path
    raise FileNotFoundError(f"File not found, both relative {path} and absolute {abs_path}")


def fix_random_seeds(seed: int = 31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    monai.utils.set_determinism(seed=seed)


def _ntuple(n: int):
    """
    Helper function to create n-tuple.
    """

    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def noop(*args, **kwargs):
    pass


model_print = noop
