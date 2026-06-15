import math
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from radfinder.paths import RADFINDER_REPO_DIR
from radfinder.utils import distributed
from radfinder.utils.logging_utils import log_info
from radfinder.utils.misc import find_path_rel_or_abs, fix_random_seeds

from visiontext.configutils import load_dotlist


def load_config_without_types(config_file, merge_dotlist: list[str] | None = None):
    config_file = find_path_rel_or_abs(config_file, RADFINDER_REPO_DIR)
    conf: DictConfig = OmegaConf.load(config_file.as_posix())
    if conf.get("huggingface_path", None) is not None:
        from radfinder.models.load_model_hf import resolve_huggingface_path

        conf = resolve_huggingface_path(conf)
    if merge_dotlist is not None:
        dict_dotlist = load_dotlist(merge_dotlist)
        # Check that all keys in dotlist already exist in the config
        check_nested_keys(dict_dotlist, conf)
        conf = OmegaConf.merge(conf, dict_dotlist)
    conf_dict: dict = OmegaConf.to_container(conf, resolve=False)
    return conf_dict


_MISSING = object()  # Sentinel value


def check_nested_keys(dotlist_dict, config, prefix=""):
    for key, value in dotlist_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        result = OmegaConf.select(config, full_key, default=_MISSING)
        if result is _MISSING:
            raise KeyError(
                f"Cannot override non-existent config key: '{full_key}'. "
                f"Key must exist in the config file before overriding."
            )
        if isinstance(value, dict):
            check_nested_keys(value, config, full_key)


def apply_scaling_rules_to_cfg(cfg: dict[str, Any]):
    """
    Apply learing rate scaling rules to the configuration object.
    """
    base_lr = cfg["optim"]["base_lr"]
    scaling_rule = cfg["optim"]["scaling_rule"]
    cfg["optim"]["lr"] = base_lr

    # Apply scaling rules
    if scaling_rule == "constant":
        return cfg

    try:
        scaling_type, ref_batch_size = scaling_rule.split("_wrt_")
        ref_batch_size = float(ref_batch_size)
    except ValueError:
        raise NotImplementedError(f"Unknown scaling rule: {scaling_rule}")

    global_batch_size = (
        cfg["train"]["batch_size"] * distributed.get_global_size() * cfg["train"]["accum_steps"]
    )
    scale_factor = global_batch_size / ref_batch_size

    final_lr = base_lr
    if scaling_type == "sqrt":
        final_lr *= math.sqrt(scale_factor)
    elif scaling_type == "linear":
        final_lr *= scale_factor
    else:
        raise NotImplementedError(f"Unsupported scaling type: {scaling_type}")
    cfg["optim"]["lr"] = final_lr
    log_info(
        f"Applied LR scaling: {base_lr=:.2e} {scaling_rule=} {global_batch_size=} "
        f"{ref_batch_size=} {scale_factor=:.2e} {final_lr=:.2e}"
    )
    return cfg


def random_seed(seed):
    rank = distributed.get_global_rank()

    fix_random_seeds(seed + rank)
