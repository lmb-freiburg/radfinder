"""HuggingFace-export loading for radfinder.

Two entry points, both opt-in via a `huggingface_path` field in a model config:

- `resolve_huggingface_path(conf)`: turns a model config that points at an exported
  HF model into a self-contained config dict (baked `radfinder_model_config` +
  `do_snippet_alignment` / `model_settings` + a `radfinder_hf_weights` marker),
  with any remaining YAML fields merged on top. Called from
  `utils.config.load_config_without_types`.
- `load_siglip_from_hf_weights(model_config, ...)`: rebuilds the SigLIP architecture
  and loads the exported safetensors. Called from `load_model.create_siglip` when it
  sees the `radfinder_hf_weights` marker.
"""

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from radfinder.models.load_model import create_siglip
from radfinder.models.vision_language import SigLIP
from radfinder.utils.logging_utils import log_info


def resolve_huggingface_path(conf: DictConfig) -> DictConfig:
    """Resolve a model config that points at an exported HF model.

    Reads the exported `config.json` (its `radfinder_model_config` becomes the base
    config, with `do_snippet_alignment` / `model_settings` carried along and
    `radfinder_hf_weights` set to the resolved dir), then merges the remaining YAML
    fields on top so they can override the HF config.
    """
    hf_path = conf.get("huggingface_path", None)
    if hf_path is None:
        return conf

    hf_dir = _resolve_hf_model_dir(hf_path)
    hf_config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    base = OmegaConf.create(hf_config["radfinder_model_config"])
    base.do_snippet_alignment = hf_config.get("do_snippet_alignment")
    base.model_settings = hf_config.get("model_settings")
    base.radfinder_hf_weights = hf_dir.as_posix()
    overrides = OmegaConf.create({k: v for k, v in conf.items() if k != "huggingface_path"})
    return OmegaConf.merge(base, overrides)


def _resolve_hf_model_dir(hf_path: str) -> Path:
    """`hf-hub:<repo_id>` downloads from the Hub; anything else is an offline dir."""
    hub_prefix = "hf-hub:"
    if hf_path.startswith(hub_prefix):
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=hf_path[len(hub_prefix) :]))
    path = Path(hf_path)
    if not path.exists():
        raise FileNotFoundError(f"huggingface_path offline model dir not found: {path}")
    return path


def load_siglip_from_hf_weights(model_config, image_feat_mode, text_feat_mode) -> SigLIP:
    """Rebuild a SigLIP from an exported HF model and load its safetensors weights.

    `model_config` is the resolved config from `resolve_huggingface_path`: the baked
    radfinder model config plus three extra keys — `radfinder_hf_weights` (path to the
    exported HF dir or a single model.safetensors), `do_snippet_alignment` and
    `model_settings`.
    """
    weights_path = Path(model_config["radfinder_hf_weights"])

    marker_keys = {"radfinder_hf_weights", "do_snippet_alignment", "model_settings"}
    component_config = {k: v for k, v in model_config.items() if k not in marker_keys}

    log_info(f"Building SigLIP architecture for exported HF weights from {weights_path}")
    model = create_siglip(
        component_config,
        image_feat_mode=image_feat_mode,
        text_feat_mode=text_feat_mode,
        skip_component_weights=True,  # weights come from the exported safetensors below
        do_snippet_alignment=model_config.get("do_snippet_alignment"),
        model_settings=model_config.get("model_settings"),
    )
    model.load_checkpoint(weights_path)
    model.eval()
    return model
