from copy import deepcopy
from pathlib import Path

import torch
from radfinder.losses.localization_loss import GaussianLocalizationLoss
from radfinder.losses.siglip_loss import SigLIPLoss
from radfinder.models.siglip_head import SigLIPProjectionHead
from radfinder.models.vision_language import MaskUsageC, SigLIP, SnippetAlignmentModeC
from radfinder.models.vision_transformer import VisionTransformer, get_default_snippet_alignment
from radfinder.models.vision_transformer_features import FeatureVisionTransformer
from radfinder.paths import RADFINDER_REPO_DIR, get_medv_output_dir
from radfinder.utils.logging_utils import log_debug, log_info, log_warning
from radfinder.utils.lora import add_lora_adapters
from transformers import Qwen3Config, Qwen3Model

from packg.constclass import Const
from packg.typext import PathType

DEFAULT_MODEL_CONFIG_FILE = RADFINDER_REPO_DIR / "configs/models/radfinder.yaml"
DEFAULT_TRAINING_CONFIG_FILE = RADFINDER_REPO_DIR / "configs/runs/training/train_radfinder_3mm.yaml"


def create_siglip_from_train_cfg(model_config, train_config):
    model_config = apply_train_cfg_to_model_config(model_config, train_config)
    image_feat_mode = train_config["train"]["image_feat_mode"]
    text_feat_mode = train_config["train"]["text_feat_mode"]
    # currently: feature training is on full scans (variable image shapes)
    # image training is on fixed crops (fixed image shapes)
    mask_usage_train = (
        MaskUsageC.TRUE
        if image_feat_mode in {FeatMode.FROZEN_LOCAL, FeatMode.FROZEN_GLOBAL}
        else MaskUsageC.FALSE
    )

    #################### Load init checkpoint maybe ####################
    init_ckpt = train_config["train"].get("init_ckpt")
    if init_ckpt is not None:
        log_info(f"Will load init checkpoint from {init_ckpt}")

    model = create_siglip(
        model_config,
        image_feat_mode=image_feat_mode,
        text_feat_mode=text_feat_mode,
        mask_usage_train=mask_usage_train,
        train_config=train_config,
        init_ckpt=init_ckpt,
    )
    model.set_frozen_state(model_config, train_config, is_init=True)
    return model, image_feat_mode, text_feat_mode


def apply_train_cfg_to_model_config(model_config: dict, train_config: dict) -> dict:
    """Apply train-config model overrides to a base model config without constructing the model."""
    model_config = deepcopy(model_config)
    for overwrite_k in ("backbone_kwargs", "feature_combiner_kwargs", "text_backbone_kwargs"):
        overwrite_kwargs = train_config["model"].get(overwrite_k, {})
        for k, v in overwrite_kwargs.items():
            model_config[overwrite_k][k] = v
            log_info(f"Set model config['{overwrite_k}']['{k}'] = {v} from train config")
    model_config["backbone_kwargs"]["grad_checkpoint_every"] = train_config["optim"][
        "image_backbone_grad_checkpoint_every"
    ]
    model_config["text_backbone_kwargs"]["grad_checkpoint"] = train_config["optim"][
        "text_backbone_grad_checkpoint"
    ]
    return model_config


def resolve_arch_settings(
    model_config: dict, train_config: dict | None = None
) -> tuple[dict, dict | None]:
    """Resolve (do_snippet_alignment, model_settings) for building/eval.

    Precedence: model_config (a self-contained config, e.g. an HF-exported model,
    carries these directly) > train_config `train.*` >
    default. The same resolved pair feeds both the model build and the dataloader
    so they can never disagree.
    """
    train_section = (train_config or {}).get("train", {})
    do_snippet = model_config.get("do_snippet_alignment")
    if do_snippet is None:
        do_snippet = train_section.get("do_snippet_alignment", get_default_snippet_alignment())
    model_settings = model_config.get("model_settings")
    if model_settings is None:
        model_settings = train_section.get("model_settings")
    return do_snippet, model_settings


def create_siglip(
    model_config,
    image_feat_mode,
    text_feat_mode,
    mask_usage_train=MaskUsageC.DYNAMIC,
    train_config: dict | None = None,
    init_ckpt: PathType | None = None,
    skip_component_weights: bool = False,
    do_snippet_alignment: dict | None = None,
    model_settings: dict | None = None,
) -> SigLIP:
    # inference entry point: a model config can point at an exported HF model,
    # in which case we rebuild the architecture and load the exported weights.
    if model_config.get("radfinder_hf_weights") is not None:

        # keep import here to avoid circular import
        from radfinder.models.load_model_hf import load_siglip_from_hf_weights

        return load_siglip_from_hf_weights(model_config, image_feat_mode, text_feat_mode)

    image_backbone, image_feature_comb, image_projection, text_backbone, text_projection = (
        create_spectre(
            model_config,
            image_feat_mode=image_feat_mode,
            text_feat_mode=text_feat_mode,
            skip_component_weights=skip_component_weights,
        )
    )
    log_debug(f"{type(image_backbone)=}")
    log_debug(f"{type(image_feature_comb)=}")
    log_debug(f"{type(image_projection)=}")
    log_debug(f"{type(text_backbone)=}")
    log_debug(f"{type(text_projection)=}")

    if train_config is None:
        # during eval these values are hardcoded for now
        train_config = dict(
            model=dict(
                learnable_t=True,
                learnable_b=True,
                normalize=True,
                init_t=2.3026,  # ~log(10)
                init_b=-10.0,
            )
        )
    criterion = SigLIPLoss(
        learnable_t=train_config["model"]["learnable_t"],
        learnable_b=train_config["model"]["learnable_b"],
        normalize=train_config["model"]["normalize"],
        init_t=train_config["model"]["init_t"],
        init_b=train_config["model"]["init_b"],
    )

    # Axis localization setup. Explicit args (passed by the inference wrapper)
    # win; otherwise resolve from model_config then train_config (precedence
    # documented in resolve_arch_settings).
    resolved_snippet, resolved_settings = resolve_arch_settings(model_config, train_config)
    if do_snippet_alignment is None:
        do_snippet_alignment = resolved_snippet
    if model_settings is None:
        model_settings = resolved_settings
    do_snippet = do_snippet_alignment
    if do_snippet.get("dual_cls_token", False):
        image_feature_comb.init_local_cls_token()
        log_info("Initialized local_cls_token from pretrained global CLS (dual-CLS mode)")

    is_axis_loc = do_snippet.get("snippet_mode") == SnippetAlignmentModeC.AXIS_LOCALIZATION
    loc_criterion = None
    if is_axis_loc:
        assert (
            image_feature_comb is not None
        ), "image_feature_comb must be provided for axis localization"
        if not do_snippet.get("axis2_use_cls_input", False):
            image_feature_comb.init_axis_patch_proj()
            log_info("Initialized axis_patch_proj from pretrained patch_proj avg-half")
        else:
            log_info(
                f"axis2_use_cls_input=True: using pretrained patch_proj, skipping axis_patch_proj"
            )
        log_info(
            f"Created loc_text_proj: Linear({model_config['text_hidden_size']}, "
            f"{model_config['feature_comb_embed_dim']})"
        )
        learnable_tau = do_snippet.get("localization_learnable_tau", False)
        loc_criterion = GaussianLocalizationLoss(
            sigma=do_snippet.get("localization_sigma", 2.0),
            tau=do_snippet.get("localization_tau", 0.1),
            learnable_tau=learnable_tau,
        )
        log_info(
            f"Created GaussianLocalizationLoss(sigma={loc_criterion.sigma}, "
            f"tau={do_snippet.get('localization_tau', 0.1)}, {learnable_tau=})"
        )
    model = SigLIP(
        image_backbone=image_backbone,
        image_feature_comb=image_feature_comb,
        image_projection=image_projection,
        text_backbone=text_backbone,
        text_projection=text_projection,
        criterion=criterion,
        loc_criterion=loc_criterion,
        mask_usage_train=mask_usage_train,
        model_settings=model_settings,
        do_snippet_alignment=do_snippet,
    )
    if init_ckpt is not None:
        log_info(f"Loading initial checkpoint from {init_ckpt}")
        missing_keys: list[str] = model.load_checkpoint(init_ckpt)

        # some modules are copied from other modules' weights. now there are 2 scenarios.
        if len(missing_keys) == 0:
            # 1) the checkpoint we just loaded contains separate weights for original and copied
            # module. in that case everything is fine.
            pass
        else:
            # 2) checkpoint only contains weights for the original module, and not the copied one.
            # so now we have the "wrong copy" because the copy is a copy of the pretrained weights,
            # but the original is from the checkpoint, so we have to update the copies as well.
            fixed = set()
            for k in missing_keys:
                if k.startswith("feature_comb_image.axis_patch_proj."):
                    if "axis_patch_proj" not in fixed:
                        model.feature_comb_image.init_axis_patch_proj()
                        fixed.add("axis_patch_proj")
                    continue
                for module, func in (
                    ("projection_image_copy", model._update_projection_image_copy_weights),
                    ("projection_text_copy", model._update_projection_text_copy_weights),
                ):
                    if k.startswith(f"{module}."):
                        if module not in fixed:
                            func()
                            fixed.add(module)
                        break
                else:
                    raise ValueError(f"Missing key {k} and no idea what to do with it.")
            log_info(f"Fixed copies of modules: {sorted(fixed)}")

    model.eval()
    return model


class FeatMode(Const):
    FULL = "full"  # run from scans
    FROZEN_LOCAL = "frozen_local"  # frozen local features + global network + projector
    FROZEN_GLOBAL = "frozen_global"  # frozen global features + projector
    NONE = "none"  # do not run this branch
    RETURN_SPACED_IMAGE = "return_spaced_image"  # return the spaced image and do not run the model
    FROM_SPACED_IMAGE = "from_spaced_image"  # run the model from the spaced image


def create_spectre(
    spectre_model_config,
    image_feat_mode=FeatMode.FULL,
    text_feat_mode=FeatMode.FULL,
    skip_component_weights: bool = False,
) -> tuple:
    """Create SPECTRE model components based on feature extraction mode.

    Returns:
        tuple: (image_backbone, image_feature_comb, image_projection, text_backbone, text_projection)
               Components not needed for the specified mode are returned as None.
    """
    log_info(
        f"## Creating SPECTRE with {image_feat_mode=}, {text_feat_mode=}, {skip_component_weights=}"
    )
    if skip_component_weights:
        # build modules with fresh weights: a full state_dict will overwrite them anyway,
        # so blank the 5 component-weight paths to avoid the wasted SPECTRE download/load
        spectre_model_config = deepcopy(spectre_model_config)
        for weight_key in (
            "backbone_checkpoint_path_or_url",
            "feature_combiner_checkpoint_path_or_url",
            "image_projection_weights",
            "text_weights",
            "text_projection_weights",
        ):
            spectre_model_config[weight_key] = None

    image_backbone = None
    if image_feat_mode in {FeatMode.FULL, FeatMode.FROM_SPACED_IMAGE}:
        image_backbone = create_image_backbone(spectre_model_config)
        image_backbone = image_backbone.eval()

    image_feature_comb = None
    # image feature comb is needed for downstream tasks so don't skip it
    # if image_feat_mode in {FeatMode.FULL, FeatMode.FROM_SPACED_IMAGE, FeatMode.FROZEN_LOCAL}:
    image_feature_comb = create_image_feature_comb(spectre_model_config)
    image_feature_comb = image_feature_comb.eval()

    image_projection = None
    if image_feat_mode in {
        FeatMode.FULL,
        FeatMode.FROM_SPACED_IMAGE,
        FeatMode.FROZEN_LOCAL,
        FeatMode.FROZEN_GLOBAL,
    }:
        image_projection = create_image_projection(spectre_model_config)
        image_projection = image_projection.eval()

    # text backbone is needed for various downstream tasks so never skip it
    text_backbone = create_text_backbone(spectre_model_config)
    text_backbone = text_backbone.eval()

    text_projection = None
    if text_feat_mode in {FeatMode.FULL, FeatMode.FROZEN_LOCAL, FeatMode.FROZEN_GLOBAL}:
        text_projection = create_text_projection(spectre_model_config)
        text_projection = text_projection.eval()

    return image_backbone, image_feature_comb, image_projection, text_backbone, text_projection


def get_spectre_weights_path(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    if path.is_absolute():
        # file is absolute and was not found
        raise FileNotFoundError(f"File {path} not found")
    # path is relative, download the spectre weights from huggingface into the
    # default HF hub cache (respects HF_HOME / HF_HUB_CACHE) and resolve against it
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    repo_id = "cclaess/SPECTRE"
    # print(f"Downloading spectre weights from huggingface repo {repo_id} for {path}")
    disable_progress_bars()
    hf_dir = Path(snapshot_download(repo_id=repo_id))
    enable_progress_bars()
    weights_path = hf_dir / path
    if not weights_path.exists():
        raise FileNotFoundError(
            f"File {path} not found in downloaded SPECTRE repo at {hf_dir}. "
            f"Available files: {sorted(p.name for p in hf_dir.iterdir())}"
        )
    return weights_path


def create_image_backbone(spectre_model_config):
    """Create and load the image backbone model."""
    weights_path = spectre_model_config["backbone_checkpoint_path_or_url"]
    backbone_kwargs = spectre_model_config["backbone_kwargs"]

    if weights_path is not None:
        weights_path = get_spectre_weights_path(weights_path)
        image_backbone = VisionTransformer.from_pretrained(
            checkpoint_path_or_url=weights_path, **backbone_kwargs
        )
    else:
        image_backbone = VisionTransformer(**backbone_kwargs)
        # post weight updates must also be applied to random weights for a consistent state dict
        image_backbone.post_weight_loading()
    return image_backbone


def create_image_feature_comb(spectre_model_config):
    """Create and load the image feature combiner model."""

    feature_combiner_kwargs = spectre_model_config["feature_combiner_kwargs"]
    feature_combiner_init_kwargs = dict(
        patch_dim=spectre_model_config["image_backbone_embed_dim"] * 2,
        num_classes=feature_combiner_kwargs["num_classes"],
        global_pool=feature_combiner_kwargs["global_pool"],
        embed_dim=spectre_model_config["feature_comb_embed_dim"],
        depth=spectre_model_config["feature_comb_num_layers"],
        num_heads=spectre_model_config["feature_comb_num_heads"],
        pos_embed=feature_combiner_kwargs["pos_embed"],
        rope_kwargs=feature_combiner_kwargs["rope_kwargs"],
        init_values=feature_combiner_kwargs["init_values"],
    )
    image_feature_comb = FeatureVisionTransformer(
        **feature_combiner_init_kwargs, stored_init_kwargs=feature_combiner_init_kwargs
    )
    weights_path = spectre_model_config["feature_combiner_checkpoint_path_or_url"]

    if weights_path is not None:
        weights_path = get_spectre_weights_path(weights_path)
        image_feature_comb.load_state_dict(
            torch.load(
                weights_path,
                map_location="cpu",
                weights_only=False,
            ),
            strict=True,
        )
    return image_feature_comb


def create_image_projection(spectre_model_config):
    """Create and load the image projection model."""
    image_projection = SigLIPProjectionHead(
        input_dim=spectre_model_config["feature_comb_embed_dim"] * 2,  # cls token + avg pooling
        output_dim=spectre_model_config["projection_dim"],
    )
    weights_path = spectre_model_config["image_projection_weights"]
    if weights_path is not None:
        weights_path = get_spectre_weights_path(weights_path)
        image_projection.load_state_dict(
            torch.load(
                weights_path,
                map_location="cpu",
                weights_only=False,
            ),
            strict=True,
        )
    return image_projection


def create_text_backbone(spectre_model_config):
    """Create and load the text backbone model."""
    qwen_config = {
        "_attn_implementation_autoset": True,
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151643,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 32768,
        "max_window_layers": 28,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "sliding_window": None,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 151669,
    }
    text_backbone = Qwen3Model(Qwen3Config.from_dict(qwen_config))
    text_backbone_kwargs = spectre_model_config["text_backbone_kwargs"]
    if text_backbone_kwargs["grad_checkpoint"]:
        log_info("Text backbone: Enabling gradient checkpointing")
        text_backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        text_backbone.config.use_cache = False
        # not sure if these 2 lines are still necessary but they should not hurt either way
        text_backbone.enable_input_require_grads()
        text_backbone.get_input_embeddings().register_forward_hook(_require_grad_on_embed_out)

    if text_backbone_kwargs["use_lora"] and text_backbone_kwargs["lora_r"] > 0:
        add_lora_adapters(
            text_backbone,
            r=text_backbone_kwargs["lora_r"],
            lora_alpha=text_backbone_kwargs["lora_alpha"],
            lora_dropout=text_backbone_kwargs["lora_dropout"],
            target_keywords=text_backbone_kwargs["lora_target_keywords"],
        )
    weights_path = spectre_model_config["text_weights"]
    if weights_path is not None:
        weights_path = get_spectre_weights_path(weights_path)
        text_backbone.load_state_dict(
            torch.load(
                weights_path,
                map_location="cpu",
                weights_only=False,
            ),
            strict=True,
        )
    return text_backbone


def _require_grad_on_embed_out(module, inp, out):
    out.requires_grad_(True)
    return out


def create_text_projection(spectre_model_config):
    """Create and load the text projection model."""
    text_projection = SigLIPProjectionHead(
        input_dim=spectre_model_config["text_hidden_size"],
        output_dim=spectre_model_config["projection_dim"],
    )
    weights_path = spectre_model_config["text_projection_weights"]
    if weights_path is not None:
        weights_path = get_spectre_weights_path(weights_path)
        text_projection.load_state_dict(
            torch.load(
                weights_path,
                map_location="cpu",
                weights_only=False,
            ),
            strict=True,
        )
    return text_projection
