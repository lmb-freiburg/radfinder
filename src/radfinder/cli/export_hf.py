"""Export the radfinder VL model to a Hugging Face Hub-ready directory.

Copies a curated subset of `src/radfinder/` into
`$MEDV_OUTPUT_DIR/radfinder_hf_export/radfinder/`, copies the HF wrapper files
from `hf_export/` alongside, builds the model from the picked YAML
config, loads the fine-tuned checkpoint, then `save_pretrained` to the export
dir. Optionally uploads to HF Hub.

Usage:
    python -m radfinder.cli.export_hf --config configs/models/spectre_pretrained_half_patch_embed.yaml --ckpt_file path/to/ckpt.pt
    python -m radfinder.cli.export_hf --config ... --ckpt_file ... --release --repo_id lmb-freiburg/radfinder
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

from attrs import define
from radfinder.paths import RADFINDER_REPO_DIR, get_medv_output_dir
from radfinder.utils.logging_utils import configure_logging, log_error, log_info

from typedparser import TypedParser, VerboseQuietArgs, add_argument

TEMPLATE_DIR = RADFINDER_REPO_DIR / "hf_export"
REPO_DIR = RADFINDER_REPO_DIR
SRC_PACKAGE = REPO_DIR / "src" / "radfinder"
EXPORT_DIR = get_medv_output_dir() / "radfinder_hf_export"
DEST_PACKAGE = EXPORT_DIR / "radfinder"

# Subset of `src/radfinder/` to ship in the HF repo. Tries to cover the
# inference path: backbone + feature combiner + image/text projections + LoRA
# text backbone wiring. Training/data/SSL stays out. Map name → True (copy as
# is) or list[str] (selective filenames; recursively copy listed children).
INCLUDE: dict = {
    "__init__.py": True,
    "paths.py": True,
    "loader_utils.py": True,
    # "save_embeddings_lib.py": True,  # for reference only; might be droppable later
    "models": True,
    "transforms": True,
    "utils": True,
    "tasks": True,
}


@define
class Args(VerboseQuietArgs):
    config: str = add_argument(
        help="Path to a radfinder model config YAML (e.g. configs/models/spectre_pretrained_half_patch_embed.yaml)",
    )
    ckpt_file: str | None = add_argument(
        help="Path to the fine-tuned checkpoint. If omitted, the model uses only the pretrained component checkpoints referenced in the YAML.",
    )
    train_cfg: str | None = add_argument(
        help="Train config the checkpoint was trained with. Its train.do_snippet_alignment / train.model_settings are baked into the exported config so from_pretrained rebuilds the exact architecture (copy projectors, axis_patch_proj, ...).",
    )
    release: bool = add_argument(
        action="store_true",
        help="Upload to HF Hub (default: False, dry-run only)",
    )
    repo_id: str = add_argument(
        default="lmb-freiburg/radfinder",
        help="HF repo ID (default: lmb-freiburg/radfinder)",
    )
    hf_token: str | None = add_argument(
        help="HF token (defaults to HF_TOKEN env var)",
    )
    skip_test: bool = add_argument(
        action="store_true",
        help="Skip local AutoModel.from_pretrained smoke test",
    )
    commit_message: str = add_argument(default="Initial commit")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logging(args)
    log_info(f"{args}")

    test_locally = not args.skip_test
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    if args.release and not hf_token:
        log_error("--release requires HF_TOKEN env var or --hf_token argument")
        sys.exit(1)

    log_info("=" * 60)
    log_info("Radfinder Export to HuggingFace")
    log_info("=" * 60)
    log_info(f"Export directory: {EXPORT_DIR}")
    log_info(f"Config YAML:      {args.config}")
    log_info(f"Checkpoint file:  {args.ckpt_file}")
    log_info(f"Repo ID:          {args.repo_id}")
    log_info(f"Test locally:     {test_locally}")
    log_info(f"Release:          {args.release}")
    log_info("=" * 60)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if DEST_PACKAGE.exists():
        log_info("Cleaning existing exported radfinder/ ...")
        shutil.rmtree(DEST_PACKAGE)

    log_info(f"Copying HF wrapper templates from {TEMPLATE_DIR} ...")
    for name in (
        "configuration_radfinder.py",
        "modeling_radfinder.py",
        "processing_radfinder.py",
        "metadata.yaml",
        "README.md",
    ):
        shutil.copy2(TEMPLATE_DIR / name, EXPORT_DIR / name)

    log_info("Copying radfinder package subset ...")
    _copy_selective(SRC_PACKAGE, DEST_PACKAGE, INCLUDE)
    log_info(f"  → wrote {DEST_PACKAGE}")

    sys.path.insert(0, str(EXPORT_DIR))

    from configuration_radfinder import RadFinderConfig
    from modeling_radfinder import RadFinderModel
    from processing_radfinder import RadFinderImageProcessor
    from radfinder.models.load_model import FeatMode, create_siglip
    from radfinder.models.vision_transformer import get_default_snippet_alignment
    from radfinder.utils.config import load_config_without_types

    RadFinderConfig.register_for_auto_class()
    RadFinderModel.register_for_auto_class("AutoModel")
    RadFinderImageProcessor.register_for_auto_class("AutoImageProcessor")

    log_info("Loading radfinder model config ...")
    radfinder_model_config = load_config_without_types(args.config)

    # The train config's `train.do_snippet_alignment` / `train.model_settings`
    # decide which extra modules the architecture gets (copy projectors,
    # axis_patch_proj, local_cls_token, global_res). Resolve them exactly the way
    # eval_retrieval does, so the build matches the checkpoint.
    do_snippet_alignment = None
    model_settings = None
    if args.train_cfg is not None:
        train_config = load_config_without_types(args.train_cfg)
        do_snippet_alignment = train_config["train"].get(
            "do_snippet_alignment", get_default_snippet_alignment()
        )
        model_settings = train_config["train"].get("model_settings")

    # The fine-tuned checkpoint only contains the trained subset of parameters;
    # the rest stay frozen at their pretrained SPECTRE values and are NOT in the
    # ckpt. So always load the full set of pretrained components first, then let
    # the checkpoint overwrite the trained subset on top.
    log_info("Building radfinder SigLIP from config ...")
    base = create_siglip(
        radfinder_model_config,
        image_feat_mode=FeatMode.FULL,
        text_feat_mode=FeatMode.FULL,
        skip_component_weights=False,
        do_snippet_alignment=do_snippet_alignment,
        model_settings=model_settings,
    )
    if args.ckpt_file is not None:
        base.load_checkpoint(args.ckpt_file)

    # Null out checkpoint paths before persisting the YAML into RadFinderConfig.
    # HF users won't have these files; the full state_dict ships in safetensors
    # and overwrites the randomly-initialised components on `from_pretrained`.
    for key in (
        "backbone_checkpoint_path_or_url",
        "feature_combiner_checkpoint_path_or_url",
        "text_weights",
        "image_projection_weights",
        "text_projection_weights",
    ):
        if key in radfinder_model_config:
            radfinder_model_config[key] = None

    log_info("Deriving HF config + processor ...")
    backbone_kwargs = radfinder_model_config.get("backbone_kwargs", {})
    image_preprocessing = {
        "pixdim": list(backbone_kwargs.get("pixdim", [0.75, 0.75, 3.0])),
        "sliding_window_size": list(backbone_kwargs.get("sliding_window_size", [128, 128, 32])),
        "intensity_a_min": -1000.0,
        "intensity_a_max": 1000.0,
        "intensity_b_min": 0.0,
        "intensity_b_max": 1.0,
        "orientation_axcodes": "RAS",
        "min_area_for_padding": float(backbone_kwargs.get("min_area_for_padding", 0.0)),
        "dtype": "float16",
    }
    config = RadFinderConfig(
        radfinder_model_config=radfinder_model_config,
        image_preprocessing=image_preprocessing,
        text_tokenizer_name=radfinder_model_config.get(
            "text_tokenizer", "Qwen/Qwen3-Embedding-0.6B"
        ),
        do_snippet_alignment=do_snippet_alignment,
        model_settings=model_settings,
    )

    log_info("Creating HF wrapper model ...")
    hf_model = RadFinderModel(config)
    hf_model.model.load_state_dict(base.state_dict())

    log_info("Saving model and config ...")
    # save_pretrained copies the modeling file into the save dir; saving
    # directly to EXPORT_DIR (where the files already live) would raise
    # SameFileError, so we save to a temp dir and promote the outputs after.
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        hf_model.save_pretrained(tmp, safe_serialization=True)
        config.save_pretrained(tmp)
        image_processor = RadFinderImageProcessor.from_radfinder_config(config)
        image_processor.save_pretrained(tmp)
        for f in tmp.iterdir():
            shutil.copy2(f, EXPORT_DIR / f.name)
    log_info(f"  → wrote {EXPORT_DIR}")
    log_info(
        f"  → use as --model_cfg via a yaml with `huggingface_path: {EXPORT_DIR.as_posix()}` "
        "(offline) or `huggingface_path: hf-hub:<repo_id>`"
    )

    if test_locally:
        log_info("Testing local model load via AutoModel ...")
        from transformers import AutoModel

        _ = AutoModel.from_pretrained(EXPORT_DIR, trust_remote_code=True)
        log_info("  ✓ AutoModel.from_pretrained succeeded")

    if args.release:
        metadata = (EXPORT_DIR / "metadata.yaml").read_text(encoding="utf-8")
        for license_file in ("LICENSE", "LICENSE_MODELS", "LICENSE_RATE"):
            src_lic = RADFINDER_REPO_DIR / license_file
            if src_lic.exists():
                shutil.copy2(src_lic, EXPORT_DIR / license_file)

        for pycache in EXPORT_DIR.rglob("__pycache__"):
            shutil.rmtree(pycache)

        log_info("Uploading to HuggingFace Hub ...")
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(EXPORT_DIR),
            path_in_repo=".",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
            delete_patterns=["*"],
        )
        log_info(f"  ✓ Uploaded to {args.repo_id}")
    else:
        log_info(
            f"Skipping upload (--release not given). To release: "
            f"--release --repo_id {args.repo_id}"
        )


def _copy_selective(src: Path, dst: Path, include: dict) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    keep_modules = {Path(name).stem for name in include}
    for name, what in include.items():
        src_path = src / name
        dst_path = dst / name
        if what is True:
            if src_path.is_dir():
                shutil.copytree(
                    src_path,
                    dst_path,
                    ignore=shutil.ignore_patterns("__pycache__", "deprecated", "devcli", "web"),
                )
            else:
                shutil.copy2(src_path, dst_path)
        else:
            _copy_selective(src_path, dst_path, {f: True for f in what})


if __name__ == "__main__":
    main()
