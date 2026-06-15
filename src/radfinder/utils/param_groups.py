from collections import defaultdict
from operator import itemgetter

import torch.nn as nn
from natsort import natsorted
from radfinder.utils.logging_utils import log_info

from visiontext.torchutils import group_params_and_data_for_display


def get_vit_lr_decay_rate(
    name: str,
    llrd_factor: float = 1.0,
    num_layers: int = 12,
    force_is_backbone: bool = False,
    shift: int = 0,
) -> float:
    """
    Get the layer-wise learning rate decay (LLRD) rate for a given parameter name.

    Args:
        name:
            The name of the parameter.
        llrd_factor:
            The decay factor for each layer.
        num_layers:
            The total number of layers in the model.
        force_is_backbone:
            If True, forces the function to treat the parameter as part of the backbone.
        shift:
            An integer to shift the layer ids, useful when combining multiple modules.

    Returns:
        The learning rate multiplier for the parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".patch_proj" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".reg_token" in name
        ):
            layer_id = 0
        elif ".blocks." in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1 + shift

    return llrd_factor ** (num_layers + 1 - layer_id)


def get_param_groups_with_decay(
    model: nn.Module,
    llrd_factor: float = 1.0,
    patch_embed_lr_mult: float = 1.0,
    projection_head_wd_mult: float = 1.0,
    lora_lr_factor: float = 1.0,
    num_layers: int | None = None,
):

    force_is_backbone = False
    shift = 0
    if num_layers is not None:
        num_layers = num_layers
    elif hasattr(model, "n_blocks"):
        num_layers = model.n_blocks
        force_is_backbone = True
    elif hasattr(model, "blocks"):
        num_layers = len(model.blocks)
        force_is_backbone = True
    elif hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        num_layers = len(model.backbone.blocks)
    elif hasattr(model, "backbone_student") and hasattr(
        model.backbone_student, "blocks"
    ):  # DINO specific
        num_layers = len(model.backbone_student.blocks)
    elif (
        hasattr(model, "backbone_student")
        and hasattr(model.backbone_student, "vit")
        and hasattr(model.backbone_student.vit, "blocks")
    ):  # DINOv2 specific
        num_layers = len(model.backbone_student.vit.blocks)
    elif hasattr(model, "backbone_image") and hasattr(
        model.backbone_image, "blocks"
    ):  # SigLIP specific
        if not hasattr(model, "feature_comb_image") or model.feature_comb_image is None:
            num_layers = len(model.backbone_image.blocks)
        else:
            num_layers = len(model.backbone_image.blocks) + len(model.feature_comb_image.blocks)
            shift = len(model.backbone_image.blocks)
            force_is_backbone = True
    else:
        num_layers = 0

    all_param_groups = []
    for n, p in model.named_parameters():
        if not "lora_" in n:
            s = shift if "feature_comb" in n else 0
            llrd_rate = get_vit_lr_decay_rate(
                n,
                llrd_factor,
                num_layers,
                force_is_backbone,
                s,
            )

            d = {
                "name": n,
                "params": p,
                "lr_mult": llrd_rate,
                "wd_mult": 1.0,
                "requires_grad": p.requires_grad,
            }

            if "head" in n or "projection" in n:
                d["wd_mult"] = projection_head_wd_mult

            # No weight-decay on biases, norm parameters, layer scale gamma, learned tokens and embeddings
            if n.endswith("bias") or "norm" in n or "gamma" in n or "fourrier_w" in n:
                d["wd_mult"] = 0.0

            if "patch_embed" in n:
                d["lr_mult"] *= patch_embed_lr_mult

        else:
            # LoRA parameters
            d = {
                "name": n,
                "params": p,
                "lr_mult": lora_lr_factor,
                "wd_mult": 1.0,
                "requires_grad": p.requires_grad,
            }
        all_param_groups.append(d)
    all_param_groups_grouped = {}
    for pg in all_param_groups:
        param_name = pg["name"]
        param = pg["params"]
        lr_mult = pg["lr_mult"]
        wd_mult = pg["wd_mult"]
        group_name = f"lr_{lr_mult:.2e}_wd_{wd_mult:.2e}_grad_{str(pg['requires_grad']).lower()}"
        if group_name not in all_param_groups_grouped:
            all_param_groups_grouped[group_name] = {
                "params": [],
                "param_names": [],
                "wd_mult": wd_mult,
                "lr_mult": lr_mult,
                "requires_grad": pg["requires_grad"],
            }
        all_param_groups_grouped[group_name]["params"].append(param)
        all_param_groups_grouped[group_name]["param_names"].append(param_name)
    return list(all_param_groups_grouped.values())


def print_optimizer_parameters(optimizer, log_fn=log_info):
    """Print all parameters in optimizer with their gradient requirements and shapes.

    Args:
        optimizer: The optimizer instance
        accelerator: Accelerator for printing
        model: Optional model to get parameter names from
        criterion: Optional criterion to get parameter names from
    """
    # Collect all parameters in optimizer
    optimizer_param_ids = set()
    all_params = []
    total_params = 0
    requires_grad_params = 0

    for group_idx, param_group in enumerate(optimizer.param_groups):
        lr_mult = param_group["lr_mult"]
        wd_mult = param_group["wd_mult"]
        requires_grad = param_group["requires_grad"]
        params_list = param_group["params"]
        param_names = param_group["param_names"]
        for param, param_name in zip(params_list, param_names):
            optimizer_param_ids.add(id(param))
            num_params = param.numel()
            total_params += num_params
            if requires_grad:
                requires_grad_params += num_params
            all_params.append(
                {
                    "name": param_name,
                    "shape": tuple(param.shape),
                    "lr_mult": lr_mult,
                    "wd_mult": wd_mult,
                    "requires_grad": requires_grad,
                    "group": group_idx,
                }
            )

    # Use torchutils to compress parameter names
    param_names = [p["name"] for p in all_params]
    compressed_names, compressed_data = group_params_and_data_for_display(param_names, all_params)

    # Print header
    log_fn("=" * 140)
    log_fn("OPTIMIZER PARAMETERS")
    log_fn("=" * 140)
    log_fn(
        f"{'Parameter Name':<70} {'Shape':<25} {'LR Mult':<12} {'WD Mult':<12} {'Grad':<6} "
        f"{'Group':<6}"
    )
    log_fn("-" * 140)

    # Print each parameter
    for i, (name, data) in enumerate(
        natsorted(zip(compressed_names, compressed_data), key=itemgetter(0))
    ):
        shape_str = str(data["shape"])
        lr_str = f"{data['lr_mult']:.2e}"
        wd_str = f"{data['wd_mult']:.2e}"
        grad_str = "X" if data["requires_grad"] else "-"
        log_fn(
            f"{name:<70} {shape_str:<25} {lr_str:<12} {wd_str:<12} "
            f"{grad_str:<6}  {data['group']}"
        )

    log_fn("=" * 140)
    log_fn(
        f"TOTAL: {total_params:,} params | Requires grad: {requires_grad_params:,} "
        f"({requires_grad_params/total_params:.2%}) | Groups: {len(optimizer.param_groups):_d}"
    )
    log_fn("=" * 140)


def print_optimizer_parameters_all(param_groups):
    for pg in param_groups:
        lr_mult = pg["lr_mult"]
        wd_mult = pg["wd_mult"]
        for param, param_name in natsorted(
            zip(pg["params"], pg["param_names"]), key=lambda x: x[1]
        ):
            param_shape_str = ",".join(str(ps) for ps in param.shape)
            log_info(
                f"{param_name:60s} {param_shape_str:30s} "
                f"{'X' if param.requires_grad else ' '}   {lr_mult:.2e} {wd_mult:.2e}"
            )
