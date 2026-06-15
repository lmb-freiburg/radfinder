from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from radfinder.models.layers.weight_norm_copy import weight_norm as weight_norm_old
from torch.nn.utils.parametrizations import weight_norm

WEIGHT_NORM = "new"  # old, new, manual


class SigLIPProjectionHead(nn.Module):
    """
    Projection head for SigLIP.

    Whereas SigLIP originally used a single linear layer for the projection
    head, we use a 3-layer MLP to deal with the partially frozen image and
    text backbones. This is similar to the DINO projection head without l2
    normalization (l2 normalization is performed in loss) and with LayerNorm.

    Attributes:
        input_dim:
            The input dimension of the head.
        hidden_dim:
            The hidden dimension.
        bottleneck_dim:
            Dimension of the bottleneck in the last layer of the head.
        output_dim:
            The output dimension of the head.
        layer_norm:
            Whether to use layer norm or not. Should be set to False when using
            a vision transformer backbone.
        freeze_last_layer:
            Number of epochs during which we keep the output layer fixed.
            Typically doing so during the first epoch helps training. Try
            increasing this value if the loss does not decrease.
        norm_last_layer:
            Whether or not to weight normalize the last layer of the DINO head.
            Not normalizing leads to better performance but can make the
            training unstable.

    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 512,
        layer_norm: bool = False,
        # old implementation of freeze_last_layer: default -1, always unfrozen
        # new: default 0, always unfrozen, but -1 means always frozen, same as backbone freezing
        freeze_last_layer: int = 0,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        blocks = [
            (
                input_dim,
                hidden_dim,
                nn.LayerNorm(hidden_dim) if layer_norm else None,
                nn.GELU(),
            ),
            (
                hidden_dim,
                hidden_dim,
                nn.LayerNorm(hidden_dim) if layer_norm else None,
                nn.GELU(),
            ),
            (hidden_dim, bottleneck_dim, None, None),
        ]

        layers: List[nn.Module] = []
        for block in blocks:
            in_dim, out_dim, ln, non_linearity, *bias = block
            use_bias = bias[0] if bias else not bool(ln)
            layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
            if ln:
                layers.append(ln)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.freeze_last_layer = freeze_last_layer

        if WEIGHT_NORM == "old":
            self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
            self.last_layer = weight_norm_old(self.last_layer)
            self.last_layer.weight_g.data.fill_(1)  # type: ignore  # old version for weight_norm_copy
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        elif WEIGHT_NORM == "new":
            self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
            self.last_layer = weight_norm(self.last_layer)
            with torch.no_grad():
                self.last_layer.parametrizations.weight.original0.fill_(1.0)  # type: ignore
            if norm_last_layer:
                self.last_layer.parametrizations.weight.original0.requires_grad = False  # type: ignore
        elif WEIGHT_NORM == "manual":
            self.last_layer = WeightNormLinear(bottleneck_dim, output_dim, bias=False, dim=0)
            with torch.no_grad():
                self.last_layer.weight_g.fill_(1.0)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad_(False)
        else:
            raise ValueError(f"Invalid weight norm: {WEIGHT_NORM}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.layer_norm = layer_norm
        self.freeze_last_layer = freeze_last_layer
        self.norm_last_layer = norm_last_layer

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """Cancel last layer gradients to stabilize the training."""
        if current_epoch >= self.freeze_last_layer:
            return
        for param in self.last_layer.parameters():
            param.grad = None

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes layers with a truncated normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init._no_grad_trunc_normal_(
                module.weight,
                mean=0,
                std=0.02,
                a=-2,
                b=2,
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes one forward pass through the head."""
        x = self.layers(x)
        x = self.last_layer(x)
        return x


class WeightNormLinear(nn.Module):
    """
    Explicit weight-norm Linear that is compatible with legacy torch.nn.utils.weight_norm
    checkpoints that store:
      - <name>.weight_g : [out_features, 1]  (when dim=0)
      - <name>.weight_v : [out_features, in_features]
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dim: int = 0):
        super().__init__()
        if dim != 0:
            raise NotImplementedError("This implementation matches dim=0 (per-output-channel g).")

        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim

        # Matches legacy weight_norm naming
        self.weight_v = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_g = nn.Parameter(torch.empty(out_features, 1))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Same-ish defaults as nn.Linear
        nn.init.kaiming_uniform_(self.weight_v, a=5**0.5)
        with torch.no_grad():
            # Common weight-norm init: g = ||v|| along dim=1 (per row), keep [out,1]
            g = torch.norm(self.weight_v, dim=1, keepdim=True).clamp_min(1e-12)
            self.weight_g.copy_(g)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def _weight(self) -> torch.Tensor:
        # w = g * v / ||v||, with g shape [out,1]
        v_norm = torch.norm(self.weight_v, dim=1, keepdim=True).clamp_min(1e-12)
        return self.weight_v * (self.weight_g / v_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._weight(), self.bias)
