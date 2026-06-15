"""
Random center crop for precomputed backbone features.

Features have shape (Hp, Wp, Dp, ..., E) where the first 3 dims are the spatial
window grid. This transform crops to a target grid size using a center crop with
a small random jitter of {-1, 0, 1} per axis. If a scan is smaller than the
target in any dimension, that dimension is left as-is (pad_and_stack in collate
handles the padding).
"""

import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform


class RandCropFeaturesd(MapTransform):
    """Randomly crop precomputed backbone features to a target grid size.

    Args:
        keys: Feature keys to crop, e.g. ["image_backbone_cls", "image_backbone_patch_average"].
        target_grid_size: Target (H, W, D) number of windows after cropping.
        max_jitter: Maximum random shift per axis (default 1, giving {-1, 0, 1}).
        allow_missing_keys: If True, skip missing keys instead of raising.
    """

    def __init__(
        self,
        keys: KeysCollection,
        target_grid_size: tuple[int, int, int],
        max_jitter: int = 1,
        allow_missing_keys: bool = False,
        random_flip_prob: float = 0.0,
        random_crop_prob: float = 1.0,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.target_grid_size = target_grid_size
        self.max_jitter = max_jitter
        self.random_flip = random_flip_prob
        self.random_crop_prob = random_crop_prob

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        # Compute random jitter once so all keys get the same crop
        jitter = tuple(
            torch.randint(-self.max_jitter, self.max_jitter + 1, (1,)).item() for _ in range(3)
        )
        # Decide which axes to flip (independent 50% chance each)
        flip_dims = [i for i in range(3) if self.random_flip and torch.rand(1).item() < 0.5]
        slices = None
        for key in self.key_iterator(d):
            feat = d[key]  # (Hp, Wp, Dp, ..., E)
            if slices is None:
                slices = self._compute_slices(feat.shape[:3], jitter)
            feat = feat[slices]
            if flip_dims:
                feat = torch.flip(feat, dims=flip_dims)
            d[key] = feat
        crop_start = []
        crop_width = []
        for i, s in enumerate(slices):
            start = s.start if s.start is not None else 0
            stop = s.stop if s.stop is not None else feat.shape[i]
            crop_start.append(start)
            crop_width.append(stop - start)
        d["feature_crop_box"] = crop_start + crop_width
        return d

    def _compute_slices(
        self, spatial_shape: tuple[int, int, int], jitter: tuple[int, int, int]
    ) -> tuple[slice, slice, slice]:
        slices = []
        for s, t, j in zip(spatial_shape, self.target_grid_size, jitter):
            if s <= t:
                # Scan smaller than target: keep everything
                slices.append(slice(None))
                continue
            if self.random_crop_prob < 1.0 and torch.rand(1).item() >= self.random_crop_prob:
                # With probability 1 - crop_prob, skip cropping and keep everything
                slices.append(slice(None))
                continue
            # Center crop with jitter, clamped to valid range
            start = (s - t) // 2 + j
            start = max(0, min(s - t, start))
            slices.append(slice(start, start + t))
        return tuple(slices)
