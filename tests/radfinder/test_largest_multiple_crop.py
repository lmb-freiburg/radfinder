import torch
from radfinder.transforms.largest_multiple_crop import LargestMultipleCenterCropd


def test_largest_multiple_crop_min_area_for_padding():
    patch_size = (128, 128, 64)
    input_shape = (1, 511, 300, 70)
    cases = [
        (0.0, (1, 384, 256, 64), 384 * 256 * 64),
        (0.5, (1, 512, 256, 64), 511 * 256 * 64),
        (1.0, (1, 384, 256, 64), 384 * 256 * 64),
    ]

    for min_area_for_padding, expected_shape, expected_sum in cases:
        transform = LargestMultipleCenterCropd(
            keys=("image",),
            patch_size=patch_size,
            min_area_for_padding=min_area_for_padding,
        )
        data = {"image": torch.ones(input_shape, dtype=torch.float32)}
        out = transform(data)["image"]

        assert tuple(out.shape) == expected_shape
        assert int(out.sum().item()) == expected_sum
