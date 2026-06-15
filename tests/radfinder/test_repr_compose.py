import re

from monai.data.image_reader import NibabelReader
from monai.transforms import (
    EnsureChannelFirstd,
    GridPatchd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)
from radfinder.transforms.generate_report import RandomReportTransformd
from radfinder.transforms.largest_multiple_crop import LargestMultipleCenterCropd
from radfinder.transforms.new_compose import TimedCompose

OUT = """
TimedCompose(lazy=False, transforms=(
  TimedCompose(lazy=False, transforms=(
    LoadImaged(
      allow_missing_keys=False,
      backend=[],
      keys=('image',),
      meta_key_postfix=('meta_dict',),
      meta_keys=(None,),
      overwriting=False
    ),
    EnsureChannelFirstd(
      allow_missing_keys=False,
      backend=[torch, numpy],
      keys=('image',)
    ),
    ScaleIntensityRanged(
      allow_missing_keys=False,
      backend=[torch, numpy],
      keys=('image',)
    ),
    Orientationd(
      allow_missing_keys=False,
      backend=[numpy, torch],
      keys=('image',),
      lazy=False,
      Orientation.as_closest_canonical=False,
      Orientation.axcodes='RAS',
      Orientation.backend=[numpy, torch],
      Orientation.labels=None,
      Orientation.lazy=False,
      Orientation.requires_current_data=False,
      Orientation.tracing=True,
      requires_current_data=False,
      tracing=True
    ),
    Spacingd(
      align_corners=(False,),
      allow_missing_keys=False,
      backend=[torch, numpy, cupy],
      dtype=(<class 'numpy.float64'>,),
      ensure_same_shape=True,
      keys=('image',),
      lazy=False,
      mode=('bilinear',),
      padding_mode=(border,),
      requires_current_data=False,
      scale_extent=(False,),
      Spacing.backend=[torch, numpy, cupy],
      Spacing.diagonal=False,
      Spacing.lazy=False,
      Spacing.max_pixdim=array([nan]),
      Spacing.min_pixdim=array([nan]),
      Spacing.pixdim=array([0.5, 0.5, 1. ]),
      Spacing.recompute_affine=False,
      Spacing.requires_current_data=False,
      Spacing.scale_extent=False,
      Spacing.SpatialResample.align_corners=False,
      Spacing.SpatialResample.backend=[torch, numpy, cupy],
      Spacing.SpatialResample.lazy=False,
      Spacing.SpatialResample.mode=bilinear,
      Spacing.SpatialResample.padding_mode=border,
      Spacing.SpatialResample.requires_current_data=False,
      Spacing.SpatialResample.tracing=True,
      Spacing.tracing=True,
      tracing=True
    ),
    LargestMultipleCenterCropd(
      allow_missing_keys=False,
      backend=[torch],
      CenterSpatialCrop.backend=[torch],
      CenterSpatialCrop.lazy=False,
      CenterSpatialCrop.requires_current_data=False,
      CenterSpatialCrop.roi_size=(128, 128, 64),
      CenterSpatialCrop.tracing=True,
      keys=('image',),
      lazy=False,
      min_area_for_padding=0.0,
      SpatialPad.backend=[torch, numpy],
      SpatialPad.kwargs={'value': 0.0},
      SpatialPad.lazy=False,
      SpatialPad.method=symmetric,
      SpatialPad.mode='constant',
      SpatialPad.requires_current_data=False,
      SpatialPad.spatial_size=(128, 128, 64),
      SpatialPad.to_pad=None,
      SpatialPad.tracing=True,
      patch_size=(128, 128, 64),
      requires_current_data=False,
      tracing=True
    ),
    SpatialPadd(
      allow_missing_keys=False,
      backend=[torch, numpy],
      keys=('image',),
      lazy=False,
      mode=('constant',),
      SpatialPad.backend=[torch, numpy],
      SpatialPad.kwargs={'value': 0.0},
      SpatialPad.lazy=False,
      SpatialPad.method=symmetric,
      SpatialPad.mode=constant,
      SpatialPad.requires_current_data=False,
      SpatialPad.spatial_size=(128, 128, 64),
      SpatialPad.to_pad=None,
      SpatialPad.tracing=True,
      requires_current_data=False,
      tracing=True
    ),
    GridPatchd(
      allow_missing_keys=False,
      backend=[torch, numpy],
      keys=('image',)
    ),
    RandomReportTransformd(
      R=RandomState(MT19937) at 0x7F706A9BA340,
      allow_missing_icd10=True,
      allow_missing_keys=False,
      backend=[],
      drop_impressions_prob=0.0,
      drop_prefix=False,
      drop_prob=0.0,
      keep_original_prob=1.0,
      keys=('findings', 'impressions'),
      language='en',
      max_num_icd10=20,
      no_comp_replace_prob=0.0,
      organ_text_replace_prob=0.0
    )
  ))
))
"""


def test_compose_repr():
    """Test that TimedCompose repr matches expected output.

    Hardcoded transform to avoid depending on get_eval_transform() changes.
    """
    # Recreate the exact transform that produced the OUT string
    inner_transforms = [
        LoadImaged(keys=("image",), reader=NibabelReader()),
        EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
        ScaleIntensityRanged(
            keys=("image",), a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        Orientationd(keys=("image",), axcodes="RAS", labels=None),
        Spacingd(keys=("image",), pixdim=(0.5, 0.5, 1.0), mode=("bilinear",)),
        LargestMultipleCenterCropd(keys=("image",), patch_size=(128, 128, 64)),
        SpatialPadd(keys=("image",), spatial_size=(128, 128, 64), mode="constant", value=0.0),
        GridPatchd(keys=("image",), patch_size=(128, 128, 64), overlap=0.0),
        RandomReportTransformd(
            keys=("findings", "impressions"),
            keep_original_prob=1.0,
            drop_prob=0.0,
            allow_missing_keys=False,
        ),
    ]
    inner_compose = TimedCompose(inner_transforms)
    transform = TimedCompose([inner_compose])

    repr_str = repr(transform)

    # Normalize the memory address in RandomState repr
    # Replace any address like "0x7F9CB98B5140" with the expected one "0x7F706A9BA340"
    normalized_repr = re.sub(
        r"R=RandomState\(MT19937\) at (0x[0-9A-Fa-f]+)",
        r"R=RandomState(MT19937) at 0x7F706A9BA340",
        repr_str,
    )
    normalized_out = OUT.strip()
    assert normalized_repr == normalized_out, f"TimedCompose repr does not match expected output"


def main():
    test_compose_repr()


if __name__ == "__main__":
    main()
