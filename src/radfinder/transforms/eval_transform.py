from functools import partial

from radfinder.loader_utils import NibabelReader  # type: ignore
from radfinder.models.load_model import FeatMode
from radfinder.models.vision_language import GlobalResC, SnippetAlignmentModeC
from radfinder.transforms.generate_report import RandomReportTransformd
from radfinder.transforms.load_features import get_feature_transforms
from radfinder.transforms.load_text_features import (
    GET_REPORT_HIDDEN_STATE,
    GET_REPORT_POOLED,
    get_text_feature_transforms,
)
from radfinder.transforms.new_compose import ReprCompose
from radfinder.transforms.not_load_image import NotLoadImaged
from radfinder.transforms.shared_utils import (
    Language,
    TransformKindC,
    get_image_device_dtype,
    get_image_transform_eval_crop,
    get_image_transform_load_and_space,
)
from radfinder.transforms.snippet_transform import RandomSnippetTransformd

from packg.constclass import Const


class TextTransformMode(Const):
    DEFAULT = "default"  # spectre authors: english findings + impression
    NONE = "none"  # keep text as is


def get_eval_transform(
    pixdim=(0.5, 0.5, 1.0),
    sliding_window_size=(128, 128, 64),
    image_feat_mode=FeatMode.FULL,
    text_feat_mode=FeatMode.FULL,
    features_dataset_name: str | None = None,
    features_model_name: str | None = None,
    do_snippet_alignment: dict | None = None,
    compose_class=ReprCompose,
    text_transform_mode: str = TextTransformMode.DEFAULT,
    language: str = Language.EN,
    lazy: bool = False,
    dtype: str = "float16",
    min_area_for_padding: float = 0.0,
    model_settings: dict | None = None,
    image_reader=None,
    drop_findings: bool = False,
    drop_impressions: bool = False,
    drop_prefix: bool = False,
    transform_kind: str = TransformKindC.SPECTRE,
    target_shape: tuple[int, int, int] | None = None,
):
    """
    Create evaluation transform based on feature extraction mode.

    Args:
        pixdim: Pixel dimensions for resampling
        sliding_window_size: Patch/window size for padding/cropping
        image_feat_mode: FeatMode for images (FULL, FROZEN_LOCAL, FROZEN_GLOBAL, FROM_SPACED_IMAGE)
        text_feat_mode: FeatMode for text (FULL, FROZEN_LOCAL, FROZEN_GLOBAL)
        features_dataset_name: Dataset name for loading features (required for non-FULL modes)
        compose_class: Compose class to use (default: Compose)
        features_dir_overwrite: Optional overwrite for features directory, instead of automatic
            based on dataset name and model.
    """
    if model_settings is None:
        model_settings = {}
    global_res = model_settings.get("global_res")
    device, dtype = get_image_device_dtype(dtype, image_feat_mode, use_gds=False)

    #################### image transforms ####################
    image_transform = None
    if transform_kind == TransformKindC.SPECTRE:
        _load_and_space = partial(get_image_transform_load_and_space, pixdim, reader=image_reader)
        _crop = partial(
            get_image_transform_eval_crop,
            sliding_window_size,
            dtype,
            device,
            min_area_for_padding=min_area_for_padding,
        )

        if image_feat_mode == FeatMode.FULL:
            image_transform = _load_and_space() + _crop()
        elif image_feat_mode == FeatMode.RETURN_SPACED_IMAGE:
            image_transform = _load_and_space()
        elif image_feat_mode == FeatMode.FROM_SPACED_IMAGE:
            image_transform = _crop()
        elif image_feat_mode in {FeatMode.NONE, FeatMode.FROZEN_LOCAL, FeatMode.FROZEN_GLOBAL}:
            # this renames "image" to "filename" which prevents monai from crashing on unloaded img
            image_transform = [NotLoadImaged(keys=("image",))]
        else:
            raise ValueError(f"Unknown {image_feat_mode=}")
    else:
        TransformKindC.verify_value(transform_kind)

    image_feature_names = []
    if image_feat_mode == FeatMode.FROM_SPACED_IMAGE:
        image_feature_names += ["image_spaced"]
    if image_feat_mode == FeatMode.FROZEN_LOCAL:
        image_feature_names += ["image_backbone_cls", "image_backbone_patch_average"]
        snippet_mode = None
        if do_snippet_alignment is not None:
            snippet_mode = do_snippet_alignment.get("snippet_mode")
        needs_axis2 = (
            snippet_mode == SnippetAlignmentModeC.AXIS_LOCALIZATION
            or global_res == GlobalResC.AXIS2
        )
        if needs_axis2:
            image_feature_names += [
                "image_backbone_patch_axis2",
            ]
    if image_feat_mode == FeatMode.FROZEN_GLOBAL:
        image_feature_names += ["image_feature_comb_cls", "image_feature_comb_patch"]
    image_feature_transform = get_feature_transforms(
        feature_names=image_feature_names,
        features_dataset_name=features_dataset_name,
        features_model_name=features_model_name,
    )
    # load features first, in case they load the spaced_image, then run the other image transforms
    total_image_transform = image_feature_transform + image_transform

    #################### text transforms ####################
    # load the text always, then we can load features based on the text hash
    if text_feat_mode == FeatMode.NONE:
        text_transform_mode = TextTransformMode.NONE
    if text_transform_mode == TextTransformMode.DEFAULT:
        text_transform = [
            RandomReportTransformd(
                keys=("findings", "impressions"),
                language=language,
                keep_original_prob=1.0,
                drop_prob=float(drop_findings),
                drop_impressions_prob=float(drop_impressions),
                allow_missing_keys=False,
                drop_prefix=drop_prefix,
            ),
            RandomSnippetTransformd(
                language=language,
                is_train=True,
                allow_missing_keys=True,
            ),
        ]
    elif text_transform_mode == TextTransformMode.NONE:
        text_transform = []
    else:
        TextTransformMode.verify_value(text_transform_mode)

    text_keys_and_features = {}
    if text_feat_mode == FeatMode.FROZEN_LOCAL:
        text_keys_and_features.update(GET_REPORT_HIDDEN_STATE)
    if text_feat_mode == FeatMode.FROZEN_GLOBAL:
        text_keys_and_features.update(GET_REPORT_POOLED)

    text_feature_transform = get_text_feature_transforms(
        keys_and_features=text_keys_and_features,
        features_dataset_name=features_dataset_name,
        features_model_name=features_model_name,
    )
    # first load texts as strings, then load features based on the text hash
    total_text_transform = text_transform + text_feature_transform

    all_transforms = total_image_transform + total_text_transform
    transform = compose_class(all_transforms, lazy=lazy)
    return transform
