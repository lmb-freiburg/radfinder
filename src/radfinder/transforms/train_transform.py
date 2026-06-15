from functools import partial

from monai.transforms import Compose
from radfinder.models.load_model import FeatMode
from radfinder.models.vision_language import GlobalResC, SnippetAlignmentModeC
from radfinder.transforms.generate_report import RandomReportTransformd
from radfinder.transforms.load_features import get_feature_transforms
from radfinder.transforms.load_text_features import (
    GET_REPORT_HIDDEN_STATE,
    GET_REPORT_POOLED,
    get_text_feature_transforms,
)
from radfinder.transforms.not_load_image import NotLoadImaged
from radfinder.transforms.rand_crop_features import RandCropFeaturesd
from radfinder.transforms.select_random_scan import SelectRandomScand
from radfinder.transforms.shared_utils import (
    Language,
    TextTransformMode,
    get_image_device_dtype,
    get_image_transform_crop_and_aug,
    get_image_transform_load_and_space,
)
from radfinder.transforms.snippet_transform import RandomSnippetTransformd


def get_train_transform(
    pixdim=(0.75, 0.75, 1.5),
    img_size=(384, 384, 256),
    sliding_window_size=(128, 128, 64),
    max_shift=(64, 64, 32),
    image_feat_mode=FeatMode.FULL,
    text_feat_mode=FeatMode.FULL,
    features_dataset_name: str | None = None,
    features_model_name: str | None = None,
    do_snippet_alignment: dict | None = None,
    compose_class=Compose,
    text_transform_mode: str = TextTransformMode.DEFAULT,
    language: str = Language.EN,
    max_num_icd10: int = 20,
    keep_original_prob: float = 0.5,
    drop_prob: float = 0.3,
    dtype: str = "float16",
    use_gds: bool = False,
    random_flip: float = 0.5,
    random_crop_features: float = 0.0,
    random_flip_features: float = 0.0,
    model_settings: dict | None = None,
    organ_text_replace_prob: float = 0.0,
    no_comp_replace_prob: float = 0.0,
    image_reader=None,
):
    """
    Create training transform based on feature extraction mode.

    Args:
        pixdim: Pixel dimensions for resampling
        img_size: Image size after padding/cropping
        sliding_window_size: Size of sliding window patches
        max_shift: Maximum shift for random cropping
        image_feat_mode: FeatMode for images (FULL, FROZEN_LOCAL, FROZEN_GLOBAL, FROM_SPACED_IMAGE)
        text_feat_mode: FeatMode for text (FULL, FROZEN_LOCAL, FROZEN_GLOBAL)
        features_dataset_name: Dataset name for loading features (required for non-FULL modes)
        features_model_name: Model name for loading features (required for non-FULL modes)
        compose_class: Compose class to use (default: Compose)
        text_transform_mode: Mode for text transforms
        language: Language for text transforms
        max_num_icd10: Maximum number of ICD-10 codes for RandomReportTransformd
        keep_original_prob: Probability to keep original text in RandomReportTransformd
        drop_prob: Probability to drop elements in RandomReportTransformd
        dtype: Data type for tensors (float16 or float32)
        use_gds: Use GPUDirect Storage
        image_reader: Optional MONAI ImageReader for loading images (default: NibabelReader).
    """
    TextTransformMode.verify_value(text_transform_mode)
    if model_settings is None:
        model_settings = {}
    global_res = model_settings.get("global_res")
    device, dtype = get_image_device_dtype(dtype, image_feat_mode, use_gds)

    #################### image transforms ####################
    # if multiple scans per report, select one at random
    # this transform must always happen, otherwise collate will choke on unused all_images key.
    image_transform = [SelectRandomScand()]
    _load_and_space = partial(get_image_transform_load_and_space, pixdim, reader=image_reader)
    _crop_and_aug = partial(
        get_image_transform_crop_and_aug,
        img_size,
        sliding_window_size,
        max_shift,
        random_flip,
        dtype,
        device,
    )
    if image_feat_mode == FeatMode.FULL:
        image_transform += _load_and_space() + _crop_and_aug()
    elif image_feat_mode == FeatMode.RETURN_SPACED_IMAGE:
        image_transform += _load_and_space()
    elif image_feat_mode == FeatMode.FROM_SPACED_IMAGE:
        image_transform += _crop_and_aug()
    elif image_feat_mode in {FeatMode.NONE, FeatMode.FROZEN_LOCAL, FeatMode.FROZEN_GLOBAL}:
        # this renames "image" to "filename" which prevents monai from crashing on unloaded img
        image_transform += [NotLoadImaged(keys=("image",))]
    else:
        raise ValueError(f"Unknown {image_feat_mode=}")
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

    # Load image features if needed
    image_feature_transform = get_feature_transforms(
        feature_names=image_feature_names,
        features_dataset_name=features_dataset_name,
        features_model_name=features_model_name,
    )
    # Optionally crop features to the training grid size
    if random_crop_features > 0.0 and image_feat_mode == FeatMode.FROZEN_LOCAL:
        target_grid = tuple(img_size[i] // sliding_window_size[i] for i in range(3))
        crop_keys = ["image_backbone_cls", "image_backbone_patch_average"]
        if needs_axis2:
            crop_keys += [
                "image_backbone_patch_axis2",
            ]
        image_feature_transform.append(
            RandCropFeaturesd(
                keys=crop_keys,
                target_grid_size=target_grid,
                random_flip_prob=random_flip_features,
                random_crop_prob=random_crop_features,
            )
        )
    # load features first, in case they load the spaced_image, then run the other image transforms
    total_image_transform = image_feature_transform + image_transform

    #################### text transforms ####################
    # load the text always, then we can load features based on the text hash
    if text_transform_mode == TextTransformMode.DEFAULT:
        text_transform = [
            RandomReportTransformd(
                keys=("findings", "impressions", "icd10"),
                language=language,
                max_num_icd10=max_num_icd10,
                keep_original_prob=keep_original_prob,
                drop_prob=drop_prob,
                allow_missing_keys=False,
                organ_text_replace_prob=organ_text_replace_prob,
                no_comp_replace_prob=no_comp_replace_prob,
            ),
            RandomSnippetTransformd(
                language=language,
                is_train=True,
                allow_missing_keys=True,
            ),
        ]
    elif text_transform_mode == TextTransformMode.NONE:
        text_transform = []

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

    transform = compose_class(total_image_transform + total_text_transform)
    return transform
