"""
Transform to load saved features from disk.

image float32 (1, 768, 768, 256)  input of backbone
image_backbone_cls float32 (6, 6, 4, 1080)  output of backbone
image_backbone_patch float32 (6, 6, 4, 512, 1080)  output of backbone
image_backbone_patch_average float32 (6, 6, 4, 1080)  output of backbone

input to "feature comb" the 2nd transformer will be concat of cls and patch average.

image_feature_comb_cls float32 (1080,)  output of feature comb
image_feature_comb_patch float32 (6, 6, 4, 1080)  output of feature comb

    "Input size 768 x 768 x 256\n",
    "Split into windows of size 128 x 128 x 64 -> 6 x 6 x 4 windows\n",
    "In each window, create patches of size 16 x 16 x 8 (2048 voxels) and downscale to 1080\n",
    "So each window has 8 x 8 x 8 patches (512 patches)\n",

# Final low level feature shape is therefore
6 x  6 x  4 x    8 x  8 x  8 x    1080
W_h  W_w  W_d    P_h  P_w  P_d    dim

# Final high level feature shape
6 x  6 x  4 x    1080
W_h  W_w  W_d    dim
"""

from pathlib import Path

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from monai.transforms import MapTransform
from radfinder.data.ct_rate import CTRateDataset
from radfinder.data.inspect import InspectDataset
from radfinder.data.merlin import MerlinDataset
from radfinder.data.rad_chestct import RadChestCTDataset
from radfinder.paths import get_medv_data_dir
from radfinder.utils.logging_utils import log_debug

from packg.iotools.jsonext import load_json
from visiontext.iotools.feature_compression import load_single_safetensor_zst


class LoadFeaturesTransformd(MapTransform):
    """Load pre-computed features from disk, based on image key."""

    def __init__(
        self,
        dataset_name: str,  # feature location may depend on dataset
        images_subdir: str,
        features_subdir: str,
        feature_names: list[str],
        keys: tuple[str] = ("image",),  # image (filepath) -> extract id -> load features for id
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.features_dir = get_medv_data_dir() / "embeddings" / f"{features_subdir}"
        self.images_dir = get_medv_data_dir() / "embeddings" / f"{images_subdir}"
        self.feature_names = feature_names
        self.dataset_name = dataset_name
        self.dtype = torch.float16  # features are stored as float16 so upscaling here would be bad.
        self.images_subdir = images_subdir
        self.features_subdir = features_subdir

    def __call__(self, data):
        d = dict(data)
        assert self.keys == ("image",), f"Unsupported: {self.keys=}"
        key = self.keys[0]

        # extract image path here, because we may overwrite the "image" key below
        image_path = d["image"]
        assert isinstance(image_path, str), f"{type(image_path)=}"

        # Extract scan_id from the data
        # Assuming the key points to a path or identifier
        if key not in d:
            if not self.allow_missing_keys:
                raise KeyError(f"Key '{key}' not found in data.")
            else:
                return d
        metadata_loaded = False
        for feat_name in self.feature_names:
            if feat_name == "image_spaced":
                feat_dir = self.images_dir
            else:
                feat_dir = self.features_dir
            feat_file = get_feature_file(feat_dir, image_path, feat_name, self.dataset_name)
            if not feat_file.is_file():
                raise FileNotFoundError(f"Feature file not found: {feat_file}")
            features = load_single_safetensor_zst(feat_file)

            assert features.dtype == self.dtype, f"{features.dtype=} != {self.dtype=}"

            if feat_name == "image_spaced":
                # special case: if loading image_spaced, put it in the "image" key
                # so that subsequent transforms can process it. and add metadata
                meta_dict = self._load_metadata_for_feature(feat_file)
                d["image"] = MetaTensor(features, meta=meta_dict)
            elif feat_name == "image_backbone_cls":
                # special case: image is not loaded but we still need the metadata, so load it from
                # the image_spaced file. the function will search it both in the "images_" storage
                # and the "{model_name}_" storage.
                d[f"{feat_name}"] = features
            else:
                d[f"{feat_name}"] = features
            d["features_dir"] = self.features_dir.as_posix()

            if not metadata_loaded:
                # in some situations we may not need the metadata however it's easiest to just
                # load it always. note it's not cropped so must be fixed in the collate function.
                meta_dict = self._load_metadata_for_feature(feat_file)
                metadata_loaded = True
                d["image_metadata_uncropped"] = meta_dict
        return d

    def _load_metadata_for_feature(self, feat_file: Path) -> dict:
        metadata_file = feat_file.parent / "image_spaced.json"
        if not metadata_file.is_file():
            # hacky fix the path to go from model feature storage to image storage
            repl_source, repl_target = f"/{self.features_subdir}/", f"/{self.images_subdir}/"
            mnew = Path(metadata_file.as_posix().replace(repl_source, repl_target))
            if mnew == metadata_file:
                raise FileNotFoundError(
                    f"Not found: {metadata_file} failed replacing: {repl_source=} -> {repl_target=}"
                )
            if not mnew.is_file():
                raise FileNotFoundError(f"Not found both {metadata_file} and {mnew}")
            metadata_file = mnew
        metadata = load_json(metadata_file)
        meta_dict = convert_metadata_from_json(metadata)  # convert metadata back to proper types
        return meta_dict


def get_feature_transforms(
    feature_names: list[str],
    features_dataset_name: str | None,
    features_model_name: str | None,
    model_name: str = "spectre",
):
    if len(feature_names) == 0:
        return []
    assert features_dataset_name is not None, f"{features_dataset_name=} required"
    assert features_model_name is not None, f"{features_model_name=} required"
    images_subdir, features_subdir = get_features_subdir(
        features_dataset_name, features_model_name, model_name=model_name
    )
    log_debug(f"[dataset={features_dataset_name}] Loading features from subdir: {features_subdir}")
    return [
        LoadFeaturesTransformd(features_dataset_name, images_subdir, features_subdir, feature_names)
    ]


def get_features_subdir(
    features_dataset_name: str, features_model_name: str, model_name: str = "spectre"
):
    """Find the folder to save the features for this dataset and model / dataloading combination."""
    features_subdir_dataset = features_dataset_name.lower()

    if (
        features_model_name.startswith("radfinder")
        or features_model_name == "spectre_pretrained_half_patch_embed"
    ):
        features_subdir_res = "3mm"
        images_subdir = "3mm"
    elif features_model_name == "spectre_pretrained_siglip_res":
        features_subdir_res = "1.5mm"
        images_subdir = "1.5mm"
    elif features_model_name == "spectre_pretrained_dino_res":
        features_subdir_res = "1mm"
        images_subdir = "1mm"
    else:
        raise NotImplementedError(f"Model {features_model_name} no feature dir implemented")
    return (
        f"images_{images_subdir}_{features_subdir_dataset}",
        f"{model_name}_{features_subdir_res}_{features_subdir_dataset}",
    )


def get_feature_file(features_dir: Path, image_path: str, feature_name: str, dataset_name: str):
    if dataset_name == "ctrate":
        dcls = CTRateDataset
    elif dataset_name == "merlin":
        dcls = MerlinDataset
    elif dataset_name == "inspect":
        dcls = InspectDataset
    elif dataset_name == "radchestct":
        dcls = RadChestCTDataset
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported for feature loading.")
    scanrserid = dcls.get_datapoint_key_from_scan_path(image_path)
    feature_dir = features_dir / dcls.get_feature_subdir_from_datapoint_key(scanrserid)
    feature_file = feature_dir / f"{feature_name}.safetensors.zst"
    return feature_file


def convert_metadata_from_json(metadata: dict) -> dict:
    """Convert metadata from JSON format back to monai MetaTensor compatible format."""
    metadata_out = {}
    for key, value in metadata.items():
        if key in NUMPY_ARRAY_FIELDS:
            dtype = NUMPY_ARRAY_FIELDS[key]
            metadata_out[key] = np.array(value, dtype=dtype)
        elif key in OTHER_FIELDS:
            cast_type = OTHER_FIELDS[key]
            if cast_type == float and value is None:
                metadata_out[key] = float("nan")
            else:
                metadata_out[key] = cast_type(value)
        else:
            raise ValueError(f"Unknown metadata field {key=} {value=}")
    return metadata_out


# Metadata fields that should be converted to numpy arrays
# Key: field name, Value: numpy dtype (or None to infer from data)
NUMPY_ARRAY_FIELDS = {
    "affine": np.float64,  # actually this comes out as tensor because it's created by transforms
    "bitpix": np.int16,
    "cal_max": np.float32,
    "cal_min": np.float32,
    "datatype": np.int16,
    "dim": np.int16,
    "dim_info": np.uint8,
    "extents": np.int32,
    "glmax": np.int32,
    "glmin": np.int32,
    "intent_code": np.int16,
    "intent_p1": np.float32,
    "intent_p2": np.float32,
    "intent_p3": np.float32,
    "location": np.int64,
    "original_affine": np.float64,
    "pixdim": np.float32,
    "qform_code": np.int16,
    "qoffset_x": np.float32,
    "qoffset_y": np.float32,
    "qoffset_z": np.float32,
    "quatern_b": np.float32,
    "quatern_c": np.float32,
    "quatern_d": np.float32,
    "scl_inter": np.float32,
    "scl_slope": np.float32,
    "session_error": np.int16,
    "sform_code": np.int16,
    "sizeof_hdr": np.int32,
    "slice_code": np.uint8,
    "slice_duration": np.float32,
    "slice_end": np.int16,
    "slice_start": np.int16,
    "spatial_shape": np.int64,
    "spatial_shape_final": np.int64,
    "srow_x": np.float32,
    "srow_y": np.float32,
    "srow_z": np.float32,
    "toffset": np.float32,
    "vox_offset": np.float32,
    "xyzt_units": np.uint8,
}

OTHER_FIELDS = {
    "as_closest_canonical": bool,
    "count": int,
    "filename_or_obj": str,
    "offset": tuple,
    "original_channel_dim": float,
    "space": str,  # from monai.utils.enums import SpaceKeys  # is a StrEnum, but Enums suck
}
