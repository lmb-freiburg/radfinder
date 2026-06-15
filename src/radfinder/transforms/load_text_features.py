"""
Transform to load saved text features from disk
"""

import torch
from monai.transforms import MapTransform
from radfinder.data.text_feature_store import TextFeatureStore
from radfinder.paths import get_medv_data_dir
from radfinder.utils.logging_utils import log_info
from torch.utils.data import get_worker_info


class LoadTextFeaturesTransformd(MapTransform):
    """Load pre-computed text features from disk, based on text hash."""

    def __init__(
        self,
        keys_and_features: dict[str, str],  # input name -> output name
        allow_missing_keys: bool = False,
        text_features_subdir: str = "spectre",
    ):
        keys = list(keys_and_features.keys())
        super().__init__(keys, allow_missing_keys)
        self.text_features_dir = get_medv_data_dir() / "embeddings" / f"text_{text_features_subdir}"
        self.keys_and_features = keys_and_features
        self._store: TextFeatureStore | None = None
        self._store_worker_id: int | None = None
        required_dbs = set()
        for feature_name in self.keys_and_features.values():
            required_dbs.add(KEY_TO_DB[feature_name])
        self.required_dbs = required_dbs

    def _ensure_store_open(self) -> None:
        """Reopen store if the worker id changes or if it is not open yet."""
        worker_info = get_worker_info()
        worker_id = None if worker_info is None else worker_info.id
        if self._store is None or self._store_worker_id != worker_id:
            log_info(f"(Re-)opening text feature store for worker {worker_id}")
            if self._store is not None:
                self._store.close()
            self._store = TextFeatureStore(self.text_features_dir)
            if "embeddings" in self.required_dbs:
                self._store._read_emb_txn = self._store.embeddings_db.begin(write=False)
            if "pool" in self.required_dbs:
                self._store._read_pool_txn = self._store.pool_db.begin(write=False)
            if "texts" in self.required_dbs:
                self._store._read_txt_txn = self._store.texts_db.begin(write=False)
            self._store_worker_id = worker_id

    def __call__(self, data):
        d = dict(data)
        self._ensure_store_open()
        for text_key, feature_name in self.keys_and_features.items():
            if text_key not in d:
                if not self.allow_missing_keys:
                    raise KeyError(f"Key '{text_key}' not found in data.")
                continue
            text = d[text_key]
            assert isinstance(text, str), f"{type(text)=}"
            assert self._store is not None
            text_hash = self._store.get_text_hash(text)
            feature_db = KEY_TO_DB[feature_name]
            if feature_db == "embeddings":
                features = self._store.get_embedding(text_hash)
            elif feature_db == "pool":
                features = self._store.get_pooled(text_hash)
            else:
                raise ValueError(f"Unknown feature database: {feature_db}")
            if features is None:
                raise KeyError(
                    f"Text feature not found for {text_key=} {text_hash.hex()} in "
                    f"{self.text_features_dir.as_posix()} scan file "
                    f"{data.get('filename', 'NONE')}"
                )
            assert features.dtype == torch.float16, f"{features.dtype=} != {torch.float16=}"
            d[feature_name] = features
        return d


GET_REPORT_HIDDEN_STATE = {"report": "report_hidden_state"}
GET_REPORT_POOLED = {"report": "report_pooled"}
KEY_TO_DB = {
    "report_hidden_state": "embeddings",
    "report_pooled": "pool",
}


def get_text_feature_transforms(
    keys_and_features: dict[str, str] | None = None,
    features_dataset_name: str | None = None,
    features_model_name: str | None = None,
):
    if keys_and_features is None:
        raise ValueError("keys_and_features is required, got None")
    if len(keys_and_features) == 0:
        return []
    assert features_dataset_name is not None, f"{features_dataset_name=} required"
    assert features_model_name is not None, f"{features_model_name=} required"
    features_subdir = get_text_features_subdir(features_dataset_name, features_model_name)
    log_info(f"Loading text features from subdir: {features_subdir}")
    return [
        LoadTextFeaturesTransformd(
            keys_and_features=keys_and_features,
            text_features_subdir=features_subdir,
        )
    ]


def get_text_features_subdir(features_dataset_name: str, features_model_name: str):
    """Find the folder to save the features for this dataset and model / dataloading combination."""
    features_subdir_dataset = features_dataset_name.lower()

    if features_model_name in {
        "spectre_pretrained_half_patch_embed",
        "spectre_pretrained_siglip_res",
    }:
        features_subdir_model = "spectre_zs"
        # if we have a finetuned model where we want to extract features, add here
    else:
        raise NotImplementedError(f"Model {features_model_name} no feature dir implemented")

    return f"{features_subdir_model}_{features_subdir_dataset}"
