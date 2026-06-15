"""
HuggingFace config class for radfinder.

Stores everything `RadFinderModel` needs to construct itself, plus image preprocessing.
"""

from transformers import PretrainedConfig

DEFAULT_IMAGE_PREPROCESSING = {
    # MONAI eval pipeline parameters (mirror configs/models/spectre_pretrained_*.yaml).
    "pixdim": [0.75, 0.75, 3.0],
    "sliding_window_size": [128, 128, 32],
    "intensity_a_min": -1000.0,
    "intensity_a_max": 1000.0,
    "intensity_b_min": 0.0,
    "intensity_b_max": 1.0,
    "orientation_axcodes": "RAS",
    "min_area_for_padding": 0.0,
    "dtype": "float16",
}

DEFAULT_TEXT_TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"


class RadFinderConfig(PretrainedConfig):
    model_type = "radfinder"

    def __init__(
        self,
        radfinder_model_config: dict | None = None,
        image_preprocessing: dict | None = None,
        text_tokenizer_name: str = DEFAULT_TEXT_TOKENIZER,
        do_snippet_alignment: dict | None = None,
        model_settings: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.radfinder_model_config = radfinder_model_config or {}
        self.image_preprocessing = image_preprocessing or dict(DEFAULT_IMAGE_PREPROCESSING)
        self.text_tokenizer_name = text_tokenizer_name
        self.do_snippet_alignment = do_snippet_alignment
        self.model_settings = model_settings
