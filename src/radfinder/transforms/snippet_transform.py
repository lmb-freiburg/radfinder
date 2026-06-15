"""
Load snippets for slices.

Input data:

{
    "990713e2370e6fa50c080c90ab219b15": {
        "shape": [638, 854],
        "dcmaffine": [
            [-0.445, -0.019, -0.056, 197.595],
            [0.019, -0.445, 0.0, 310.215],
            [-0.012, -0.0, 1.999, 1799.125],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "axcodes": "LPS",
        "slice_axis": 2,
        "de_cleanhist": [
            "Dichter imponierende subpleurale Verdichtung..."
        ],
        "en_cleanhist": [
            "Dense, impressive subpleural consolidation..."
        ],
    },
    "7d4af62297d590271c92e05e71d74c09": {
        "shape": [638, 854],
        "dcmaffine": [
            [-0.445, -0.019, -0.056, 196.872],
            [0.019, -0.445, 0.0, 310.218],
            [-0.012, -0.0, 1.999, 1825.113],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "axcodes": "LPS",
        "slice_axis": 2,
        "de_cleanhist": ["Kleine pleurale Ausziehungen..."],
        "en_cleanhist": ["Small pleural extensions..."],
    },
}

"""

from typing import Any, Hashable, Mapping

from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable
from radfinder.transforms.shared_utils import Language


class RandomSnippetTransformd(Randomizable, MapTransform):
    def __init__(
        self,
        keys: KeysCollection = ("slices",),
        language: str = Language.EN,
        allow_missing_keys: bool = False,
        is_train: bool = True,
    ):
        language = Language.verify_value(language)

        super().__init__(keys, allow_missing_keys)
        self.language = language
        self.is_train = is_train
        self._rand_state = {}

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        assert len(self.keys) == 1, f"{self.keys=} {len(self.keys)=}"
        key = self.keys[0]
        slicesdata = data.get(key, None)
        if slicesdata is None:
            if self.allow_missing_keys:
                return data
            raise KeyError(f"Key '{key}' not found in data. Available keys: {list(data.keys())}")
        if len(slicesdata) == 0:
            # some scans don't have snippets
            return data

        for _slicersopid, slicedata in slicesdata.items():
            # get texts for this slice and language setting
            if self.language == Language.BOTH:
                assert self.is_train, "Language.BOTH is only supported in training"
                selected_language = self.R.choice([Language.EN, Language.DE])
            else:
                selected_language = self.language
            textlist = slicedata[f"{selected_language}_cleanhist"]
            # select one of the texts
            if self.is_train:
                str_out = self.R.choice(textlist)
            else:
                str_out = textlist[0]
            slicedata["snippet"] = str_out
        return data
