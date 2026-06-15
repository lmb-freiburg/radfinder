import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform


class SelectRandomScand(MapTransform):
    """Select a random scan out of multiple for a datapoint."""

    def __init__(self, keys="image") -> None:
        super().__init__(keys, allow_missing_keys=True)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        if "all_images" in d:
            all_images = d.pop("all_images")
            image_num = torch.randint(len(all_images), (1,)).item()
            selected_image = all_images[image_num]
            d["image"] = selected_image
        return d
