from monai.transforms.transform import MapTransform


class NotLoadImaged(MapTransform):
    """
    Delete imagepath from dataset in case it's not needed
    """

    def __init__(
        self,
        renamer: tuple[tuple[str, str], ...] = (("image", "filename"),),
        *args,
        **kwargs,
    ) -> None:
        keys = tuple(k0 for k0, _ in renamer)
        super().__init__(keys, allow_missing_keys=True)
        self.renamer = renamer

    def __call__(self, data, *args, **kwargs) -> dict:
        d = dict(data)
        # print(f"Deleter got data: {d}")
        for k0, k1 in self.renamer:
            if k0 in d:
                d[k1] = d.pop(k0)
        # print(f"Deleter returning data: {d}")
        return d
