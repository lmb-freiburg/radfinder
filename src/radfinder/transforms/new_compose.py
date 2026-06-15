"""
Add modified versions of Compose
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from timeit import default_timer
from typing import Any

from monai.config import NdarrayOrTensor
from monai.transforms import Compose
from monai.transforms.lazy.functional import apply_pending_transforms
from monai.transforms.traits import ThreadUnsafe
from monai.transforms.transform import apply_transform
from radfinder.transforms.repr_compose import repr_compose
from radfinder.utils.logging_utils import log_error, log_info

from packg.misc import format_exception_with_chain
from visiontext.distutils import get_torch_worker_id


class ReprCompose(Compose):
    def __repr__(self):
        return repr_compose(self)


class SafeCompose(Compose):
    """
    A Compose that skips broken files without dying.
    """

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        except Exception as e:
            log_error(
                f"ERROR: Monai transform failed with error: "
                f"{format_exception_with_chain(e, with_source=True)}\n"
                f"Input was: {args=}, {kwargs=}"
            )
            return None

    def __repr__(self):
        return repr_compose(self)


class TimedCompose(Compose):
    """
    A Compose that times each transform and logs the time taken.
    """

    def __init__(self, *args, min_time: float = 0.1, **kwargs):
        self.min_time = min_time
        super().__init__(*args, **kwargs)

    def __call__(self, input_, start=0, end=None, threading=False, lazy: bool | None = None):
        _lazy = self._lazy if lazy is None else lazy
        result = execute_compose_timed(
            input_,
            transforms=self.transforms,
            start=start,
            end=end,
            map_items=self.map_items,
            unpack_items=self.unpack_items,
            lazy=_lazy,
            overrides=self.overrides,
            threading=threading,
            log_stats=self.log_stats,
            min_time=self.min_time,
        )
        return result

    def __repr__(self):
        return repr_compose(self)


class SafeTimedCompose(TimedCompose):
    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        except Exception as e:
            log_error(
                f"ERROR: Monai transform failed with error: "
                f"{format_exception_with_chain(e, with_source=True)}\n"
                f"Input was: {args=}, {kwargs=}"
            )
            return None

    def __repr__(self):
        return repr_compose(self)


def execute_compose_timed(
    data: NdarrayOrTensor | Sequence[NdarrayOrTensor] | Mapping[Any, NdarrayOrTensor],
    transforms: Sequence[Any],
    map_items: bool | int = True,
    unpack_items: bool = False,
    start: int = 0,
    end: int | None = None,
    lazy: bool | None = False,
    overrides: dict | None = None,
    threading: bool = False,
    log_stats: bool | str = False,
    min_time: float = 0.0,
) -> NdarrayOrTensor | Sequence[NdarrayOrTensor] | Mapping[Any, NdarrayOrTensor]:
    end_ = len(transforms) if end is None else end
    if start is None:
        raise ValueError(f"'start' ({start}) cannot be None")
    if start < 0:
        raise ValueError(f"'start' ({start}) cannot be less than 0")
    if start > end_:
        raise ValueError(f"'start' ({start}) must be less than 'end' ({end_})")
    if end_ > len(transforms):
        raise ValueError(
            f"'end' ({end_}) must be less than or equal to the transform count ({len(transforms)}"
        )

    # no-op if the range is empty
    if start == end:
        return data

    t0 = default_timer()
    tlast = t0
    wid = get_torch_worker_id()
    print_worker(min_time, t0, tlast, wid, lazy, f"Start executing composes {start} to {end}")
    for _transform in transforms[start:end]:
        if threading:
            _transform = (
                deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
            )
        data = apply_transform(
            _transform,
            data,
            map_items,
            unpack_items,
            lazy=lazy,
            overrides=overrides,
            log_stats=log_stats,
        )
        print_worker(
            min_time, t0, tlast, wid, lazy, f"Applied transform {_transform.__class__.__name__}"
        )
        tlast = default_timer()
    data = apply_pending_transforms(data, None, overrides, logger_name=log_stats)
    print_worker(min_time, t0, tlast, wid, lazy, f"Applied pending transforms")
    return data


def print_worker(min_time, t0, tlast, wid, lazy, *args):
    t1 = default_timer()
    elapsed_total = t1 - t0
    elapsed_here = t1 - tlast
    if elapsed_here >= min_time:
        log_info(
            f"[Worker {wid}] lazy={lazy} total={elapsed_total:.1f}s this={elapsed_here:.1f}s: "
            + " ".join(str(arg) for arg in args)
        )
