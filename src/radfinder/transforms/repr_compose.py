"""
Compose that represents transforms properly
"""

from __future__ import annotations

import inspect

from monai.transforms import Compose
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import LazyTransform


def public_data_attrs(
    obj,
    *,
    recurse_types=(),
    compose_types=(),
    prefix="",
    _seen=None,
):
    """
    Return (name, value) for all non-private, non-callable attributes on obj.

    If an attribute value is an instance of `recurse_types`, recurse into it and
    return flattened names like: "<SubClassName>.<sub_attr_name>".
    (You can change the naming scheme below if you prefer using the parent attr name.)

    Cycle-safe via object id tracking.
    """
    if _seen is None:
        _seen = set()

    oid = id(obj)
    if oid in _seen:
        return []
    _seen.add(oid)

    out = []
    for name, value in inspect.getmembers(obj):
        if name.startswith("_"):
            continue

        # Recurse into nested transforms (InvertibleTransform/LazyTransform, etc.)
        if recurse_types and isinstance(value, recurse_types):
            sub_prefix = f"{prefix}{value.__class__.__name__}."
            out.extend(
                public_data_attrs(
                    value,
                    recurse_types=recurse_types,
                    compose_types=compose_types,
                    prefix=sub_prefix,
                    _seen=_seen,
                )
            )
            continue

        # Optionally: skip printing nested composes as attrs (they'll be handled by repr_compose)
        if compose_types and isinstance(value, compose_types):
            continue

        if callable(value):
            continue

        out.append((f"{prefix}{name}", value))

    return out


def repr_compose(
    compose,
    indent=0,
    *,
    recurse_types=(InvertibleTransform, LazyTransform),
    compose_types=(Compose,),
    _seen=None,
):
    """
    Pretty repr for Compose-like objects, supporting recursive/nested composes.

    - recurse_types: tuple of types (e.g., (InvertibleTransform, LazyTransform))
      whose instances should be flattened in public_data_attrs.
    - compose_types: tuple of types (e.g., (Compose,)) treated as composes for recursion.
    """
    if _seen is None:
        _seen = set()

    oid = id(compose)
    if oid in _seen:
        return " " * indent + "<recursive compose>"
    _seen.add(oid)

    cls = compose.__class__.__name__
    lazy_val = getattr(compose, "_lazy", getattr(compose, "lazy", None))

    lines = [f"{cls}(lazy={lazy_val!r}, transforms=("]

    for ti, t in enumerate(getattr(compose, "transforms", ())):
        # Nested compose support
        if compose_types and isinstance(t, compose_types):
            nested = repr_compose(
                t,
                indent=indent + 2,
                recurse_types=recurse_types,
                compose_types=compose_types,
                _seen=_seen,
            )
            lines.append(nested.rstrip() + ("," if ti < len(compose.transforms) - 1 else ""))
            continue

        # Normal transform
        lines.append(f"  {t.__class__.__name__}(")

        attrs = public_data_attrs(
            t,
            recurse_types=recurse_types,
            compose_types=compose_types,
            _seen=_seen,
        )
        for i, (name, value) in enumerate(attrs):
            comma = "," if i < len(attrs) - 1 else ""
            lines.append(f"    {name}={value!r}{comma}")

        lines.append("  )" + ("," if ti < len(compose.transforms) - 1 else ""))

    lines.append("))")
    return "\n".join((" " * indent) + line for line in lines)
