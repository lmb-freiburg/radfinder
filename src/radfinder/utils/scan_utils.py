"""
Scan-affine helpers.

Functions to erive window- and slice-resolution affines for axis-aware feature pooling
and to sample DICOM-defined slices out of an axis-aligned scan volume.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def downscale_affine_to_windows(
    scan_affine: np.ndarray,
    wh: int,
    ww: int,
    wd: int,
    center: bool = True,
) -> np.ndarray:
    """
    Affine for pooled-window grid given the dense scan affine.

    Assuming the scan was already cropped to a bunch of windows each of size
    wh*ww*wd, so the scan shape is hp*wh, wp*ww, dp*wd, and assuming the scan
    affine is correct for that crop, this returns the affine for the pooled
    features where one feature represents a window of size wh x ww x wd.

    Args:
        scan_affine: 4x4 affine for dense scan grid (hp*wh, wp*ww, dp*wd).
        wh: Window size in voxels along axis 0 — also the downscale factor.
        ww: Same for axis 1.
        wd: Same for axis 2.
        center: If True, window voxel represents the center of each block.
            If False, represents the block origin (top-left-front).

    Returns:
        4x4 affine for the pooled window grid.
    """
    scan_affine = np.asarray(scan_affine, dtype=np.float64)
    off_h = (wh - 1) / 2.0 if center else 0.0
    off_w = (ww - 1) / 2.0 if center else 0.0
    off_d = (wd - 1) / 2.0 if center else 0.0
    T = np.array(
        [
            [wh, 0, 0, off_h],
            [0, ww, 0, off_w],
            [0, 0, wd, off_d],
            [0, 0, 0, 1.0],
        ],
        dtype=np.float64,
    )
    return scan_affine @ T


def get_slice_from_scan_with_affines(
    scanarr: np.ndarray,
    scan_affine: np.ndarray,  # 4x4, scan IJK -> world (same frame as dicom_affine)
    dicom_affine: np.ndarray,  # 4x4, dicom (x,y,z) -> world for this plane/series
    dicom_shape_rc: tuple[int, int],  # (rows, cols)
    order: int = 1,
    mode: str = "constant",
    cval: float = -1.0,
) -> np.ndarray:
    """
    Sample one DICOM-plane slice out of an axis-aligned scan volume.

    Build dicom index points as (x, y, 0) and let dicom_affine handle
    orientation/spacing/position. Then map world -> scan IJK and trilinearly
    sample scanarr.

    Assumptions:
      - dicom_affine is correct for the target plane and maps (x,y,0) pixel
        indices to world.
      - dicom_shape_rc is (rows, cols) in DICOM layout.
      - scanarr axis order matches scan_affine (IJK -> world).
    """
    rows, cols = dicom_shape_rc
    rr, cc = np.meshgrid(
        np.arange(rows, dtype=np.float64),
        np.arange(cols, dtype=np.float64),
        indexing="ij",
    )

    # DICOM "pixel index" convention: x = col, y = row, z = 0 (plane)
    x = cc.reshape(-1)
    y = rr.reshape(-1)
    z = np.zeros_like(x)

    dicom_xyz_h = np.vstack([x, y, z, np.ones_like(x)])  # 4xN

    # dicom -> world -> scan continuous ijk
    world_h = dicom_affine @ dicom_xyz_h
    scan_ijk = (np.linalg.inv(scan_affine) @ world_h)[:3, :]

    sampled = map_coordinates(
        scanarr,
        coordinates=scan_ijk,
        order=order,
        mode=mode,
        cval=cval,
    )

    return sampled.reshape(rows, cols)
