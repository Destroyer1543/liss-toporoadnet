"""Derive multi-target supervision (skeleton, width, affinity) from a binary road mask."""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

# 8-neighbor offsets, order fixed across the codebase: E, NE, N, NW, W, SW, S, SE
NEIGHBOR_OFFSETS = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]


def make_targets(mask: np.ndarray) -> dict:
    """mask: (H, W) binary {0,1}. Returns float32 arrays:
    skel (H,W), width (H,W) half-width in px (valid on skeleton), aff (8,H,W).
    """
    mask = (mask > 0).astype(np.uint8)
    skel = skeletonize(mask.astype(bool)).astype(np.float32)
    # half-width from the distance transform (distance to background)
    edt = distance_transform_edt(mask).astype(np.float32)
    width = edt * skel  # supervised only on skeleton pixels
    aff = np.zeros((8, *mask.shape), dtype=np.float32)
    for k, (dy, dx) in enumerate(NEIGHBOR_OFFSETS):
        shifted = np.zeros_like(skel)
        ys = slice(max(dy, 0), skel.shape[0] + min(dy, 0))
        xs = slice(max(dx, 0), skel.shape[1] + min(dx, 0))
        ys_src = slice(max(-dy, 0), skel.shape[0] + min(-dy, 0))
        xs_src = slice(max(-dx, 0), skel.shape[1] + min(-dx, 0))
        shifted[ys, xs] = skel[ys_src, xs_src]
        aff[k] = skel * shifted
    return {"skel": skel, "width": width, "aff": aff}
