"""Affinity-guided graph post-processing for LISS-TopoRoadNet outputs.

Turns raw per-pixel predictions into a cleaner road graph using the model's
own topology heads (skeleton, 8-neighbour affinity, width) — heads the raster
baselines do not have. Pipeline:
  1. skeletonize thresholded mask
  2. prune spur branches shorter than `min_spur`
  3. bridge gaps: for each skeleton endpoint, extend along the affinity /
     local-direction cone up to `max_gap` px and connect to a nearby skeleton
     pixel if the mask supports the link
  4. (optional) re-widen the cleaned skeleton with the predicted width map

Designed for improving graph metrics (APLS, junction recall) at inference with
no retraining.
"""

import numpy as np
from scipy.ndimage import convolve
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize

# same order as targets.NEIGHBOR_OFFSETS: E, NE, N, NW, W, SW, S, SE
NEIGHBOR_OFFSETS = np.array([(0, 1), (-1, 1), (-1, 0), (-1, -1),
                             (0, -1), (1, -1), (1, 0), (1, 1)], dtype=np.float64)
_K8 = np.ones((3, 3), dtype=np.uint8); _K8[1, 1] = 0


def _neighbor_count(skel):
    return convolve(skel.astype(np.uint8), _K8, mode="constant")


def prune_spurs(skel, min_spur=6):
    """Remove branch segments that dead-end (degree-1) and are shorter than min_spur."""
    skel = skel.copy()
    h, w = skel.shape
    for _ in range(min_spur):  # iterative endpoint erosion, bounded so long roads survive
        nb = _neighbor_count(skel)
        endpoints = np.argwhere(skel & (nb == 1))
        if len(endpoints) == 0:
            break
        removed = False
        for y, x in endpoints:
            # walk the branch; stop at a junction (deg>=3) or after min_spur steps
            path = [(y, x)]
            cy, cx, prev = y, x, None
            for _ in range(min_spur):
                nbrs = [(cy + int(dy), cx + int(dx)) for dy, dx in NEIGHBOR_OFFSETS
                        if 0 <= cy + int(dy) < h and 0 <= cx + int(dx) < w
                        and skel[cy + int(dy), cx + int(dx)] and (cy + int(dy), cx + int(dx)) != prev]
                if len(nbrs) != 1:
                    break  # junction or dead-loop -> this is a real branch, keep
                prev = (cy, cx); cy, cx = nbrs[0]; path.append((cy, cx))
            else:
                # walked min_spur steps without hitting a junction -> long enough, keep
                continue
            if _neighbor_count(skel)[cy, cx] >= 3 or len(path) < min_spur:
                for py, px in path[:-1]:
                    skel[py, px] = 0
                removed = True
        if not removed:
            break
    return skel


def _affinity_dir(aff, y, x, back=None):
    """Unit road direction at (y,x) from 8-ch affinity, tie-broken away from `back`."""
    a = aff[:, y, x]
    v = (a[:, None] * NEIGHBOR_OFFSETS).sum(0)
    if np.hypot(*v) < 1e-3 and back is not None:
        v = np.array([y - back[0], x - back[1]], dtype=np.float64)
    n = np.hypot(*v)
    return v / n if n > 1e-6 else np.array([0.0, 0.0])


def bridge_gaps(skel, aff, prob, max_gap=8, cone_cos=0.5, min_prob=0.3):
    """Connect endpoints across short gaps guided by the affinity direction field."""
    skel = skel.copy()
    h, w = skel.shape
    nb = _neighbor_count(skel)
    endpoints = [tuple(p) for p in np.argwhere(skel & (nb == 1))]
    ep_set = set(endpoints)
    for (y, x) in endpoints:
        if not skel[y, x]:
            continue
        # local incoming direction (from one step back)
        back = None
        for dy, dx in NEIGHBOR_OFFSETS.astype(int):
            if 0 <= y + dy < h and 0 <= x + dx < w and skel[y + dy, x + dx]:
                back = (y + dy, x + dx); break
        d = _affinity_dir(aff, y, x, back)
        if d[0] == 0 and d[1] == 0:
            continue
        best, best_score = None, -1
        for gy in range(max(0, y - max_gap), min(h, y + max_gap + 1)):
            for gx in range(max(0, x - max_gap), min(w, x + max_gap + 1)):
                if not skel[gy, gx] or (gy, gx) == (y, x):
                    continue
                vec = np.array([gy - y, gx - x], dtype=np.float64)
                dist = np.hypot(*vec)
                if dist < 1 or dist > max_gap:
                    continue
                cos = float(np.dot(vec / dist, d))
                if cos < cone_cos:
                    continue
                rr, cc = draw_line(y, x, gy, gx)
                seg_prob = prob[rr, cc].mean()
                if seg_prob < min_prob:
                    continue
                score = cos + seg_prob  # prefer aligned + well-supported links
                # prefer connecting to another endpoint
                if (gy, gx) in ep_set:
                    score += 0.5
                if score > best_score:
                    best_score, best = score, (rr, cc)
        if best is not None:
            skel[best] = 1
    return skel


def widen(skel, width, k=9):
    """Re-dilate skeleton by the predicted half-width (differentiable-free version)."""
    from scipy.ndimage import distance_transform_edt
    if skel.sum() == 0:
        return skel.astype(np.uint8)
    # distance from every pixel to nearest skeleton px, and that px's width
    inv = ~skel.astype(bool)
    dist, (iy, ix) = distance_transform_edt(inv, return_indices=True)
    nearest_w = width[iy, ix]
    return (dist <= nearest_w).astype(np.uint8)


def postprocess(prob, skel_prob, aff, width, thr=0.5, skel_thr=0.4,
                min_spur=6, max_gap=8, do_widen=True):
    """Full pipeline. Returns dict with cleaned skeleton and (optional) mask."""
    base = (prob >= thr)
    sk = skeletonize(base) if base.any() else base.astype(bool)
    sk = prune_spurs(sk, min_spur)
    sk = bridge_gaps(sk, aff, prob, max_gap=max_gap)
    out = {"skeleton": sk.astype(np.uint8)}
    if do_widen:
        out["mask"] = widen(sk, np.clip(width, 0.5, 5.0))
    return out
