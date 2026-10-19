import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

SQRT2 = float(np.sqrt(2.0))


def _skel_graph(skel):
    ys, xs = np.nonzero(skel)
    n = len(ys)
    idx = -np.ones(skel.shape, dtype=np.int64)
    idx[ys, xs] = np.arange(n)
    rows, cols, data = [], [], []
    h, w = skel.shape
    for dy, dx, cost in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)
        ok[ok] &= skel[y2[ok], x2[ok]]
        a = idx[ys[ok], xs[ok]]
        b = idx[y2[ok], x2[ok]]
        rows += [a, b]
        cols += [b, a]
        data += [np.full(len(a), cost)] * 2
    if not rows:
        return idx, None
    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    return idx, coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _snap(point, skel, tol):
    y, x = point
    h, w = skel.shape
    y0, y1 = max(0, y - tol), min(h, y + tol + 1)
    x0, x1 = max(0, x - tol), min(w, x + tol + 1)
    win = skel[y0:y1, x0:x1]
    if not win.any():
        return None
    yy, xx = np.nonzero(win)
    d2 = (yy + y0 - y) ** 2 + (xx + x0 - x) ** 2
    k = np.argmin(d2)
    return (yy[k] + y0, xx[k] + x0)


def apls_raster(pred, gt, thr=0.5, n_pairs=30, snap_tol=5, rng=None):
    rng = rng or np.random.default_rng(0)
    sk_g = skeletonize(gt >= 0.5)
    sk_p = skeletonize(pred >= thr)
    if sk_g.sum() < 2:
        return float("nan")
    idx_g, adj_g = _skel_graph(sk_g)
    idx_p, adj_p = _skel_graph(sk_p) if sk_p.any() else (None, None)
    gy, gx = np.nonzero(sk_g)
    penalties = []
    for _ in range(n_pairs):
        i, j = rng.integers(0, len(gy), 2)
        a_g, b_g = idx_g[gy[i], gx[i]], idx_g[gy[j], gx[j]]
        if a_g == b_g:
            continue
        d_g = dijkstra(adj_g, indices=a_g, min_only=True, directed=False)[b_g]
        if not np.isfinite(d_g) or d_g < 1:
            continue
        if adj_p is None:
            penalties.append(1.0)
            continue
        pa = _snap((gy[i], gx[i]), sk_p, snap_tol)
        pb = _snap((gy[j], gx[j]), sk_p, snap_tol)
        if pa is None or pb is None:
            penalties.append(1.0)
            continue
        d_p = dijkstra(adj_p, indices=idx_p[pa], min_only=True, directed=False)[idx_p[pb]]
        if not np.isfinite(d_p):
            penalties.append(1.0)
        else:
            penalties.append(min(1.0, abs(d_g - d_p) / d_g))
    if not penalties:
        return float("nan")
    return 1.0 - float(np.mean(penalties))


def _junctions(skel):
    from scipy.ndimage import convolve
    k = np.ones((3, 3)); k[1, 1] = 0
    nb = convolve(skel.astype(np.uint8), k, mode="constant")
    return (skel & (nb >= 3))


def junction_recall(pred, gt, thr=0.5, tol=3):
    sk_g = skeletonize(gt >= 0.5)
    sk_p = skeletonize(pred >= thr)
    jg = _junctions(sk_g)
    if not jg.any():
        return float("nan")
    jp = _junctions(sk_p)
    if not jp.any():
        return 0.0
    from scipy.ndimage import distance_transform_edt
    d_to_pred = distance_transform_edt(~jp)
    return float((d_to_pred[jg] <= tol).mean())


def graph_metrics_batch(probs, masks, thr=0.5, n_pairs=30, max_items=None):
    apls_v, jr_v = [], []
    n = len(probs) if max_items is None else min(len(probs), max_items)
    rng = np.random.default_rng(0)
    for i in range(n):
        a = apls_raster(probs[i, 0], masks[i, 0], thr=thr, n_pairs=n_pairs, rng=rng)
        j = junction_recall(probs[i, 0], masks[i, 0], thr=thr)
        if not np.isnan(a):
            apls_v.append(a)
        if not np.isnan(j):
            jr_v.append(j)
    return {"apls_r": float(np.mean(apls_v)) if apls_v else float("nan"),
            "junction_recall": float(np.mean(jr_v)) if jr_v else float("nan")}
