import numpy as np
from scipy.ndimage import convolve, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

GSD = 5.8
SQRT2 = float(np.sqrt(2.0))
K8 = np.ones((3, 3), np.uint8); K8[1, 1] = 0
CELL = 9
SPUR_PX = 3
MIN_COMP_PX = 9


def clean_skeleton(mask, spur_px=SPUR_PX, min_comp_px=MIN_COMP_PX, already_thin=False):
    sk = mask.astype(bool) if already_thin else skeletonize(mask.astype(bool))
    if not sk.any():
        return sk
    lab, n = label(sk, structure=np.ones((3, 3)))
    if n:
        sizes = np.bincount(lab.ravel())
        drop = sizes < min_comp_px; drop[0] = False
        sk &= ~drop[lab]
    if spur_px > 0 and sk.any():
        deg = convolve(sk.astype(np.uint8), K8, mode="constant") * sk
        H, W = sk.shape
        kill = []
        for y, x in zip(*np.nonzero(deg == 1)):
            path, prev, cur = [(y, x)], None, (y, x)
            for _ in range(spur_px):
                nb = [(cur[0] + dy, cur[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                      if (dy or dx) and 0 <= cur[0] + dy < H and 0 <= cur[1] + dx < W
                      and sk[cur[0] + dy, cur[1] + dx] and (cur[0] + dy, cur[1] + dx) != prev]
                if not nb:
                    break
                nxt = nb[0] if len(nb) == 1 else None
                if nxt is None or deg[nxt] >= 3:
                    if nxt is not None or len(nb) > 1:
                        kill.extend(path)
                    break
                prev, cur = cur, nxt
                path.append(cur)
        if kill:
            ky, kx = zip(*kill)
            sk[np.array(ky), np.array(kx)] = False
    return sk


class SkelGraph:

    def __init__(self, sk):
        ys, xs = np.nonzero(sk)
        self.n = len(ys)
        self.yx = np.stack([ys, xs], 1)
        self.idx = -np.ones(sk.shape, np.int64)
        self.idx[ys, xs] = np.arange(self.n)
        rows, cols, w = [], [], []
        H, W = sk.shape
        for dy, dx, c in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2)):
            y2, x2 = ys + dy, xs + dx
            ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
            ok[ok] &= sk[y2[ok], x2[ok]]
            a, b = self.idx[ys[ok], xs[ok]], self.idx[y2[ok], x2[ok]]
            rows += [a, b]; cols += [b, a]; w += [np.full(len(a), c * GSD)] * 2
        if self.n and rows:
            self.adj = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(self.n, self.n)).tocsr()
        else:
            self.adj = None
        if self.n:
            deg = convolve(sk.astype(np.uint8), K8, mode="constant")[ys, xs]
            special = np.flatnonzero((deg == 1) | (deg >= 3))
            cell = (ys // CELL) * (W // CELL + 1) + (xs // CELL)
            _, first = np.unique(cell, return_index=True)
            self.cp = np.unique(np.concatenate([special, first]))
        else:
            self.cp = np.zeros(0, np.int64)
        self.tree = cKDTree(self.yx) if self.n else None


def _dist(adj, sources, targets, chunk=32):
    out = np.empty((len(sources), len(targets)), np.float32)
    for i in range(0, len(sources), chunk):
        d = dijkstra(adj, directed=False, indices=sources[i:i + chunk])
        out[i:i + chunk] = d[:, targets]
    return out


def directional(src, dst, buffers, rng, smax=300):
    n = len(src.cp)
    out = {b: (np.nan, np.zeros(0)) for b in buffers}
    if n < 2 or src.adj is None:
        return out
    if dst.n == 0:
        near, dnear = -np.ones(n, np.int64), np.full(n, np.inf)
    else:
        dnear, near = dst.tree.query(src.yx[src.cp], distance_upper_bound=max(buffers) + 1e-6)
        near = np.where(np.isfinite(dnear), near, -1)
    S = np.arange(n) if n <= smax else np.sort(rng.choice(n, smax, replace=False))
    Ls = _dist(src.adj, src.cp[S], src.cp)
    Ld = np.full(Ls.shape, np.inf, np.float32)
    ms, mt = near[S], near
    r_ok, c_ok = ms >= 0, mt >= 0
    if dst.adj is not None and r_ok.any() and c_ok.any():
        Ld[np.ix_(r_ok, c_ok)] = _dist(dst.adj, ms[r_ok], mt[c_ok])
    valid = np.isfinite(Ls) & (Ls > 0)
    valid[np.arange(len(S)), S] = False
    if not valid.any():
        return out
    Lsv = Ls[valid].astype(np.float64)
    Ldv_all = Ld[valid].astype(np.float64)
    ii, jj = np.nonzero(valid)
    for b in buffers:
        snap = np.isfinite(dnear) & (dnear <= b + 1e-6)
        ok = snap[S][ii] & snap[jj]
        Ldv = np.where(ok, Ldv_all, np.inf)
        pen = np.where(np.isfinite(Ldv), np.minimum(1.0, np.abs(Lsv - Ldv) / Lsv), 1.0)
        out[b] = (1.0 - pen.mean(), pen)
    return out


def apls_window(gt_sk, pr_sk, buffers=(2,), rng=None, smax=300):
    rng = rng or np.random.default_rng(0)
    if np.isscalar(buffers):
        buffers = (buffers,)
    G, P = SkelGraph(gt_sk), SkelGraph(pr_sk)
    d1 = directional(G, P, buffers, rng, smax)
    if np.isnan(d1[buffers[0]][0]):
        return None
    d2 = directional(P, G, buffers, rng, smax)
    res = {}
    for b in buffers:
        c1, pen1 = d1[b]; c2, pen2 = d2[b]
        apls = 0.0 if (np.isnan(c2) or c1 <= 0 or c2 <= 0) else 2 * c1 * c2 / (c1 + c2)
        res[b] = {"apls": apls, "gt2pr": c1, "pr2gt": (np.nan if np.isnan(c2) else c2),
                  "n_gt_pairs": len(pen1), "n_pr_pairs": len(pen2),
                  "route10": float((pen1 <= 0.10).mean()), "route25": float((pen1 <= 0.25).mean()),
                  "gt_len_km": G.n * GSD / 1000, "pr_len_km": P.n * GSD / 1000}
    return res


def _selftest():
    rng = np.random.default_rng(0)
    gt = np.zeros((256, 256), bool)
    gt[128, 10:246] = True; gt[10:246, 128] = True; gt[60, 40:200] = True; gt[40:200, 60] = True
    g = clean_skeleton(gt, already_thin=True)
    r = apls_window(g, g.copy())[2]; print("identity          ", round(r["apls"], 4), r["gt2pr"], r["pr2gt"])
    sh = np.roll(g, 1, axis=0); r = apls_window(g, clean_skeleton(sh, already_thin=True))[2]
    print("1-px shift        ", round(r["apls"], 4))
    cut = g.copy(); cut[:, 120:136] = False; r = apls_window(g, clean_skeleton(cut, already_thin=True))[2]
    print("16-px cut band    ", round(r["apls"], 4), "gt2pr", round(r["gt2pr"], 3), "pr2gt", round(r["pr2gt"], 3))
    extra = g.copy(); extra[200, 10:246] = True; extra[10:246, 220] = True
    r = apls_window(g, clean_skeleton(extra, already_thin=True))[2]
    print("spurious roads    ", round(r["apls"], 4), "gt2pr", round(r["gt2pr"], 3), "pr2gt", round(r["pr2gt"], 3))
    r = apls_window(g, np.zeros_like(g))[2]; print("empty prediction  ", r["apls"])
    thick = np.zeros_like(gt)
    for dy in (-1, 0, 1):
        thick |= np.roll(gt, dy, 0)
    r = apls_window(g, clean_skeleton(thick))[2]; print("3-px thick pred   ", round(r["apls"], 4))
    rr = apls_window(g, clean_skeleton(sh, already_thin=True), buffers=(0, 1, 2, 3))
    print("1-px shift by buffer", {b: round(v["apls"], 3) for b, v in rr.items()})


if __name__ == "__main__":
    _selftest()
