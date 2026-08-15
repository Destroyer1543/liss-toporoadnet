"""Validate the raster APLS_r against a standard graph-based APLS.

Graph APLS builds a junction-node graph (nodes = skeleton pixels of degree != 2,
edges = the skeleton paths between them, weighted by pixel length), which is the
standard APLS formulation, and compares shortest-path lengths between sampled node
pairs. We correlate it with APLS_r on held-out test tiles and plot the result.
"""
import sys, pathlib, numpy as np
sys.path.insert(0, "src")
import torch, networkx as nx
from scipy.ndimage import convolve
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from graph_metrics import apls_raster
from model import LISSTopoRoadNet

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def skel_to_graph(skel):
    """junction-node graph: nodes = skeleton pixels with degree != 2, edges = traced paths."""
    deg = convolve(skel.astype(int), K8, mode="constant") * skel
    node_mask = skel & (deg != 2)
    ys, xs = np.nonzero(node_mask)
    nid = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    G = nx.Graph()
    for (y, x), i in nid.items():
        G.add_node(i, pos=(y, x))
    H, W = skel.shape

    def nbrs(y, x):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx_ < W and skel[ny, nx_]:
                        yield ny, nx_

    for (y, x) in list(nid):
        for start in nbrs(y, x):
            prev, cur = (y, x), start
            length = 1
            while cur not in nid:
                nxts = [p for p in nbrs(*cur) if p != prev]
                if len(nxts) != 1:
                    break
                prev, cur = cur, nxts[0]
                length += 1
            if cur in nid:
                a, b = nid[(y, x)], nid[cur]
                if a != b and (not G.has_edge(a, b) or length < G[a][b]["weight"]):
                    G.add_edge(a, b, weight=length)
    return G


def _snap_nodes(G, pos_list):
    if G.number_of_nodes() == 0:
        return None, None
    ids = list(G.nodes)
    pts = np.array([G.nodes[i]["pos"] for i in ids])
    return cKDTree(pts), ids


def graph_apls(gt, pr, n_pairs=120, snap_tol=8, rng=None):
    rng = rng or np.random.default_rng(0)
    Gg = skel_to_graph(skeletonize(gt >= 0.5))
    Gp = skel_to_graph(skeletonize(pr >= 0.5))
    if Gg.number_of_nodes() < 2:
        return float("nan")
    tree_p, ids_p = _snap_nodes(Gp, None)

    def one_dir(Gsrc, Gdst, tree_dst, ids_dst):
        src = list(Gsrc.nodes)
        if len(src) < 2:
            return []
        Ldst = dict(nx.all_pairs_dijkstra_path_length(Gdst, weight="weight")) if Gdst.number_of_nodes() else {}
        pen = []
        for _ in range(n_pairs):
            a, b = src[rng.integers(len(src))], src[rng.integers(len(src))]
            if a == b:
                continue
            try:
                Ls = nx.shortest_path_length(Gsrc, a, b, weight="weight")
            except nx.NetworkXNoPath:
                continue
            if Ls <= 0:
                continue
            if tree_dst is None:
                pen.append(1.0); continue
            da, ia = tree_dst.query(Gsrc.nodes[a]["pos"])
            db, ib = tree_dst.query(Gsrc.nodes[b]["pos"])
            if da > snap_tol or db > snap_tol:
                pen.append(1.0); continue
            na, nb = ids_dst[ia], ids_dst[ib]
            Ld = Ldst.get(na, {}).get(nb, None)
            pen.append(1.0 if Ld is None else min(1.0, abs(Ls - Ld) / Ls))
        return pen

    tree_g, ids_g = _snap_nodes(Gg, None)
    pen = one_dir(Gg, Gp, tree_p, ids_p) + one_dir(Gp, Gg, tree_g, ids_g)
    return float(1.0 - np.mean(pen)) if pen else float("nan")


def main():
    m = LISSTopoRoadNet().to(DEV)
    m.load_state_dict(torch.load("runs/exp15_toporoad_full/best.pt", map_location=DEV,
                                 weights_only=False)["model"]); m.eval()
    tiles = sorted(pathlib.Path("/data/road_liss4/liss4_tiles/test").glob("*.npz"))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(tiles))
    rr, gg = [], []
    n = 0
    for k in idx:
        d = np.load(tiles[k]); img = d["image"].astype(np.float32); gt = d["mask"].astype(bool)
        if gt.mean() < 0.03:          # road-rich tiles give lower-variance per-tile APLS
            continue
        with torch.no_grad():
            pr = (torch.sigmoid(m(torch.from_numpy(img[None]).to(DEV))["fused_logit"])[0, 0] > 0.5).cpu().numpy()
        a = apls_raster(pr, gt, n_pairs=200, rng=np.random.default_rng(1))
        b = graph_apls(gt, pr, n_pairs=200, rng=np.random.default_rng(2))
        if np.isfinite(a) and np.isfinite(b):
            rr.append(a); gg.append(b); n += 1
        if n >= 140:
            break
    rr, gg = np.array(rr), np.array(gg)
    r = np.corrcoef(rr, gg)[0, 1]
    rho = None
    try:
        from scipy.stats import spearmanr
        rho = spearmanr(rr, gg).statistic
    except Exception:
        pass
    np.savez("/data/road_liss4/apls_corr.npz", aplsr=rr, aplsg=gg, pearson=r, spearman=rho)
    print(f"n={n}  Pearson r={r:.3f}  Spearman={rho:.3f}  "
          f"APLSr mean={rr.mean():.3f} graphAPLS mean={gg.mean():.3f}")


if __name__ == "__main__":
    main()
