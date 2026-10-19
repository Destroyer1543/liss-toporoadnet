import contextlib, io, json, os, pathlib, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import convolve, label
import networkx as nx
from shapely.geometry import LineString

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
REF = pathlib.Path(os.environ.get("APLS_REF", "third_party/apls"))
sys.path.insert(0, str(REF))
import stubs
import apls
from vector_apls import clean_skeleton
from vector_apls_run import load_gt, unpack, MOS, windows, EMASK
from vector_apls_run import EXT as EXT_DIR


def ignore_mask(scene, H, W):
    if (EXT_DIR / f"{scene}.npz").exists():
        d = np.load(EXT_DIR / f"{scene}.npz"); return unpack(d["ignore"], tuple(d["shape"]))[:H, :W]
    return np.load(EMASK / f"{scene}.npz")["ignore"][:H, :W].astype(bool)

GSD = 5.8
SPEED = 13.41
SNAPS = (11.6, 4.0, 17.4)
TEST = ["R2F24AUG2023064073010100066SSANSTUC00GTDD", "RAF23OCT2023035685010100064SSANSTUC00GTDA"]
EXT = ["R2F13DEC2025076029011100053SSANSTUC00GTDA", "R2F25JAN2026076640011000053SSANSTUC00GTDD",
       "RAF18JAN2026047300011100053SSANSTUC00GTDC", "RAF20DEC2025046888011000052SSANSTUC00GTDD"]
RUNS = ["exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2",
        "exp16_unet_full", "exp23_unet_seed1", "exp25_unet_seed2",
        "exp17_dlinknet_full", "exp36_dlinknet_seed1", "exp37_dlinknet_seed2", "exp26_roadformer_full"]
N_PER_SCENE = 60
OUT = pathlib.Path("data/rebuttal/official_apls")
K8 = np.ones((3, 3), int); K8[1, 1] = 0
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skel_to_graph(sk):
    sk = sk.astype(bool)
    H, W = sk.shape
    deg = convolve(sk.astype(int), K8, mode="constant") * sk
    node_px = sk & (deg != 2)
    comp, nc = label(sk, structure=np.ones((3, 3)))
    if nc:
        has_node = np.zeros(nc + 1, bool); has_node[np.unique(comp[node_px])] = True
        for c in range(1, nc + 1):
            if not has_node[c]:
                r, cc = np.argwhere(comp == c)[0]; node_px[r, cc] = True
    nlab, nn = label(node_px, structure=np.ones((3, 3)))
    G = nx.MultiGraph()
    cent = {}
    for i in range(1, nn + 1):
        rr, cc = np.nonzero(nlab == i)
        cent[i] = ((cc.mean() + 0.5) * GSD, (rr.mean() + 0.5) * GSD)
        G.add_node(i, x=cent[i][0], y=cent[i][1])
    pt = lambda r, c: ((c + 0.5) * GSD, (r + 0.5) * GSD)

    def add_edge(a, b, coords):
        line = LineString([cent[a]] + coords + [cent[b]])
        if a == b and len(coords) < 3:
            return
        if line.length <= 0:
            return
        G.add_edge(a, b, geometry=line, length=float(line.length), inferred_speed_mps=SPEED)

    visited = np.zeros_like(sk)
    direct = set()
    for r0, c0 in np.argwhere(node_px):
        a = nlab[r0, c0]
        for dr, dc in NB:
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < H and 0 <= c < W) or not sk[r, c]:
                continue
            if node_px[r, c]:
                b = nlab[r, c]
                if b != a and (min(a, b), max(a, b)) not in direct:
                    direct.add((min(a, b), max(a, b))); add_edge(a, b, [])
                continue
            if visited[r, c]:
                continue
            path, prev, cur = [(r, c)], (r0, c0), (r, c)
            visited[r, c] = True
            end = None
            while end is None:
                nxt = None
                for er, ec in NB:
                    q = (cur[0] + er, cur[1] + ec)
                    if q == prev or not (0 <= q[0] < H and 0 <= q[1] < W) or not sk[q]:
                        continue
                    if node_px[q]:
                        if nlab[q] == a and len(path) == 1:
                            continue
                        end = nlab[q]; break
                    if not visited[q]:
                        nxt = q
                if end is None:
                    if nxt is None:
                        nn += 1; end = nn
                        cent[end] = pt(*cur); G.add_node(end, x=cent[end][0], y=cent[end][1])
                        path = path[:-1]
                        break
                    visited[nxt] = True; path.append(nxt); prev, cur = cur, nxt
            add_edge(a, end, [pt(*p) for p in path])
    return G


def official(g_gt, g_p, snap):
    if g_p.number_of_edges() == 0:
        return 0.0, 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        out = apls.make_graphs(g_gt.copy(), g_p.copy(), weight="length", linestring_delta=50, is_curved_eps=-1,
                               max_snap_dist=snap, allow_renaming=True)
        if out is None:
            return float("nan"), float("nan"), float("nan")
        (_, _, _, _, cp_gt, cp_p, gt_nat, p_nat, gt_pr, p_pr) = out
        c, c_gp, c_pg = apls.compute_apls_metric(gt_nat, p_nat, gt_pr, p_pr, cp_gt, cp_p, res_dir="",
                                                 min_path_length=0.001)
    return float(c), float(c_gp), float(c_pg)


def job(args):
    scene, run, (y, x), gsk, psk = args
    g_gt, g_p = skel_to_graph(gsk), skel_to_graph(psk)
    r = {"scene": scene, "run": run, "y": y, "x": x, "gt_nodes": g_gt.number_of_nodes(), "p_nodes": g_p.number_of_nodes()}
    for snap in SNAPS:
        try:
            c, c_gp, c_pg = official(g_gt, g_p, snap)
        except Exception as e:
            c = c_gp = c_pg = float("nan"); r[f"err_{snap}"] = repr(e)[:200]
        r[f"apls_{snap}"], r[f"gt2pr_{snap}"], r[f"pr2gt_{snap}"] = c, c_gp, c_pg
    return r


def crops(args):
    scene, run, wins = args
    meta = np.load(MOS / scene / "meta.npz"); H, W = (int(v) for v in meta["shape"])
    if run == "GT":
        gt, _ = load_gt(scene, H, W); sk = gt[2]["raster"]
    else:
        ign = ignore_mask(scene, H, W)
        m = unpack(np.load(MOS / scene / f"{run}.npz")["mask"], (H, W))
        sk = clean_skeleton(m & ~ign)
    return scene, run, {w: sk[w[0]:w[0] + 256, w[1]:w[1] + 256].copy() for w in wins}


def ours_scores(scene, run):
    p = pathlib.Path(f"data/rebuttal/vapls/{scene}__{run}.json")
    return {(w["y"], w["x"]): w["apls"] for w in json.loads(p.read_text())
            if w["tier"] == 2 and w["gt"] == "raster" and w["scale"] == 256 and w["buffer"] == 2}


def main(procs=32):
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {}
    for scene in TEST + EXT:
        scored = sorted(set.intersection(*[set(ours_scores(scene, r)) for r in RUNS]))
        pick = np.random.default_rng(2026).choice(len(scored), min(N_PER_SCENE, len(scored)), replace=False)
        plan[scene] = [scored[i] for i in sorted(pick)]
        print(f"{scene[:8]}: {len(scored)} scored windows, {len(plan[scene])} sampled", flush=True)
    t0 = time.time()
    with Pool(min(procs, 24)) as pool:
        got = pool.map(crops, [(sc, r, plan[sc]) for sc in plan for r in ["GT"] + RUNS], chunksize=1)
    C = {(sc, r): d for sc, r, d in got}
    print(f"skeleton crops ready [{time.time()-t0:.0f}s]", flush=True)
    tasks = [(sc, r, w, C[(sc, "GT")][w], C[(sc, r)][w]) for sc in plan for w in plan[sc] for r in RUNS]
    with Pool(procs) as pool:
        res = pool.map(job, tasks, chunksize=4)
    ours = {}
    for r in res:
        key = (r["scene"], r["run"])
        if key not in ours:
            ours[key] = ours_scores(*key)
        r["ours_2px"] = ours[key].get((r["y"], r["x"]))
    (OUT / "results.json").write_text(json.dumps(res, default=float))
    print(f"done: {len(res)} window-runs [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
