import json, os, pathlib, sys, time
from multiprocessing import Pool
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from official_apls_check import skel_to_graph, official, ignore_mask, RUNS, TEST, EXT, ours_scores
from vector_apls import clean_skeleton
from vector_apls_run import load_gt, unpack, MOS

V = pathlib.Path("data/rebuttal/vapls")
SNAP = 11.6
G = {}


def gt_skeleton(scene):
    meta = np.load(MOS / scene / "meta.npz"); H, W = (int(v) for v in meta["shape"])
    gt, _ = load_gt(scene, H, W)
    return scene, gt[2]["raster"]


def score(args):
    scene, run = args
    out = V / f"{scene}__{run}__official.json"
    if out.exists():
        return f"{scene[:8]} {run}: exists"
    t0 = time.time()
    meta = np.load(MOS / scene / "meta.npz"); H, W = (int(v) for v in meta["shape"])
    psk = clean_skeleton(unpack(np.load(MOS / scene / f"{run}.npz")["mask"], (H, W)) & ~ignore_mask(scene, H, W))
    gsk = G[scene]
    res = []
    for (y, x) in G["wins"][scene]:
        g, p = gsk[y:y + 256, x:x + 256], psk[y:y + 256, x:x + 256]
        try:
            c, c_gp, c_pg = official(skel_to_graph(g), skel_to_graph(p), SNAP)
        except Exception as e:
            c = c_gp = c_pg = float("nan")
        res.append({"tier": 2, "gt": "raster", "scale": 256, "buffer": 2, "y": y, "x": x,
                    "apls": c, "gt2pr": c_gp, "pr2gt": c_pg, "impl": "cosmiq_official", "snap_m": SNAP})
    out.write_text(json.dumps(res, default=float))
    return f"{scene[:8]} {run}: {len(res)} windows [{time.time()-t0:.0f}s]"


def main(procs=32):
    t0 = time.time()
    G["wins"] = {sc: sorted(set.intersection(*[set(ours_scores(sc, r)) for r in RUNS])) for sc in TEST + EXT}
    with Pool(6) as pool:
        for sc, sk in pool.imap_unordered(gt_skeleton, TEST + EXT):
            G[sc] = sk
    print(f"GT skeletons + window lists ready [{time.time()-t0:.0f}s]: " +
          ", ".join(f"{sc[:8]} {len(G['wins'][sc])}" for sc in TEST + EXT), flush=True)
    tasks = sorted([(sc, r) for sc in TEST + EXT for r in RUNS], key=lambda t: -len(G["wins"][t[0]]))
    with Pool(procs) as pool:
        for msg in pool.imap_unordered(score, tasks):
            print(msg, flush=True)
    print(f"ALL DONE [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
