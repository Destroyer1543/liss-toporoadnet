import json, pathlib, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vector_apls import clean_skeleton, apls_window

MOS = pathlib.Path("data/rebuttal/mosaics")
EXT = pathlib.Path("data/rebuttal/external")
EMASK = pathlib.Path("data/eval_masks")
OUT = pathlib.Path("data/rebuttal/vapls")
import os
SCALES = [int(v) for v in os.environ.get("REB_SCALES", "1024,256,4096,0").split(",")]
BUFFERS_SENS = [1, 3, 5]
SMAX = {256: 300, 1024: 300, 4096: 100, 0: 50}
TIERS = [int(v) for v in os.environ.get("REB_TIERS", "1,2").split(",")]
SUB256 = float(os.environ.get("REB_SUB256", "1.0"))
GTONLY = os.environ.get("REB_GTONLY")
SUFFIX = os.environ.get("REB_SUFFIX", "")
G = {}


def unpack(a, shape):
    return np.unpackbits(a, axis=-1)[:, :shape[1]].astype(bool)


def load_gt(scene, H, W):
    if (EXT / f"{scene}.npz").exists():
        d = np.load(EXT / f"{scene}.npz")
        sh = tuple(d["shape"])
        t1, major, ign = (unpack(d[k], sh)[:H, :W] for k in ("t1", "major", "ignore"))
        t1v, majv = (unpack(d[k], sh)[:H, :W] for k in ("t1_cl", "major_cl"))
        vec = {1: clean_skeleton(t1v), 2: clean_skeleton(majv)}
    else:
        d = np.load(EMASK / f"{scene}.npz")
        major, ign = d["major"][:H, :W].astype(bool), d["ignore"][:H, :W].astype(bool)
        t1 = major | ign
        vec = {1: None, 2: None}
        tv = pathlib.Path("data/rebuttal/testvec") / f"{scene}.npz"
        if os.environ.get("REB_TESTVEC") == "1" and tv.exists():
            dv = np.load(tv); shv = tuple(dv["shape"])
            vec = {1: clean_skeleton(unpack(dv["t1_cl"], shv)[:H, :W]),
                   2: clean_skeleton(unpack(dv["major_cl"], shv)[:H, :W])}
    ras = {1: clean_skeleton(t1), 2: clean_skeleton(major)}
    return {t: {"raster": ras[t], "vector": vec[t]} for t in (1, 2)}, ign


def windows(H, W, s):
    if s == 0:
        return [(0, 0, H, W)]
    return [(y, x, min(y + s, H), min(x + s, W)) for y in range(0, H, s) for x in range(0, W, s)]


def score(run):
    t0 = time.time()
    H, W = G["shape"]
    m = unpack(np.load(MOS / G["scene"] / f"{run}.npz")["mask"], (H, W))
    preds = {1: clean_skeleton(m), 2: clean_skeleton(m & ~G["ignore"])}
    res = []
    for s in SCALES:
        for tier in TIERS:
            for src in ("raster", "vector"):
                if GTONLY and src != GTONLY:
                    continue
                gsk = G["gt"][tier][src]
                if gsk is None:
                    continue
                bufs = tuple([2] + (BUFFERS_SENS if s == 1024 else []))
                wins = windows(H, W, s)
                if s == 256 and SUB256 < 1.0:
                    keep = np.random.default_rng(123).random(len(wins)) < SUB256
                    wins = [w for w, k in zip(wins, keep) if k]
                for (y0, x0, y1, x1) in wins:
                    g = gsk[y0:y1, x0:x1]
                    if g.sum() < 2:
                        continue
                    p = preds[tier][y0:y1, x0:x1]
                    rb = apls_window(g, p, buffers=bufs, rng=np.random.default_rng(0), smax=SMAX[s])
                    if rb is None:
                        continue
                    for b, r in rb.items():
                        r.update(tier=tier, gt=src, scale=s, buffer=b, y=y0, x=x0)
                        res.append(r)
                (OUT / f"{G['scene']}__{run}{SUFFIX}.partial.json").write_text(json.dumps(res, default=float))
    (OUT / f"{G['scene']}__{run}{SUFFIX}.json").write_text(json.dumps(res, default=float))
    return f"{run}: {len(res)} window-scores [{time.time()-t0:.0f}s]"


def main(scene, runs, procs):
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [r for r in runs if not (OUT / f"{scene}__{r}{SUFFIX}.json").exists()]
    if not runs:
        print(scene, "all done"); return
    meta = np.load(MOS / scene / "meta.npz")
    H, W = (int(v) for v in meta["shape"])
    t0 = time.time()
    G["scene"], G["shape"] = scene, (H, W)
    G["gt"], G["ignore"] = load_gt(scene, H, W)
    print(f"{scene}: GT skeletons ready [{time.time()-t0:.0f}s]", flush=True)
    with Pool(min(procs, len(runs))) as pool:
        for msg in pool.imap_unordered(score, runs):
            print("  ", msg, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]) if len(sys.argv) > 3 else 8)
