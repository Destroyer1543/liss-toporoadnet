import json, pathlib, re, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from graph_metrics import apls_raster, junction_recall
from metrics import cldice_metric

import os
PRED = pathlib.Path(os.environ.get("REB_PRED", "data/rebuttal/preds_test"))
OUT = pathlib.Path(os.environ.get("REB_OUT", "data/rebuttal/metrics_test"))
TILEDIR = pathlib.Path("data/liss4_tiles/test")
EMASK = pathlib.Path("data/eval_masks")
G = {}

def parse(name):
    scene, tile = name[:-4].split("__")
    return scene, int(re.search(r"y(\d+)", tile).group(1)), int(re.search(r"x(\d+)", tile).group(1))

CACHE = pathlib.Path("data/rebuttal/gt_test_cache.npz")

def load_gt(names):
    if CACHE.exists():
        d = np.load(CACHE)
        un = lambda a: np.unpackbits(a, axis=-1)
        return un(d["gt"]), un(d["mj"]), un(d["ig"]), d["ok"]
    em = {}
    gt = np.zeros((len(names), 256, 256), np.uint8)
    mj = np.zeros_like(gt); ig = np.zeros_like(gt); ok = np.ones(len(names), bool)
    for i, n in enumerate(names):
        s, y, x = parse(n)
        if s not in em:
            d = np.load(EMASK / f"{s}.npz"); em[s] = (d["major"], d["ignore"])
        gt[i] = np.load(TILEDIR / n)["mask"]
        a, b = em[s][0][y:y+256, x:x+256], em[s][1][y:y+256, x:x+256]
        if a.shape != (256, 256):
            ok[i] = False; continue
        mj[i], ig[i] = a, b
    pk = lambda a: np.packbits(a > 0, axis=-1)
    np.savez(CACHE, gt=pk(gt), mj=pk(mj), ig=pk(ig), ok=ok)
    return gt, mj, ig, ok

def conf(p, t, ignore):
    keep = ignore < 1
    p = p & keep; t = (t > 0) & keep
    return float((p & t).sum()), float((p & ~t).sum()), float((~p & t).sum())

def one(i):
    p = G["pred"][i]; gt, mj, ig = G["gt"][i], G["mj"][i], G["ig"][i]
    z = np.zeros_like(ig)
    r = {}
    for tier, g, ign in ((1, gt, z), (2, mj, ig)):
        tp, fp, fn = conf(p, g, ign)
        keep = (ign < 1)
        pk = (p & keep).astype(np.float32); gk = ((g > 0) & keep).astype(np.float32)
        r[f"t{tier}_tp"], r[f"t{tier}_fp"], r[f"t{tier}_fn"] = tp, fp, fn
        r[f"t{tier}_cldice"] = cldice_metric(pk[None, None], gk[None, None], 0.5)
        r[f"t{tier}_apls"] = apls_raster(pk, gk, thr=0.5)
        r[f"t{tier}_jrec"] = junction_recall(pk, gk, thr=0.5)
    return r

def run(rn, procs=24):
    out = OUT / f"{rn}.npz"
    if out.exists():
        print(f"{rn}: exists, skip", flush=True); return
    G["pred"] = np.unpackbits(np.load(PRED / f"{rn}.npy"), axis=-1).astype(bool)
    with Pool(procs) as pool:
        rows = pool.map(one, range(len(G["pred"])), chunksize=32)
    keys = rows[0].keys()
    np.savez(out, ok=G["ok"], **{k: np.array([r[k] for r in rows], np.float64) for k in keys})
    print(f"{rn}: done", flush=True)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = json.loads((pathlib.Path("data/rebuttal/preds_test") / "tiles.json").read_text())
    G["gt"], G["mj"], G["ig"], G["ok"] = load_gt(names)
    print(f"GT loaded: {len(names)} tiles, {int((~G['ok']).sum())} edge tiles without tier-2 crop", flush=True)
    for rn in sys.argv[1].split(","):
        run(rn)
