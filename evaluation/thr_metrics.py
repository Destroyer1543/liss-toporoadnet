import json, pathlib, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent)); sys.path.insert(0, "src")
import tile_metrics as TM
from graph_metrics import apls_raster

THR_DIR = pathlib.Path("data/rebuttal/preds_test_thr")
P32 = pathlib.Path("data/rebuttal/preds_test32")
G = {}


def one(i):
    p = G["pred"][i]; mj, ig = G["mj"][i], G["ig"][i]
    keep = ig < 1
    pk = p & keep; t = (mj > 0) & keep
    tp, fp, fn = float((pk & t).sum()), float((pk & ~t).sum()), float((~pk & t).sum())
    return tp, fp, fn, apls_raster(pk.astype(np.float32), t.astype(np.float32), thr=0.5), float(pk.sum())


def main(runs):
    names = json.loads(pathlib.Path("data/rebuttal/preds_test/tiles.json").read_text())
    _, G["mj"], G["ig"], _ = TM.load_gt(names)
    res = {}
    for rn in runs:
        res[rn] = {}
        for t in (0.3, 0.4, 0.5, 0.6, 0.7):
            f = (P32 / f"{rn}.npy") if t == 0.5 else (THR_DIR / f"{rn}__t{t}.npy")
            G["pred"] = np.unpackbits(np.load(f), axis=-1).astype(bool)
            with Pool(16) as pool:
                rows = np.array(pool.map(one, range(len(G["pred"])), chunksize=32), dtype=np.float64)
            tp, fp, fn, ap, npx = rows.T
            res[rn][str(t)] = {"road_frac": float(npx.sum() / (len(G["pred"]) * 65536)),
                               "iou_pool": float(tp.sum() / (tp.sum() + fp.sum() + fn.sum())),
                               "apls_r": float(np.nanmean(ap))}
            print(rn, t, res[rn][str(t)], flush=True)
    pathlib.Path("data/rebuttal/thr_metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].split(","))
