import json, pathlib, re, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import tile_metrics as TM

PRED = pathlib.Path("data/rebuttal/preds_test")
HW = pathlib.Path("data/highway_masks")
OUT = pathlib.Path("data/rebuttal/strata_test")


def main(runs):
    OUT.mkdir(parents=True, exist_ok=True)
    names = json.loads((PRED / "tiles.json").read_text())
    gt, mj, ig, ok = TM.load_gt(names)
    hw = {}
    hmj = np.zeros_like(mj); hig = np.zeros_like(ig)
    for i, n in enumerate(names):
        s, y, x = TM.parse(n)
        if s not in hw:
            d = np.load(HW / f"{s}.npz"); hw[s] = (d["major"], d["ignore"])
        hmj[i] = hw[s][0][y:y + 256, x:x + 256]; hig[i] = hw[s][1][y:y + 256, x:x + 256]
    dens = gt.reshape(len(gt), -1).mean(1)
    pos = dens > 0
    terc = np.full(len(gt), -1)
    q = np.quantile(dens[pos], [1 / 3, 2 / 3])
    terc[pos] = np.digitize(dens[pos], q)
    cls_hw = hmj > 0
    cls_other = (mj > 0) & ~cls_hw
    cls_minor = (gt > 0) & ~(mj > 0)
    for rn in runs:
        out = OUT / f"{rn}.npz"
        if out.exists():
            continue
        P = np.unpackbits(np.load(PRED / f"{rn}.npy"), axis=-1).astype(bool)
        res = {"dens": dens, "terc": terc, "q": q}
        for k, c in (("hw", cls_hw), ("other", cls_other), ("minor", cls_minor)):
            res[f"rec_{k}"] = (P & c).sum() / max(c.sum(), 1)
            res[f"n_{k}"] = c.sum()
        TM.G.update(pred=P, gt=gt, mj=hmj, ig=hig)
        with Pool(24) as pool:
            rows = pool.map(TM.one, range(len(P)), chunksize=32)
        for k in ("t2_tp", "t2_fp", "t2_fn", "t2_cldice", "t2_apls"):
            res["strict_" + k[3:]] = np.array([r[k] for r in rows], np.float64)
        np.savez(out, **res)
        print(f"{rn}: done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
