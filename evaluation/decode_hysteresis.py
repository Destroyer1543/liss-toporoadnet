import pathlib, sys
import numpy as np
from scipy.ndimage import label

MOS = pathlib.Path("data/rebuttal/mosaics")
PAIRS = {"H1": (0.05, 0.20), "H2": (0.10, 0.30), "H3": (0.20, 0.50)}
EIGHT = np.ones((3, 3), bool)


def hysteresis(p, low, high, support=None):
    cand = p >= round(low * 255)
    if support is not None:
        cand &= support
    lab, n = label(cand, structure=EIGHT)
    if n == 0:
        return cand
    seeds = np.unique(lab[(p >= round(high * 255)) & cand])
    keep = np.zeros(n + 1, bool); keep[seeds] = True; keep[0] = False
    return keep[lab]


def main(scene, runs):
    od = MOS / scene
    nodata = np.unpackbits(np.load(od / "meta.npz")["nodata"], axis=-1)
    for rn in runs:
        if (od / f"{rn}__P_skel.npz").exists():
            p = np.load(od / f"{rn}__P_skel.npz")["prob"]
            support = np.load(od / f"{rn}__P_mask.npz")["prob"] >= round(0.1 * 255)
        else:
            p = np.load(od / f"{rn}__P_out.npz")["prob"]; support = None
        nd = nodata[:, :p.shape[1]].astype(bool)
        for k, (lo, hi) in PAIRS.items():
            out = od / f"{rn}__{k}.npz"
            if out.exists():
                continue
            m = hysteresis(p, lo, hi, support) & ~nd
            np.savez_compressed(out, mask=np.packbits(m, axis=-1))
            print(f"{rn}__{k}: road-frac {m[~nd].mean():.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
