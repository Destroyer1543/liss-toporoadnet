import json, pathlib, sys
import numpy as np
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vector_apls import clean_skeleton
from vector_apls_run import load_gt, unpack, MOS

VAL = "R2F08FEB2024066453010100064SSANSTUC00GTDA"
OUT = pathlib.Path("data/rebuttal/loc_metrics")


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (OUT / f"{n}.json").exists()]
    if not todo:
        return
    meta = np.load(MOS / VAL / "meta.npz"); H, W = (int(v) for v in meta["shape"])
    gt, ign = load_gt(VAL, H, W)
    ys, xs = slice(4000, 12192), slice(4000, 12192)
    g = gt[2]["raster"][ys, xs]; ig = ign[ys, xs]
    dt_g = distance_transform_edt(~g)
    for n in todo:
        m = unpack(np.load(MOS / VAL / f"{n}.npz")["mask"], (H, W))[ys, xs] & ~ig
        sk = clean_skeleton(m)
        r = {"far10_norm": float((dt_g[sk] > 10).sum() / max(g.sum(), 1)),
             "width_px": float(m.sum() / max(sk.sum(), 1)),
             "osm_within2": float((distance_transform_edt(~sk)[g] <= 2).mean())}
        (OUT / f"{n}.json").write_text(json.dumps(r))
        print(n, r, flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
