import json, pathlib, sys
import numpy as np
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vector_apls import clean_skeleton
from vector_apls_run import load_gt, unpack, MOS

VAL = "R2F08FEB2024066453010100064SSANSTUC00GTDA"
RUNS = {"ours": [f"{r}__F05" for r in ("exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2")],
        "unet": [f"{r}__T05" for r in ("exp16_unet_full", "exp23_unet_seed1", "exp25_unet_seed2")],
        "dlinknet": ["exp17_dlinknet_full__T05"], "roadformer": ["exp26_roadformer_full__T05"],
        "ours_S03": [f"{r}__S03" for r in ("exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2")]}
BINS = [0, 1.01, 1.5, 2.01, 2.9, 5, 10, 1e9]
LAB = ["0-1", "1-1.5", "1.5-2", "2-2.9", "2.9-5", "5-10", ">10"]
meta = np.load(MOS / VAL / "meta.npz"); H, W = (int(v) for v in meta["shape"])
gt, ign = load_gt(VAL, H, W)
ys, xs = slice(4000, 12192), slice(4000, 12192)
g = gt[2]["raster"][ys, xs]; ig = ign[ys, xs]
dt_g = distance_transform_edt(~g)
res = {}
for fam, runs in RUNS.items():
    acc = []
    for r in runs:
        m = unpack(np.load(MOS / VAL / f"{r}.npz")["mask"], (H, W))[ys, xs] & ~ig
        sk = clean_skeleton(m)
        dp = dt_g[sk]; dg = distance_transform_edt(~sk)[g]
        acc.append({"pred_hist": np.histogram(dp, BINS)[0] / max(len(dp), 1),
                    "gt_hist": np.histogram(dg, BINS)[0] / max(len(dg), 1),
                    "pred_len": int(sk.sum()), "width": float(m.sum() / max(sk.sum(), 1))})
    ph = np.mean([a["pred_hist"] for a in acc], 0); gh = np.mean([a["gt_hist"] for a in acc], 0)
    res[fam] = {"pred_px_by_dist_to_gt": dict(zip(LAB, ph.round(3).tolist())),
                "gt_px_by_dist_to_pred": dict(zip(LAB, gh.round(3).tolist())),
                "pred_len_over_gt_len": round(np.mean([a["pred_len"] for a in acc]) / g.sum(), 3),
                "mask_width_px": round(np.mean([a["width"] for a in acc]), 2)}
    print(fam, json.dumps(res[fam]), flush=True)
pathlib.Path("data/rebuttal/loc_diag_val.json").write_text(json.dumps(res, indent=1))
