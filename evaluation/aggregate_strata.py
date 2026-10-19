import json, pathlib, sys
import numpy as np
S = pathlib.Path("data/rebuttal/strata_test")
M = pathlib.Path("data/rebuttal/metrics_test")
GROUPS = {"Ours": ["exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2"],
          "U-Net": ["exp16_unet_full", "exp23_unet_seed1", "exp25_unet_seed2"],
          "D-LinkNet": ["exp17_dlinknet_full"], "RoadFormer": ["exp26_roadformer_full"]}
def ms(v):
    v = np.array(v, float); return f"{v.mean():.4f}" + (f"±{v.std(ddof=1):.4f}" if len(v) > 1 else "")
rows = {}
for g, runs in GROUPS.items():
    runs = [r for r in runs if (S / f"{r}.npz").exists()]
    if not runs: continue
    acc = {k: [] for k in ["rec_hw", "rec_other", "rec_minor", "strict_apls", "strict_iou", "strict_cld"] +
           [f"apls_t{t}" for t in range(3)] + [f"rec_t{t}" for t in range(3)]}
    for r in runs:
        d = np.load(S / f"{r}.npz"); m = np.load(M / f"{r}.npz")
        for k in ("rec_hw", "rec_other", "rec_minor"): acc[k].append(float(d[k]))
        acc["strict_apls"].append(np.nanmean(d["strict_apls"]))
        tp, fp, fn = d["strict_tp"], d["strict_fp"], d["strict_fn"]
        acc["strict_iou"].append((tp / (tp + fp + fn + 1e-6)).mean()); acc["strict_cld"].append(np.nanmean(d["strict_cldice"]))
        for t in range(3):
            sel = d["terc"] == t
            acc[f"apls_t{t}"].append(np.nanmean(m["t2_apls"][sel]))
            tp, fn = m["t1_tp"][sel], m["t1_fn"][sel]
            acc[f"rec_t{t}"].append(tp.sum() / (tp.sum() + fn.sum()))
    rows[g] = acc
    q = np.load(S / f"{runs[0]}.npz")["q"]
print("density tercile edges (OSM road px fraction):", q)
for k in ["rec_hw", "rec_other", "rec_minor", "strict_iou", "strict_cld", "strict_apls", "apls_t0", "apls_t1", "apls_t2", "rec_t0", "rec_t1", "rec_t2"]:
    print(f"{k:12s} " + "  ".join(f"{g} {ms(a[k])}" for g, a in rows.items()))
