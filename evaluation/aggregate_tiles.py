import json, pathlib, sys
import numpy as np

import os
M = pathlib.Path(os.environ.get("REB_OUT", "data/rebuttal/metrics_test"))
NAMES = json.loads(pathlib.Path("data/rebuttal/preds_test/tiles.json").read_text())
SCENE = np.array([n.split("__")[0][:12] for n in NAMES])
PAPER = json.loads(pathlib.Path("runs/test_twotier_results.json").read_text())


def tier_stats(d, t, sel):
    tp, fp, fn = d[f"t{t}_tp"][sel], d[f"t{t}_fp"][sel], d[f"t{t}_fn"][sel]
    iou_tile = (tp / (tp + fp + fn + 1e-6)).mean()
    iou_pool = tp.sum() / (tp.sum() + fp.sum() + fn.sum())
    return {"iou": iou_tile, "iou_pool": iou_pool,
            "cldice": np.nanmean(d[f"t{t}_cldice"][sel]), "apls": np.nanmean(d[f"t{t}_apls"][sel]),
            "jrec": np.nanmean(d[f"t{t}_jrec"][sel]), "n_apls": int(np.isfinite(d[f"t{t}_apls"][sel]).sum())}


def load(rn):
    return dict(np.load(M / f"{rn}.npz"))


def main(runs):
    all_sel = np.ones(len(NAMES), bool)
    print("=== 1. REPRODUCTION (paper subset = first N sorted tiles) ===")
    for rn in runs:
        if not (M / f"{rn}.npz").exists():
            continue
        d = load(rn); p = PAPER.get(rn)
        for N in (250, 300):
            sub = np.zeros(len(NAMES), bool); sub[:N] = True
            s2 = tier_stats(d, 2, sub)
            full2 = tier_stats(d, 2, all_sel)
            tag = ""
            if p:
                tag = (f" | paper T2 IoU {p['t2_iou']:.4f} clD {p['t2_cldice']:.4f} APLS {p['t2_apls']:.4f}")
            print(f"{rn:26s} N={N}: T2 IoU(all) {full2['iou']:.4f} clD {s2['cldice']:.4f} "
                  f"APLS_r {s2['apls']:.4f} (n={s2['n_apls']}){tag}")
    print("\n=== 2. CORRECTED: all 6,076 tiles, both scenes; and per scene ===")
    hdr = f"{'run':26s} {'scope':10s} {'T1 IoU':>7s} {'T1 clD':>7s} {'T1 APLSr':>8s} {'T2 IoU':>7s} {'T2 pIoU':>7s} {'T2 clD':>7s} {'T2 APLSr':>8s} {'T2 JR':>6s}"
    print(hdr)
    out = {}
    for rn in runs:
        if not (M / f"{rn}.npz").exists():
            continue
        d = load(rn); out[rn] = {}
        for scope, sel in (("ALL", all_sel), ("R2F24AUG", SCENE == "R2F24AUG2023"), ("RAF23OCT", SCENE == "RAF23OCT2023")):
            a, b = tier_stats(d, 1, sel), tier_stats(d, 2, sel)
            out[rn][scope] = {"t1": a, "t2": b}
            print(f"{rn:26s} {scope:10s} {a['iou']:7.4f} {a['cldice']:7.4f} {a['apls']:8.4f} "
                  f"{b['iou']:7.4f} {b['iou_pool']:7.4f} {b['cldice']:7.4f} {b['apls']:8.4f} {b['jrec']:6.4f}")
    pathlib.Path(str(M) + "_summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].split(","))
