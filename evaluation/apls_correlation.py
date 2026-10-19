import json, pathlib, re, sys
import numpy as np
from scipy.stats import spearmanr

V = pathlib.Path("data/rebuttal/vapls")
M = pathlib.Path("data/rebuttal/metrics_test32")
NAMES = json.loads(pathlib.Path("data/rebuttal/preds_test/tiles.json").read_text())
RUNS = ["exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2", "exp16_unet_full",
        "exp23_unet_seed1", "exp25_unet_seed2", "exp17_dlinknet_full", "exp26_roadformer_full"]
GROUP = {"exp15": "Ours", "exp22": "Ours", "exp24": "Ours", "exp16": "U-Net", "exp23": "U-Net", "exp25": "U-Net",
         "exp17": "D-LinkNet", "exp26": "RoadFormer"}


def tile_index():
    idx = {}
    for i, n in enumerate(NAMES):
        sc, t = n[:-4].split("__")
        y, x = int(re.search(r"y(\d+)", t).group(1)), int(re.search(r"x(\d+)", t).group(1))
        idx[(sc, y, x)] = i
    return idx


def main(scenes, tier=2):
    idx = tile_index()
    rows = []
    for rn in RUNS:
        m = np.load(M / f"{rn}.npz")
        for sc in scenes:
            for suf in (".json", ".partial.json"):
                p = V / f"{sc}__{rn}{suf}"
                if p.exists():
                    break
            else:
                continue
            for w in json.loads(p.read_text()):
                if w["scale"] != 256 or w["tier"] != tier or w["gt"] != "raster" or w["buffer"] != 2:
                    continue
                i = idx.get((sc, w["y"], w["x"]))
                if i is None:
                    continue
                a = m[f"t{tier}_apls"][i]
                if np.isfinite(a):
                    rows.append((rn, i, a, w["apls"], w["gt2pr"]))
    if not rows:
        print("no matched tiles yet"); return
    runs = sorted({r[0] for r in rows})
    common = set.intersection(*[{r[1] for r in rows if r[0] == rn} for rn in runs])
    rows = [r for r in rows if r[1] in common]
    a = np.array([r[2] for r in rows]); v = np.array([r[3] for r in rows]); g = np.array([r[4] for r in rows])
    print(f"tier {tier}: {len(common)} common tiles x {len(runs)} runs")
    print(f"pooled Spearman: APLS_r vs vector APLS {spearmanr(a, v).statistic:.3f}; "
          f"vs vector GT->pred {spearmanr(a, g).statistic:.3f}")
    means = {}
    for rn in runs:
        sel = [r for r in rows if r[0] == rn]
        aa = np.array([r[2] for r in sel]); vv = np.array([r[3] for r in sel]); gg = np.array([r[4] for r in sel])
        means[rn] = (aa.mean(), vv.mean(), gg.mean())
        print(f"  {rn:24s} APLS_r {aa.mean():.4f}  vecAPLS {vv.mean():.4f}  vecGT2PR {gg.mean():.4f}  "
              f"rho {spearmanr(aa, vv).statistic:.3f}")
    grp = {}
    for rn, mv in means.items():
        grp.setdefault(GROUP[rn[:5]], []).append(mv)
    print("model means (seed-averaged):")
    order = {}
    for k, j in (("APLS_r", 0), ("vecAPLS", 1), ("vecGT2PR", 2)):
        vals = {gname: np.mean([x[j] for x in xs]) for gname, xs in grp.items()}
        order[k] = sorted(vals, key=lambda n: -vals[n])
        print(f"  {k:9s} " + "  ".join(f"{n} {vals[n]:.4f}" for n in order[k]))
    print("same ranking (APLS_r vs vecAPLS):", order["APLS_r"] == order["vecAPLS"])


if __name__ == "__main__":
    main(sys.argv[1].split(","), int(sys.argv[2]) if len(sys.argv) > 2 else 2)
