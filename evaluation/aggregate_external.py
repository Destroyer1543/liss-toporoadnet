import pathlib, sys
import numpy as np

M = pathlib.Path("data/rebuttal/metrics_ext")
GROUPS = {"Ours": ["exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2"],
          "U-Net": ["exp16_unet_full", "exp23_unet_seed1", "exp25_unet_seed2"],
          "D-LinkNet": ["exp17_dlinknet_full", "exp36_dlinknet_seed1", "exp37_dlinknet_seed2"],
          "RoadFormer": ["exp26_roadformer_full"], "U-Net+clDice": ["exp47_unet_cldice_seed0"]}
HWY = {"motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link", "secondary_link"}
OTHER = {"tertiary", "unclassified", "residential"}


def stats(ds):
    cat = lambda k: np.concatenate([d[k] for d in ds])
    o = {}
    for t in (1, 2):
        tp, fp, fn = cat(f"t{t}_tp"), cat(f"t{t}_fp"), cat(f"t{t}_fn")
        o[f"t{t}_iou"] = (tp / (tp + fp + fn + 1e-6)).mean()
        o[f"t{t}_piou"] = tp.sum() / (tp.sum() + fp.sum() + fn.sum())
        o[f"t{t}_cld"] = np.nanmean(cat(f"t{t}_cldice")); o[f"t{t}_apls"] = np.nanmean(cat(f"t{t}_apls"))
    agg = {}
    for d in ds:
        for k in d.files:
            if k.startswith("cls_") or k.startswith("surf_"):
                agg[k] = agg.get(k, 0) + int(d[k])
    def rec(names, prefix="cls_"):
        n = sum(agg.get(f"{prefix}{c}_n", 0) for c in names); h = sum(agg.get(f"{prefix}{c}_hit", 0) for c in names)
        return h / n if n else np.nan
    allc = {k[4:-2] for k in agg if k.startswith("cls_") and k.endswith("_n")}
    o["rec_hwy"], o["rec_other"], o["rec_minor"] = rec(HWY), rec(OTHER), rec(allc - HWY - OTHER)
    o["rec_paved"], o["rec_unpaved"] = rec(["paved"], "surf_"), rec(["unpaved"], "surf_")
    o["n_tiles"] = len(cat("t1_tp"))
    return o


def main(scenes):
    keys = ["t1_iou", "t1_piou", "t1_apls", "t2_iou", "t2_piou", "t2_cld", "t2_apls", "rec_hwy", "rec_other",
            "rec_minor", "rec_paved", "rec_unpaved"]
    for scope in scenes + ["POOLED"]:
        sc = scenes if scope == "POOLED" else [scope]
        print(f"\n=== {scope[:12]} ===")
        res = {}
        for g, runs in GROUPS.items():
            vals = []
            for r in runs:
                ds = [dict(np.load(M / f"{s}__{r}.npz")) for s in sc if (M / f"{s}__{r}.npz").exists()]
                if len(ds) == len(sc):
                    vals.append(stats([np.load(M / f"{s}__{r}.npz") for s in sc]))
            if vals:
                res[g] = vals
        if not res:
            continue
        print("n_tiles", {g: v[0]["n_tiles"] for g, v in res.items()})
        for k in keys:
            line = f"{k:11s} "
            for g, v in res.items():
                a = np.array([x[k] for x in v], float)
                line += f"{g} {a.mean():.4f}" + (f"±{a.std(ddof=1):.4f}" if len(a) > 1 else "") + "  "
            if "Ours" in res and "U-Net" in res:
                o = np.mean([x[k] for x in res["Ours"]]); u = np.mean([x[k] for x in res["U-Net"]])
                line += f"| rel {100 * (o / u - 1):+.1f}%"
            print(line)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
