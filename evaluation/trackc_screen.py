import csv, json, pathlib, sys
import numpy as np
from btrack_screen import score, VAL, V

CONF = {"published": ["exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2"],
        "no_spectral": ["exp18_abl_nospectral", "exp40_abl_nospectral_seed1", "exp44_abl_nospectral_seed2"],
        "no_recon": ["exp19_abl_notopo", "exp20_abl_norecon", "exp38_abl_notopo_seed1", "exp39_abl_norecon_seed1",
                     "exp43_abl_norecon_seed2"],
        "no_deepsup": ["exp21_abl_nodeepsup", "exp45_abl_nodeepsup_seed1", "exp46_abl_nodeepsup_seed2"],
        "antiblob": ["exp29_toporoad_antiblob"],
        "drop_topo": ["exp50_abl_droptopo_seed0", "exp51_abl_droptopo_seed1"]}
UNET_VAL = 0.0347


def val_iou(run):
    rows = [r for r in csv.reader(open(f"runs/{run}/log.csv")) if r and r[0] != "epoch"]
    best = max(rows, key=lambda r: float(r[5]))
    return float(best[3])


def summary():
    out = {}
    for c, runs in CONF.items():
        have = [r for r in runs if (V / f"{VAL}__{r}__F05.json").exists()]
        if not have:
            continue
        vals = {k: [score(r, "F05", "", sc, k2) for r in have] for k, (sc, k2) in
                {"apls15": (256, "apls"), "gt2pr15": (256, "gt2pr"), "pr2gt15": (256, "pr2gt"), "apls6": (1024, "apls")}.items()}
        out[c] = {k: float(np.mean(v)) for k, v in vals.items()}
        out[c]["n"] = len(have); out[c]["runs"] = have
        out[c]["val_iou"] = float(np.mean([val_iou(r) for r in have]))
    return out


if __name__ == "__main__":
    need = sys.argv[1].split(",") if len(sys.argv) > 1 else []
    miss = [r for r in need if not (V / f"{VAL}__{r}__F05.json").exists()]
    if miss:
        print("INCOMPLETE — not showing any score; missing:", miss); sys.exit(0)
    s = summary(); p = s["published"]
    print(f"{'config':12s} n  APLS@1.5km  APLS@6km  GT→pred  pred→GT  valIoU  promote?")
    best, bestv = None, -1
    for c, r in s.items():
        ok = (c != "published" and r["apls15"] >= p["apls15"] + 0.002 and r["gt2pr15"] >= p["gt2pr15"]
              and r["apls6"] >= p["apls6"] and r["val_iou"] >= p["val_iou"] - 0.005)
        print(f"{c:12s} {r['n']}  {r['apls15']:.4f}      {r['apls6']:.4f}    {r['gt2pr15']:.4f}   {r['pr2gt15']:.4f}   "
              f"{r['val_iou']:.4f}  {'YES' if ok else ('ref' if c == 'published' else 'no')}")
        if ok and r["apls15"] > bestv:
            best, bestv = c, r["apls15"]
    print("PROMOTED:", best, f"(must also beat U-Net val {UNET_VAL} with 3 seeds before any test number)")
    (V.parent / "trackc_c0_screen.json").write_text(json.dumps({"summary": s, "promoted": best}, indent=1))
