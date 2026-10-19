import json, pathlib, sys
import numpy as np
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import aggregate_families as A

R = pathlib.Path("data/rebuttal")
T32 = R / "metrics_test32"
TEST = ["R2F24AUG2023064073010100066SSANSTUC00GTDD", "RAF23OCT2023035685010100064SSANSTUC00GTDA"]
EXT = ["R2F13DEC2025076029011100053SSANSTUC00GTDA", "RAF20DEC2025046888011000052SSANSTUC00GTDD",
       "R2F25JAN2026076640011000053SSANSTUC00GTDD", "RAF18JAN2026047300011100053SSANSTUC00GTDC"]
CONF = {"full": ["exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2"],
        "unet": ["exp16_unet_full", "exp23_unet_seed1", "exp25_unet_seed2"],
        "dlinknet": ["exp17_dlinknet_full", "exp36_dlinknet_seed1", "exp37_dlinknet_seed2"],
        "unet_cldice": ["exp47_unet_cldice_seed0", "exp48_unet_cldice_seed1", "exp49_unet_cldice_seed2"],
        "unet_cldice_ds": ["exp65_unet_cldice_ds_seed0", "exp66_unet_cldice_ds_seed1", "exp67_unet_cldice_ds_seed2"],
        "-topology heads (true)": ["exp50_abl_droptopo_seed0", "exp51_abl_droptopo_seed1", "exp64_abl_droptopo_seed2"],
        "-reconstruction": ["exp19_abl_notopo", "exp20_abl_norecon", "exp38_abl_notopo_seed1",
                            "exp39_abl_norecon_seed1", "exp43_abl_norecon_seed2"],
        "-spectral stem": ["exp18_abl_nospectral", "exp40_abl_nospectral_seed1", "exp44_abl_nospectral_seed2"],
        "-deep supervision": ["exp21_abl_nodeepsup", "exp45_abl_nodeepsup_seed1", "exp46_abl_nodeepsup_seed2"]}


def tile_run(run, scene=None):
    d = np.load(T32 / f"{run}.npz")
    ok = d["ok"].astype(bool) if "ok" in d.files else np.ones(len(d["t2_apls"]), bool)
    if scene is not None:
        names = json.loads((R / "preds_test" / "tiles.json").read_text())
        ok &= np.array([scene[:12] in n for n in names])
    tp, fp, fn = d["t2_tp"][ok], d["t2_fp"][ok], d["t2_fn"][ok]
    return {"aplsr": float(np.nanmean(d["t2_apls"][ok])), "iou_pooled": float(tp.sum() / (tp.sum() + fp.sum() + fn.sum())),
            "iou_tile": float((tp / (tp + fp + fn + 1e-6)).mean())}


def section(t):
    print(f"\n=== {t}")


if __name__ == "__main__":
    section("1. Corrections table — all 6,076 test tiles, Tier-2 APLS_r (mean ± sd over runs; Welch p vs full)")
    full = [tile_run(r)["aplsr"] for r in CONF["full"]]
    for c, runs in CONF.items():
        v = [tile_run(r)["aplsr"] for r in runs]
        p = stats.ttest_ind(full, v, equal_var=False).pvalue if c != "full" else float("nan")
        io = np.mean([tile_run(r)["iou_pooled"] for r in runs])
        print(f"{c:24s} n={len(v)} APLS_r {np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}  Δ {np.mean(v)-np.mean(full):+.4f}  "
              f"p {p:.3f}  pooled IoU {io:.4f}")
    for sc in TEST:
        o = np.mean([tile_run(r, sc)["aplsr"] for r in CONF["full"]]); u = np.mean([tile_run(r, sc)["aplsr"] for r in CONF["unet"]])
        print(f"per scene {sc[:8]}: ours {o:.4f} U-Net {u:.4f} → {100*(o/u-1):+.1f}%")
    print(f"U-Net vs U-Net+clDice Welch p: {stats.ttest_ind([tile_run(r)['aplsr'] for r in CONF['unet']], [tile_run(r)['aplsr'] for r in CONF['unet_cldice']], equal_var=False).pvalue:.3f}")
    base = tile_run("exp15_toporoad_full")["aplsr"]
    for r, b in (("exp33_band_noNIR", "NIR"), ("exp35_band_noG", "G"), ("exp34_band_noR", "R")):
        print(f"band zeroed at test, {b}: APLS_r {tile_run(r)['aplsr']:.4f} ({100*(tile_run(r)['aplsr']/base-1):+.1f}% vs seed-0 full {base:.4f})")

    section("2. Strict APLS, test scenes pooled, Tier 2, raster GT, 2 px (Δ = ours − family, 95% block-bootstrap CI)")
    for sc_px in (256, 1024):
        A.fmt(A.compare(TEST, 2, "raster", sc_px, "apls"), f"T2 {sc_px}px apls")

    section("3. External scenes: strict APLS per scene, scene average, pooled")
    for sc_px in (256, 1024):
        per = {}
        for sc in EXT:
            r = A.compare([sc], 2, "raster", sc_px, "apls")
            per[sc] = r
            A.fmt(r, f"{sc[:8]} {sc_px}px")
        fams = list(next(iter(per.values())).keys())
        print("scene average:", {f: round(float(np.mean([per[s][f]["mean"] for s in EXT])), 4) for f in fams})
        A.fmt(A.compare(EXT, 2, "raster", sc_px, "apls"), f"POOLED {sc_px}px apls")
        A.fmt(A.compare(EXT, 2, "raster", sc_px, "pr2gt"), f"POOLED {sc_px}px pr2gt")

    section("4. Human-confirmed recall (human_check.json)")
    h = json.loads((R / "human_check.json").read_text())
    print({k: v for k, v in h.items() if k in ("n_tiles", "tracers_per_tile")})
    for k, v in h.items():
        if k.startswith("exp"):
            print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v)

    section("5. Official CosmiQ APLS cross-check (official_apls/results.json)")
    f = R / "official_apls" / "results.json"
    if f.exists():
        res = json.loads(f.read_text())
        a = np.array([[r["ours_2px"], r["apls_11.6"]] for r in res if r.get("ours_2px") is not None and np.isfinite(r["apls_11.6"])])
        print(f"window-runs {len(res)}; usable {len(a)}; Pearson {stats.pearsonr(a[:,0], a[:,1])[0]:.3f}; "
              f"Spearman {stats.spearmanr(a[:,0], a[:,1])[0]:.3f}; mean |diff| {np.abs(a[:,0]-a[:,1]).mean():.4f}")
    else:
        print("not yet computed")


if __name__ == "__main__":
    section("6. Official CosmiQ/SpaceNet APLS on ALL 1.5-km windows (vapls/*__official.json), Tier 2, snap 11.6 m")
    FO = {k: v for k, v in A.FAMILIES.items() if k != "unet_cldice"}
    for lab, sc in (("TEST pooled", TEST), ("EXTERNAL pooled", EXT)):
        A.fmt(A.compare(sc, 2, "raster", 256, "apls", families=FO, suffix="__official"), f"OFFICIAL {lab}")
    for sc in EXT:
        A.fmt(A.compare([sc], 2, "raster", 256, "apls", families={"ours": FO["ours"], "unet": FO["unet"]}, suffix="__official"), f"OFFICIAL {sc[:8]}")
