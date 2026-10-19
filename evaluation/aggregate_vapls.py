import json, pathlib, sys
import numpy as np

V = pathlib.Path("data/rebuttal/vapls")
OURS = ["exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2"]
UNET = ["exp16_unet_full", "exp23_unet_seed1", "exp25_unet_seed2"]
OTHER = ["exp17_dlinknet_full", "exp26_roadformer_full"]


_CACHE = {}


def load(scene, run):
    key = (scene, run)
    if key not in _CACHE:
        _CACHE[key] = None
        for p in (V / f"{scene}__{run}.json", V / f"{scene}__{run}.partial.json"):
            if p.exists():
                _CACHE[key] = json.loads(p.read_text()); break
    return _CACHE[key]


def table(scenes, buffer=2, key="apls"):
    rows = []
    for tier in (1, 2):
        for src in ("raster", "vector"):
            for scale in (256, 1024, 4096, 0):
                b = buffer
                per = {}
                for sc in scenes:
                    for r in OURS + UNET + OTHER:
                        d = load(sc, r)
                        if d is None:
                            continue
                        for w in d:
                            if w["tier"] == tier and w["gt"] == src and w["scale"] == scale and w["buffer"] == b:
                                v = w[key]
                                per.setdefault(r, {})[(sc, w["y"], w["x"])] = np.nan if v is None else v
                if not per:
                    continue
                means = {r: np.nanmean(list(v.values())) for r, v in per.items()}
                o = [means[r] for r in OURS if r in means]; u = [means[r] for r in UNET if r in means]
                row = {"tier": tier, "gt": src, "scale": scale, "buffer": b, "key": key,
                       "ours": (np.mean(o), np.std(o, ddof=1) if len(o) > 1 else np.nan, len(o)),
                       "unet": (np.mean(u), np.std(u, ddof=1) if len(u) > 1 else np.nan, len(u)),
                       "dlinknet": means.get("exp17_dlinknet_full"), "roadformer": means.get("exp26_roadformer_full")}
                ro = [r for r in OURS if r in per]; ru = [r for r in UNET if r in per]
                if ro and ru:
                    keys = set.intersection(*[set(per[r]) for r in ro + ru])
                    keys = sorted(keys)
                    diff = np.array([np.nanmean([per[r][k] for r in ro]) - np.nanmean([per[r][k] for r in ru]) for k in keys])
                    ok = np.isfinite(diff); keys = [k for k, f in zip(keys, ok) if f]; diff = diff[ok]
                    blk = [(k[0], k[1] // 1024, k[2] // 1024) if scale == 256 else k for k in keys]
                    groups = {}
                    for i, g in enumerate(blk):
                        groups.setdefault(g, []).append(i)
                    gl = list(groups.values())
                    rng = np.random.default_rng(0)
                    boots = []
                    if len(gl) > 1:
                        bysc = {}
                        for gi, g in enumerate(gl):
                            bysc.setdefault(keys[g[0]][0], []).append(gi)
                        for _ in range(1000):
                            idx = []
                            for sc, gis in bysc.items():
                                pick = rng.choice(gis, len(gis), replace=True)
                                for p in pick:
                                    idx.extend(gl[p])
                            boots.append(diff[idx].mean())
                    row["diff"] = float(diff.mean()) if len(diff) else np.nan
                    row["ci95"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (np.nan, np.nan)
                    row["n_windows"] = len(diff); row["n_blocks"] = len(gl)
                    row["frac_win_ours_better"] = float((diff > 0).mean()) if len(diff) else np.nan
                rows.append(row)
    return rows


def fmt(rows):
    for r in rows:
        sc = "full" if r["scale"] == 0 else f"{r['scale']}px"
        o, u = r["ours"], r["unet"]
        s = (f"T{r['tier']} {r['gt']:6s} {sc:7s} b{r['buffer']} {r['key']:6s} ours {o[0]:.4f}±{o[1]:.4f}(n{o[2]}) "
             f"unet {u[0]:.4f}±{u[1]:.4f}(n{u[2]}) dlink {r['dlinknet'] if r['dlinknet'] is None else round(r['dlinknet'],4)} "
             f"rf {r['roadformer'] if r['roadformer'] is None else round(r['roadformer'],4)}")
        if "diff" in r:
            s += f" | diff {r['diff']:+.4f} CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] win {r['n_windows']} ours>unet {r['frac_win_ours_better']:.2f}"
        print(s)


if __name__ == "__main__":
    scenes = sys.argv[1].split(","); b = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    for key in ("apls", "gt2pr", "pr2gt", "route25"):
        fmt(table(scenes, b, key))
