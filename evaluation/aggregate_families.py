import json, pathlib, sys
import numpy as np

V = pathlib.Path("data/rebuttal/vapls")
FAMILIES = {
    "ours": ["exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2"],
    "unet": ["exp16_unet_full", "exp23_unet_seed1", "exp25_unet_seed2"],
    "dlinknet": ["exp17_dlinknet_full", "exp36_dlinknet_seed1", "exp37_dlinknet_seed2"],
    "roadformer": ["exp26_roadformer_full"],
    "unet_cldice": ["exp47_unet_cldice_seed0"],
}
_CACHE = {}


def load(scene, run):
    if (scene, run) not in _CACHE:
        _CACHE[(scene, run)] = None
        for p in (V / f"{scene}__{run}.json", V / f"{scene}__{run}.partial.json"):
            if p.exists():
                _CACHE[(scene, run)] = json.loads(p.read_text()); break
    return _CACHE[(scene, run)]


def windows(scenes, run, tier, src, scale, key, buffer=2):
    out = {}
    for sc in scenes:
        d = load(sc, run)
        for w in d or []:
            if w["tier"] == tier and w["gt"] == src and w["scale"] == scale and w["buffer"] == buffer:
                out[(sc, w["y"], w["x"])] = np.nan if w[key] is None else w[key]
    return out


def bootstrap(keys, diff, scale, n=1000):
    blk = [(k[0], k[1] // 1024, k[2] // 1024) if scale == 256 else k for k in keys]
    groups = {}
    for i, g in enumerate(blk):
        groups.setdefault(g, []).append(i)
    gl = list(groups.values())
    if len(gl) < 2:
        return (np.nan, np.nan)
    bysc = {}
    for gi, g in enumerate(gl):
        bysc.setdefault(keys[g[0]][0], []).append(gi)
    rng = np.random.default_rng(0); boots = []
    for _ in range(n):
        idx = []
        for gis in bysc.values():
            for p in rng.choice(gis, len(gis), replace=True):
                idx.extend(gl[p])
        boots.append(diff[idx].mean())
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def compare(scenes, tier, src, scale, key, families=FAMILIES, ref="ours", suffix="", buffer=2):
    per = {f: {r: windows(scenes, r + suffix, tier, src, scale, key, buffer) for r in runs} for f, runs in families.items()}
    per = {f: {r: v for r, v in d.items() if v} for f, d in per.items()}
    res = {}
    for f, d in per.items():
        if not d:
            continue
        m = [np.nanmean(list(v.values())) for v in d.values()]
        res[f] = {"mean": float(np.mean(m)), "sd": float(np.std(m, ddof=1)) if len(m) > 1 else np.nan, "n": len(m)}
    if ref not in res:
        return res
    for f in res:
        if f == ref:
            continue
        runs_r, runs_f = list(per[ref].values()), list(per[f].values())
        keys = sorted(set.intersection(*[set(v) for v in runs_r + runs_f]))
        diff = np.array([np.nanmean([v[k] for v in runs_r]) - np.nanmean([v[k] for v in runs_f]) for k in keys])
        ok = np.isfinite(diff); keys = [k for k, o in zip(keys, ok) if o]; diff = diff[ok]
        if len(diff):
            res[f].update(diff=float(diff.mean()), ci=bootstrap(keys, diff, scale), nwin=len(diff))
    return res


def fmt(res, label):
    s = [label]
    for f, r in res.items():
        t = f"{f} {r['mean']:.4f}" + (f"±{r['sd']:.4f}" if r["n"] > 1 else "") + f"(n{r['n']})"
        if "diff" in r:
            sig = "*" if (r["ci"][0] > 0 or r["ci"][1] < 0) else " "
            t += f" Δ{r['diff']:+.4f}[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]{sig}"
        s.append(t)
    print("  ".join(s), flush=True)


if __name__ == "__main__":
    scenes = sys.argv[1].split(",")
    tier = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    src = sys.argv[3] if len(sys.argv) > 3 else "raster"
    suffix = sys.argv[4] if len(sys.argv) > 4 else ""
    for key in ("apls", "gt2pr", "pr2gt"):
        for scale in (256, 1024):
            fmt(compare(scenes, tier, src, scale, key, suffix=suffix), f"T{tier} {src} {scale}px {key:6s}")
