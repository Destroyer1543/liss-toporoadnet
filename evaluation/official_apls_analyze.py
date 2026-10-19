import json, pathlib
import numpy as np
from scipy import stats

R = pathlib.Path("data/rebuttal/official_apls")
FAM = {"ours": ["exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2"],
       "unet": ["exp16_unet_full", "exp23_unet_seed1", "exp25_unet_seed2"],
       "dlinknet": ["exp17_dlinknet_full", "exp36_dlinknet_seed1", "exp37_dlinknet_seed2"],
       "roadformer": ["exp26_roadformer_full"]}
TEST = ("R2F24AUG", "RAF23OCT")
res = json.loads((R / "results.json").read_text())
errs = sum(1 for r in res for k in r if k.startswith("err_"))
print(f"window-runs: {len(res)}; official-code exceptions recorded: {errs}")

for snap in ("11.6", "4.0", "17.4"):
    a = np.array([[r["ours_2px"], r[f"apls_{snap}"]] for r in res
                  if r.get("ours_2px") is not None and np.isfinite(r[f"apls_{snap}"])])
    print(f"[1] snap {snap} m: n={len(a)}  Pearson {stats.pearsonr(a[:,0], a[:,1])[0]:.3f}  "
          f"Spearman {stats.spearmanr(a[:,0], a[:,1])[0]:.3f}  mean ours {a[:,0].mean():.4f}  mean official {a[:,1].mean():.4f}  "
          f"mean |diff| {np.abs(a[:,0]-a[:,1]).mean():.4f}")

def table(key):
    out = {}
    for r in res:
        out.setdefault((r["scene"], r["y"], r["x"]), {})[r["run"]] = r[key]
    return out

def fam_means(tab, scenes):
    w = [k for k in tab if k[0][:8] in scenes]
    per = {f: np.array([np.nanmean([tab[k][run] for run in runs]) for k in w]) for f, runs in FAM.items()}
    return w, per

def boot(w, d, n=1000):
    rng = np.random.default_rng(0)
    by = {}
    for i, k in enumerate(w):
        by.setdefault(k[0], []).append(i)
    bs = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(v, len(v), replace=True) for v in by.values()])
        bs.append(np.nanmean(d[idx]))
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

for snap in ("11.6", "4.0", "17.4"):
    T_off, T_our = table(f"apls_{snap}"), table("ours_2px")
    for name, scenes in (("TEST (2 scenes)", TEST), ("EXTERNAL (4 scenes)", None)):
        sel = scenes if scenes else tuple(sorted({k[0][:8] for k in T_off} - set(TEST)))
        w, off = fam_means(T_off, sel); _, our = fam_means(T_our, sel)
        line = f"[2/3] snap {snap} m {name:20s} n_win={len(w)}  " + "  ".join(
            f"{f} off {np.nanmean(off[f]):.4f} (ours-impl {np.nanmean(our[f]):.4f})" for f in FAM)
        print(line)
        for f in ("unet", "dlinknet", "roadformer"):
            d = off["ours"] - off[f]; lo, hi = boot(w, d)
            dd = our["ours"] - our[f]; lo2, hi2 = boot(w, dd)
            print(f"      ours − {f:10s}: official {np.nanmean(d):+.4f} [{lo:+.4f},{hi:+.4f}]{'*' if lo > 0 or hi < 0 else ' '}"
                  f"   our implementation (same windows) {np.nanmean(dd):+.4f} [{lo2:+.4f},{hi2:+.4f}]{'*' if lo2 > 0 or hi2 < 0 else ' '}")
