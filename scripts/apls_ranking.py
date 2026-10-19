import sys, pathlib, numpy as np
sys.path.insert(0, "src")
import torch
from graph_metrics import apls_raster
from apls_validate import graph_apls
from model import LISSTopoRoadNet
from baselines import BASELINES
from scipy.stats import spearmanr

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load(run, kind):
    m = (LISSTopoRoadNet() if kind == "topo" else BASELINES[kind]()).to(DEV)
    m.load_state_dict(torch.load(f"runs/{run}/best.pt", map_location=DEV, weights_only=False)["model"])
    m.eval(); return m

MODELS = [("Ours", load("exp15_toporoad_full", "topo")),
          ("U-Net", load("exp16_unet_full", "unet")),
          ("D-LinkNet", load("exp17_dlinknet_full", "dlinknet")),
          ("RoadFormer", load("exp26_roadformer_full", "roadformer"))]

tiles = sorted(pathlib.Path("data/liss4_tiles/test").glob("*.npz"))
rng = np.random.default_rng(0)
sel = [t for t in tiles if np.load(t)["mask"].mean() >= 0.03]
sel = [sel[i] for i in rng.permutation(len(sel))[:140]]

per_r = {n: [] for n, _ in MODELS}
per_g = {n: [] for n, _ in MODELS}
pool_r, pool_g = [], []
for t in sel:
    d = np.load(t); img = d["image"].astype(np.float32); gt = d["mask"].astype(bool)
    x = torch.from_numpy(img[None]).to(DEV)
    for name, m in MODELS:
        with torch.no_grad():
            out = m(x); logit = out["fused_logit"] if "fused_logit" in out else out["mask_logit"]
            pr = (torch.sigmoid(logit)[0, 0] > 0.5).cpu().numpy()
        a = apls_raster(pr, gt, n_pairs=200, rng=np.random.default_rng(1))
        b = graph_apls(gt, pr, n_pairs=200, rng=np.random.default_rng(2))
        if np.isfinite(a) and np.isfinite(b):
            per_r[name].append(a); per_g[name].append(b)
            pool_r.append(a); pool_g.append(b)

names = [n for n, _ in MODELS]
mr = {n: float(np.mean(per_r[n])) for n in names}
mg = {n: float(np.mean(per_g[n])) for n in names}
rho = spearmanr(pool_r, pool_g).statistic
pear = float(np.corrcoef(pool_r, pool_g)[0, 1])
np.savez("data/apls_ranking.npz",
         names=names, mean_r=[mr[n] for n in names], mean_g=[mg[n] for n in names],
         pool_r=np.array(pool_r), pool_g=np.array(pool_g), spearman=rho, pearson=pear)
print(f"n_tiles={len(sel)}  pooled Spearman={rho:.3f} Pearson={pear:.3f}")
print(f"{'model':<12}{'APLS_r':>9}{'graphAPLS':>11}")
for n in names:
    print(f"{n:<12}{mr[n]:>9.4f}{mg[n]:>11.4f}")
ord_r = sorted(names, key=lambda n: -mr[n]); ord_g = sorted(names, key=lambda n: -mg[n])
print("rank by APLS_r   :", " > ".join(ord_r))
print("rank by graphAPLS:", " > ".join(ord_g))
print("SAME RANKING" if ord_r == ord_g else "RANKINGS DIFFER")
