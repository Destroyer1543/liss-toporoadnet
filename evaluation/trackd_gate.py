import json, pathlib, sys
import numpy as np
from btrack_screen import score
R = pathlib.Path("data/rebuttal")
RUNS = ["exp68_wide_seed0", "exp69_wide_seed1", "exp70_wide_seed2"]
s = lambda run, sc, k: score(run, "F05", "", sc, k)
v = {k: float(np.mean([s(r, sc, kk) for r in RUNS])) for k, (sc, kk) in
     {"apls15": (256, "apls"), "gt2pr15": (256, "gt2pr"), "apls6": (1024, "apls")}.items()}
t = [np.load(R / "metrics_val" / f"{r}__F05.npz") for r in RUNS]
v["aplsr"] = float(np.mean([np.nanmean(d["t2_apls"]) for d in t]))
v["iou"] = float(np.mean([(d["t2_tp"] / (d["t2_tp"] + d["t2_fp"] + d["t2_fn"] + 1e-6)).mean() for d in t]))
per_seed = {r: round(s(r, 256, "apls"), 4) for r in RUNS}
g = {"G1 apls15 >= 0.0367": v["apls15"] >= 0.0367, "G2 apls6 >= 0.0137 and gt2pr15 >= 0.0351": v["apls6"] >= 0.0137 and v["gt2pr15"] >= 0.0351,
     "G3 val APLS_r >= 0.2272 and IoU >= 0.0567": v["aplsr"] >= 0.2272 and v["iou"] >= 0.0567}
print("ours-wide validation 3-seed means:", {k: round(x, 4) for k, x in v.items()}, "per-seed APLS@1.5km:", per_seed)
for k, ok in g.items(): print(("PASS " if ok else "fail ") + k)
verdict = all(g.values()); print("TRACK D GATE:", "PASS" if verdict else "FAIL")
(R / "trackd_gate.json").write_text(json.dumps({"means": v, "per_seed": per_seed, "gates": g, "pass": verdict}, indent=1))
sys.exit(0 if verdict else 3)
