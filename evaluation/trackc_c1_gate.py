import json, pathlib, sys
import numpy as np
from btrack_screen import score

R = pathlib.Path("data/rebuttal")
CAND, ORIG, CTL = "exp52_c1_ours_bal_s0__F05", "exp15_toporoad_full__F05", "exp53_c1_ours_ctl_s0__F05"


def strict(name):
    run, dec = name.rsplit("__", 1)
    return {"apls15": score(run, dec, "", 256, "apls"), "pr2gt15": score(run, dec, "", 256, "pr2gt"),
            "gt2pr15": score(run, dec, "", 256, "gt2pr"), "apls6": score(run, dec, "", 1024, "apls")}


def tiles(name):
    d = np.load(R / "metrics_val" / f"{name}.npz")
    tp, fp, fn = d["t2_tp"], d["t2_fp"], d["t2_fn"]
    return {"aplsr": float(np.nanmean(d["t2_apls"])), "iou": float((tp / (tp + fp + fn + 1e-6)).mean())}


def loc(name):
    return json.loads((R / "loc_metrics" / f"{name}.json").read_text())


def main():
    m = {}
    for n in (ORIG, CTL, CAND):
        m[n] = {**strict(n), **tiles(n), **loc(n)}
        print(n, {k: round(v, 4) for k, v in m[n].items()})
    c, o, k = m[CAND], m[ORIG], m[CTL]
    best = lambda key: max(o[key], k[key])
    cond = {
        "1 apls15 >= max ref + 0.003 and pr2gt > max ref": c["apls15"] >= best("apls15") + 0.003 and c["pr2gt15"] > best("pr2gt15"),
        "2 apls6 and gt2pr15 >= max ref - 0.001": c["apls6"] >= best("apls6") - 0.001 and c["gt2pr15"] >= best("gt2pr15") - 0.001,
        "3 val APLS_r and IoU >= max ref": c["aplsr"] >= best("aplsr") and c["iou"] >= best("iou"),
        "4 far10 <= 0.85x orig and width <= 0.90x orig": c["far10_norm"] <= 0.85 * o["far10_norm"] and c["width_px"] <= 0.90 * o["width_px"],
        "5 osm_within2 >= orig - 0.005": c["osm_within2"] >= o["osm_within2"] - 0.005,
    }
    for t, ok in cond.items():
        print(("PASS " if ok else "fail ") + t)
    verdict = all(cond.values())
    print("C1 SEED-0 GATE:", "PASS" if verdict else "FAIL")
    (R / "trackc_c1_gate.json").write_text(json.dumps({"metrics": m, "conditions": cond, "pass": verdict}, indent=1))
    return verdict


if __name__ == "__main__":
    sys.exit(0 if main() else 3)
