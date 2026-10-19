import json, pathlib
import numpy as np
import aggregate_families as A

R = pathlib.Path("data/rebuttal")
man = json.loads((R / "BTRACK_FROZEN_MANIFEST.json").read_text())
SEL, ORIG, RUNS = man["selected"], man["original"], man["runs"]
TEST, EXT = man["evaluation"]["scenes_test"], man["evaluation"]["scenes_external"]


def families(which):
    out = {}
    for f, runs in RUNS.items():
        d = SEL[f] if which == "selected" else ORIG[f]
        out[f] = [r if d == ORIG[f] else f"{r}__{d}" for r in runs]
    return out


def block(label, scenes, which, src="raster", tiers=(2,), keys=("apls", "gt2pr", "pr2gt")):
    print(f"\n### {label} — {which} decoders {[(f, SEL[f] if which == 'selected' else ORIG[f]) for f in RUNS]}")
    res = {}
    for t in tiers:
        for key in keys:
            for sc in (256, 1024):
                r = A.compare(scenes, t, src, sc, key, families=families(which))
                A.fmt(r, f"T{t} {src} {sc}px {key:6s}")
                res[(t, key, sc)] = r
    return res


if __name__ == "__main__":
    print("manifest frozen at", man["frozen_at"], "selected:", SEL)
    out = {}
    for which in ("selected", "original"):
        for sc in TEST:
            out[(which, sc[:8])] = block(f"TEST {sc[:8]}", [sc], which, tiers=(2, 1))
        out[(which, "TEST")] = block("TEST pooled", TEST, which, tiers=(2, 1))
        for sc in EXT:
            out[(which, sc[:8])] = block(f"EXTERNAL {sc[:8]}", [sc], which)
        out[(which, "EXT")] = block("EXTERNAL pooled (raster GT)", EXT, which)
        block("EXTERNAL pooled (OSM-centreline GT)", EXT, which, src="vector", keys=("apls",))
    ok, why = True, []
    for unit in [TEST[0][:8], TEST[1][:8], "EXT"]:
        r = out[("selected", unit)][(2, "apls", 256)]
        for f, v in r.items():
            if f == "ours":
                continue
            lo = v.get("ci", (np.nan, np.nan))[0]
            if not (lo > 0):
                ok = False; why.append(f"{unit}: vs {f} Δ{v.get('diff', np.nan):+.4f} CI-low {lo:+.4f}")
    print("\nSUCCESS RULE:", "PASS" if ok else "FAIL", "; ".join(why))
