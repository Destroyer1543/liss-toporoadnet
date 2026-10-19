import json, pathlib, sys
import numpy as np

V = pathlib.Path("data/rebuttal/vapls")
VAL = "R2F08FEB2024066453010100064SSANSTUC00GTDA"
FAM = {"ours": ["exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2"],
       "unet": ["exp16_unet_full", "exp23_unet_seed1", "exp25_unet_seed2"],
       "dlinknet": ["exp17_dlinknet_full", "exp36_dlinknet_seed1", "exp37_dlinknet_seed2"],
       "roadformer": ["exp26_roadformer_full"]}
A2 = ["BS2", "BS1", "BH2", "BH1"]
DEC = {"ours": ["F05", "M05", "S03", "S05", "S07", "FS", "H1", "H2", "H3", "FH", "FW1", "FWC"] + A2,
       "unet": ["T03", "T05", "T07", "H1", "H2", "H3"] + A2, "dlinknet": ["T03", "T05", "T07", "H1", "H2", "H3"] + A2,
       "roadformer": ["T03", "T05", "T07", "H1", "H2", "H3"] + A2}
ORIG = {"ours": "F05", "unet": "T05", "dlinknet": "T05", "roadformer": "T05"}
TIE = 0.002


def score(run, dec, suffix, scale, key):
    p = V / f"{VAL}__{run}__{dec}{suffix}.json"
    if not p.exists():
        return None
    v = [w[key] for w in json.loads(p.read_text())
         if w["tier"] == 2 and w["gt"] == "raster" and w["scale"] == scale and w["buffer"] == 2]
    v = [np.nan if x is None else x for x in v]
    return float(np.nanmean(v)) if v else None


def main(suffix=""):
    if suffix:
        fin = json.loads((V.parent / "btrack_screen_screen.json").read_text())
        decs = {f: [ORIG[f]] + fin[f]["_finalists"] for f in FAM}
    else:
        decs = DEC
    missing = [f"{r}__{d}" for f in FAM for d in decs[f] for r in FAM[f]
               if not (V / f"{VAL}__{r}__{d}{suffix}.json").exists()]
    if missing:
        print(f"INCOMPLETE ({len(missing)} candidate files missing) — not showing any score:", missing[:6], "...")
        return
    scales = (256,) if suffix else (256, 1024)
    out = {}
    for f in FAM:
        rows = {}
        for d in decs[f]:
            r = {}
            for sc in scales:
                for key in (("apls", "gt2pr", "pr2gt") if sc == 256 else ("apls",)):
                    vals = [score(run, d, suffix, sc, key) for run in FAM[f]]
                    r[f"{key}@{sc}"] = float(np.mean(vals)); r[f"{key}@{sc}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if suffix:
                r["apls@1024"] = score_mean_6km(f, d)
            rows[d] = r
        o = rows.get(ORIG[f])
        for d, r in rows.items():
            if d == ORIG[f] or o is None:
                r["eligible"] = True
            else:
                ok = r["gt2pr@256"] >= o["gt2pr@256"]
                if "apls@1024" in r:
                    ok &= r["apls@1024"] >= o["apls@1024"]
                r["eligible"] = bool(ok)
        out[f] = rows
        print(f"## {f} ({len(FAM[f])} seed{'s' if len(FAM[f]) > 1 else ''}), original = {ORIG[f]}")
        for d, r in sorted(rows.items(), key=lambda kv: -kv[1]["apls@256"]):
            s = f"  {d:4s} APLS@1.5km {r['apls@256']:.4f}±{r['apls@256_sd']:.4f}"
            if "apls@1024" in r:
                s += f"  APLS@6km {r['apls@1024']:.4f}"
            s += f"  GT→pred {r['gt2pr@256']:.4f}  pred→GT {r['pr2gt@256']:.4f}  {'eligible' if r['eligible'] else 'GATED'}"
            print(s)
        elig = [d for d, r in rows.items() if r["eligible"] and d != ORIG[f]]
        elig.sort(key=lambda d: -rows[d]["apls@256"])
        if not suffix:
            print(f"  → finalists for full validation: {elig[:2]} (+ original {ORIG[f]} as reference)")
            out[f]["_finalists"] = elig[:2]
        else:
            best = max([ORIG[f]] + elig, key=lambda d: rows[d]["apls@256"])
            top = rows[best]["apls@256"]
            tied = [d for d in [ORIG[f]] + elig if top - rows[d]["apls@256"] < TIE]
            if len(tied) > 1:
                sc6 = {d: score_mean_6km(f, d) for d in tied}
                best6 = max(tied, key=lambda d: sc6[d] if sc6[d] is not None else -1)
                tied6 = [d for d in tied if sc6[d] is not None and sc6[best6] - sc6[d] < TIE]
                best = ORIG[f] if ORIG[f] in tied6 else best6
            print(f"  → SELECTED: {best}  (tied within {TIE}: {tied})")
            out[f]["_selected"] = best
    (V.parent / f"btrack_screen{suffix or '_screen'}.json").write_text(json.dumps(out, indent=1))


def score_mean_6km(f, d):
    vals = [score(run, d, "", 1024, "apls") for run in FAM[f]]
    return float(np.mean(vals)) if all(v is not None for v in vals) else None


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
