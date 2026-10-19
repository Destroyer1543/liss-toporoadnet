import sys, json, base64, pathlib, numpy as np
from scipy.ndimage import distance_transform_edt, zoom

GT = pathlib.Path("data/annot_gt")

def unpack(mask_b64, Mres):
    raw = base64.b64decode(mask_b64)
    bits = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little")[:Mres*Mres]
    return bits.reshape(Mres, Mres).astype(bool)

def main(subdir):
    subdir = pathlib.Path(subdir)
    recs = []
    for jf in subdir.glob("*.jsonl"):
        for line in open(jf):
            line = line.strip()
            if line:
                try: recs.append(json.loads(line))
                except Exception: pass
    students = {r.get("student") for r in recs}
    print(f"{len(recs)} tile-annotations from {len(students)} students")

    per_tile = {}
    for r in recs:
        t = r["tile"]; Mres = r.get("res", 128)
        hm = np.zeros((256,256), bool) if t["noRoads"] else zoom(unpack(t["mask"], Mres).astype(float), 256/Mres, order=0) > 0.5
        per_tile.setdefault(t["id"], {"cat": t["cat"], "hits": []})["hits"].append(hm)

    width_bins = [(0,1.5),(1.5,2.5),(2.5,4),(4,999)]
    hit={b:0 for b in width_bins}; tot={b:0 for b in width_bins}
    cat_hit={}; cat_tot={}
    for tid, info in per_tile.items():
        gtp = GT/f"{tid}.npy"
        if not gtp.exists(): continue
        gt = np.load(gtp).astype(bool)
        if gt.sum()==0: continue
        width = distance_transform_edt(gt)*2
        for hm in info["hits"]:
            for b in width_bins:
                sel = gt & (width>=b[0]) & (width<b[1])
                tot[b]+=int(sel.sum()); hit[b]+=int((sel&hm).sum())
            c=info["cat"]; cat_tot[c]=cat_tot.get(c,0)+int(gt.sum()); cat_hit[c]=cat_hit.get(c,0)+int((gt&hm).sum())

    print("\n=== Human recall vs road width (observability curve) ===")
    for b in width_bins:
        r = hit[b]/tot[b] if tot[b] else 0
        hi = "inf" if b[1]==999 else f"{b[1]*5.8:.0f}"
        print(f"  {b[0]*5.8:3.0f}-{hi:>4} m : recall {r:.3f}  (n={tot[b]})")
    print("\n=== Human recall by tile class ===")
    for c in ["highway","major","minor"]:
        if cat_tot.get(c): print(f"  {c:8s}: {cat_hit[c]/cat_tot[c]:.3f}")
    n=[len(v['hits']) for v in per_tile.values()]
    if n: print(f"\n  tiles annotated: {len(per_tile)}/24 | tracers/tile avg {np.mean(n):.1f}")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "annotation/submissions")
