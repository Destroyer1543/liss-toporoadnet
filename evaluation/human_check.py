import base64, json, pathlib, sys
import numpy as np
from scipy.ndimage import zoom, binary_dilation

SUB = pathlib.Path("human_submissions/submissions")
GT = pathlib.Path("data/annot_gt")
MAP = json.loads(pathlib.Path("data/rebuttal/human_tile_map.json").read_text())
NAMES = json.loads(pathlib.Path("data/rebuttal/preds_test/tiles.json").read_text())
PRED = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data/rebuttal/preds_test")


def unpack(b64, res):
    bits = np.unpackbits(np.frombuffer(base64.b64decode(b64), np.uint8), bitorder="little")[:res * res]
    m = bits.reshape(res, res).astype(float)
    return (zoom(m, 256 / res, order=0) > 0.5) if res != 256 else m.astype(bool)


def load_tracings():
    latest = {}
    for f in SUB.glob("*/*.json"):
        r = json.loads(f.read_text())
        key = (r["student"], r["tile"]["id"])
        if key not in latest or r["server_time"] > latest[key][0]:
            hm = np.zeros((256, 256), bool) if r["tile"]["noRoads"] else unpack(r["tile"]["mask"], r.get("res", 128))
            latest[key] = (r["server_time"], hm)
    per = {}
    for (stu, tid), (_, hm) in latest.items():
        gt = np.load(GT / f"{tid}.npy").astype(bool)
        if hm.sum() > 6 * max(gt.sum(), 1):
            continue
        per.setdefault(tid, []).append(hm)
    return per


def main(runs):
    per = load_tracings()
    idx = {n: i for i, n in enumerate(NAMES)}
    out = {"n_tiles": len(per), "tracers_per_tile": float(np.mean([len(v) for v in per.values()]))}
    tot_conf = tot_unconf = 0
    for rn in runs:
        P = np.unpackbits(np.load(PRED / f"{rn}.npy"), axis=-1).astype(bool)
        hc = hu = nc = nu = pred_n = pred_human_only = 0
        for tid, hms in per.items():
            gt = np.load(GT / f"{tid}.npy").astype(bool)
            cnt = np.sum(hms, axis=0)
            conf = gt & (cnt >= 2); unconf = gt & (cnt < 2)
            p = P[idx[MAP[tid]]]
            hc += (p & conf).sum(); nc += conf.sum(); hu += (p & unconf).sum(); nu += unconf.sum()
            human_only = (cnt >= 2) & ~binary_dilation(gt, iterations=2)
            pred_n += p.sum(); pred_human_only += (p & human_only).sum()
        out[rn] = {"recall_confirmed": hc / max(nc, 1), "recall_unconfirmed": hu / max(nu, 1),
                   "n_confirmed_px": int(nc), "n_unconfirmed_px": int(nu),
                   "pred_share_on_human_only_roads": pred_human_only / max(pred_n, 1)}
        print(rn, {k: round(float(v), 4) for k, v in out[rn].items()}, flush=True)
    pathlib.Path("data/rebuttal/human_check.json").write_text(json.dumps(out, indent=1, default=float))
    print("tiles", out["n_tiles"], "tracers/tile", round(out["tracers_per_tile"], 2))


if __name__ == "__main__":
    main(sys.argv[1].split(","))
