import pathlib, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from tile_metrics import one, G

EXT = pathlib.Path("data/rebuttal/external")
MOS = pathlib.Path("data/rebuttal/mosaics")
OUT = pathlib.Path("data/rebuttal/metrics_ext")
T = 256


def unpack(a, W):
    return np.unpackbits(a, axis=-1)[:, :W].astype(bool)


def select_tiles(t1, nodata):
    H, W = nodata.shape
    rng = np.random.default_rng(0)
    keep = []
    for y in range(0, H - T + 1, T):
        for x in range(0, W - T + 1, T):
            if nodata[y:y + T, x:x + T].mean() > 0.3:
                continue
            if t1[y:y + T, x:x + T].mean() < 0.001 and rng.random() > 0.1:
                continue
            keep.append((y, x))
    return keep


def main(scene, runs, procs=24):
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(EXT / f"{scene}.npz"); sh = tuple(int(v) for v in d["shape"])
    meta = np.load(MOS / scene / "meta.npz"); H, W = (int(v) for v in meta["shape"])
    nodata = unpack(meta["nodata"], W)
    t1, major, ign = (unpack(d[k], sh[1])[:H, :W] for k in ("t1", "major", "ignore"))
    cls, surf = d["cls"][:H, :W], d["surf"][:H, :W]
    tiles = select_tiles(t1, nodata)
    G["gt"] = np.stack([t1[y:y + T, x:x + T] for y, x in tiles]).astype(np.uint8)
    G["mj"] = np.stack([major[y:y + T, x:x + T] for y, x in tiles]).astype(np.uint8)
    G["ig"] = np.stack([ign[y:y + T, x:x + T] for y, x in tiles]).astype(np.uint8)
    info = {"yx": np.array(tiles), "dens": G["gt"].reshape(len(tiles), -1).mean(1)}
    print(f"{scene}: {len(tiles)} tiles kept (road-free kept at 10%)", flush=True)
    for rn in runs:
        out = OUT / f"{scene}__{rn}.npz"
        mp = MOS / scene / f"{rn}.npz"
        if out.exists() or not mp.exists():
            continue
        m = unpack(np.load(mp)["mask"], W)
        G["pred"] = np.stack([m[y:y + T, x:x + T] for y, x in tiles])
        with Pool(procs) as pool:
            rows = pool.map(one, range(len(tiles)), chunksize=16)
        P = G["pred"]
        strat = {}
        for ci, cname in enumerate(d["classes"]):
            gt_c = np.stack([(cls[y:y + T, x:x + T] == ci) for y, x in tiles])
            n = gt_c.sum()
            if n:
                strat[f"cls_{cname}_n"] = n; strat[f"cls_{cname}_hit"] = (P & gt_c).sum()
        for si, sname in ((1, "paved"), (2, "unpaved")):
            gs = np.stack([(surf[y:y + T, x:x + T] == si) for y, x in tiles])
            strat[f"surf_{sname}_n"] = gs.sum(); strat[f"surf_{sname}_hit"] = (P & gs).sum()
        keys = rows[0].keys()
        np.savez(out, **{k: np.array([r[k] for r in rows], np.float64) for k in keys},
                 **info, **{k: np.int64(v) for k, v in strat.items()})
        print(f"  {rn}: done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
