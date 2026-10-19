import pathlib, sys
import numpy as np
from scipy.ndimage import label

MOS = pathlib.Path("data/rebuttal/mosaics")
EIGHT = np.ones((3, 3), bool)
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
OURS = ("exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2")


def n_touching(pl, npieces, bl):
    H, W = pl.shape
    pairs = []
    for dy, dx in OFFS:
        a = pl[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
        b = bl[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
        sel = (a > 0) & (b > 0)
        if sel.any():
            pairs.append(np.stack([a[sel], b[sel]], 1))
    if not pairs:
        return np.zeros(npieces + 1, np.int64)
    P = np.unique(np.concatenate(pairs), axis=0)
    return np.bincount(P[:, 0], minlength=npieces + 1)


def bridge(base, evid, ks=(2, 1)):
    pl, npc = label(evid & ~base, structure=EIGHT)
    bl, _ = label(base, structure=EIGHT)
    cnt = n_touching(pl, npc, bl)
    out = {}
    for k in ks:
        keep = cnt >= k; keep[0] = False
        out[k] = base | keep[pl]
    return out, cnt


def unpack(p, shape):
    return np.unpackbits(p, axis=-1)[:, :shape[1]].astype(bool)[:shape[0]]


def main(scene, runs):
    od = MOS / scene
    H, W = (int(v) for v in np.load(od / "meta.npz")["shape"])
    for rn in runs:
        ours = rn in OURS
        base = unpack(np.load(od / f"{rn}__{'F05' if ours else 'T05'}.npz")["mask"], (H, W))
        for e, src in (("S", "S03" if ours else "T03"), ("H", "H2")):
            if all((od / f"{rn}__B{e}{k}.npz").exists() for k in (2, 1)):
                continue
            if not (od / f"{rn}__{src}.npz").exists():
                print(f"{rn}: no {src} evidence on this scene, B{e}* skipped", flush=True)
                continue
            evid = unpack(np.load(od / f"{rn}__{src}.npz")["mask"], (H, W))
            outs, cnt = bridge(base, evid)
            for k, m in outs.items():
                np.savez_compressed(od / f"{rn}__B{e}{k}.npz", mask=np.packbits(m, axis=-1))
            added = {k: int((outs[k] & ~base).sum()) for k in outs}
            print(f"{rn}__B{e}*: pieces {len(cnt)-1}, touching>=2 {int((cnt[1:] >= 2).sum())}, >=1 "
                  f"{int((cnt[1:] >= 1).sum())}; added px k2 {added[2]} k1 {added[1]} (base {int(base.sum())})",
                  flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
