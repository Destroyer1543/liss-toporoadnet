import json, pathlib, re, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import tile_metrics as TM

VAL = "R2F08FEB2024066453010100064SSANSTUC00GTDA"
TILES = sorted(pathlib.Path("data/liss4_tiles/val").glob("*.npz"))
MOS = pathlib.Path("data/rebuttal/mosaics") / VAL
OUT = pathlib.Path("data/rebuttal/metrics_val")


def main(runs):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = np.load(MOS / "meta.npz"); H, W = (int(v) for v in meta["shape"])
    em = np.load(f"data/eval_masks/{VAL}.npz")
    emj, eig = em["major"], em["ignore"]
    pos, gt, mj, ig = [], [], [], []
    for t in TILES:
        tile = t.stem.split("__")[1]
        y, x = int(re.search(r"y(\d+)", tile).group(1)), int(re.search(r"x(\d+)", tile).group(1))
        if y + 256 > H or x + 256 > W:
            continue
        pos.append((y, x)); gt.append(np.load(t)["mask"])
        mj.append(emj[y:y + 256, x:x + 256]); ig.append(eig[y:y + 256, x:x + 256])
    TM.G.update(gt=np.stack(gt).astype(np.uint8), mj=np.stack(mj).astype(np.uint8), ig=np.stack(ig).astype(np.uint8))
    print(f"{len(pos)} validation tiles", flush=True)
    for rn in runs:
        out = OUT / f"{rn}.npz"
        if out.exists():
            continue
        m = np.unpackbits(np.load(MOS / f"{rn}.npz")["mask"], axis=-1)[:, :W].astype(bool)
        TM.G["pred"] = np.stack([m[y:y + 256, x:x + 256] for y, x in pos])
        with Pool(16) as pool:
            rows = pool.map(TM.one, range(len(pos)), chunksize=32)
        np.savez(out, **{k: np.array([r[k] for r in rows], np.float64) for k in rows[0]})
        print(rn, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
