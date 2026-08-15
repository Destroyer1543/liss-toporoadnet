"""Precompute topology targets (skel/width/aff) into each tile .npz once.

Rewrites tile files in the scene dirs with extra keys: skel (uint8), width
(float16), aff (uint8). NpzTileDataset uses them when present instead of
recomputing per epoch (which cost ~12 ms/tile => ~4 min/epoch single-threaded).
Multiprocessing over plain files — no torch, no shm.
"""

import pathlib
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from targets import make_targets

ROOT = pathlib.Path("/data/road_liss4/liss4_tiles")


def process(path):
    try:
        d = dict(np.load(path))
        if "skel" in d or "mask" not in d:
            return 0
        t = make_targets(d["mask"])
        d["skel"] = t["skel"].astype(np.uint8)
        d["width"] = t["width"].astype(np.float16)
        d["aff"] = t["aff"].astype(np.uint8)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **d)
        tmp.rename(path)
        return 1
    except Exception as e:
        print(f"FAIL {path}: {e}", flush=True)
        return 0


def main():
    tiles = [p for d in ROOT.iterdir() if d.is_dir() and d.name not in ("train", "val", "test")
             for p in d.glob("*.npz")]
    print(f"{len(tiles)} tiles", flush=True)
    with Pool(16) as pool:
        done = sum(pool.imap_unordered(process, tiles, chunksize=64))
    print(f"updated {done}", flush=True)


if __name__ == "__main__":
    main()
