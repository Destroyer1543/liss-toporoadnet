"""Whole-scene holdout splits for LISS-IV tiles (v2 — replaces spatial-band v1).

Test scenes (never seen in training, one per satellite, spanning road density):
  R2F24AUG2023... (RS-2,  road frac 0.020)
  RAF23OCT2023... (RS-2A, road frac 0.011)
Val scene: R2F08FEB2024... (0.011)
Train: remaining 7 scenes, including the singleton extremes (urban 0.050,
forest 0.002) which must appear in training to be learnable at all.

Clears previous split symlinks (v1 spatial bands) before rebuilding.
"""

import pathlib

ROOT = pathlib.Path("/data/road_liss4/liss4_tiles")
TEST = ["R2F24AUG2023064073010100066SSANSTUC00GTDD",
        "RAF23OCT2023035685010100064SSANSTUC00GTDA"]
VAL = ["R2F08FEB2024066453010100064SSANSTUC00GTDA"]


def main():
    for split in ("train", "val", "test"):
        d = ROOT / split
        d.mkdir(exist_ok=True)
        for link in d.glob("*.npz"):
            link.unlink()
    counts = {"train": 0, "val": 0, "test": 0}
    scenes = [d for d in ROOT.iterdir() if d.is_dir() and d.name not in ("train", "val", "test")]
    assert len(scenes) == 10, f"expected 10 scenes, found {len(scenes)}"
    for scene in scenes:
        split = "test" if scene.name in TEST else ("val" if scene.name in VAL else "train")
        for t in scene.glob("*.npz"):
            link = ROOT / split / f"{scene.name}__{t.name}"
            link.symlink_to(t.resolve())
            counts[split] += 1
    print(counts)


if __name__ == "__main__":
    main()
