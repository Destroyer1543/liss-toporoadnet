"""Build the simulated-LISS-IV proxy benchmark from a high-resolution road dataset.

Pipeline (per source image + mask, e.g. DeepGlobe 50 cm RGB 1024x1024):
  1. Area-downsample image by `factor` (default ~11.6x: 50 cm -> 5.8 m).
  2. Simulate G/R/NIR: G,R taken from RGB; NIR proxied from a vegetation-index
     heuristic on the source (documented in RESEARCH.md — this is a proxy, and
     the paper reports it as such; real LISS-IV is the headline set).
  3. Downsample the mask with max-pooling then re-thin, so 1-px roads survive.
  4. Save chips + masks as .npz tiles with region-based split assignment.

Blocked on: disk space for the source dataset. Run once space is freed:
  python scripts/build_proxy.py --src /data/road_liss4/deepglobe --out /data/road_liss4/proxy
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


def downsample_image(img, factor):
    """img: (H,W,C) float32. Exact area pooling via reshape when divisible."""
    h, w, c = img.shape
    hh, ww = h // factor * factor, w // factor * factor
    img = img[:hh, :ww]
    return img.reshape(hh // factor, factor, ww // factor, factor, c).mean(axis=(1, 3))


def downsample_mask(mask, factor, keep_frac=0.15):
    """Max-pool-ish: a coarse pixel is road if >= keep_frac of source px are road.
    keep_frac tuned so 1-source-lane roads survive without fattening everything."""
    h, w = mask.shape
    hh, ww = h // factor * factor, w // factor * factor
    m = mask[:hh, :ww].reshape(hh // factor, factor, ww // factor, factor)
    return (m.mean(axis=(1, 3)) >= keep_frac).astype(np.uint8)


def simulate_grnir(rgb_small):
    """rgb_small: (h,w,3) in [0,1]. Returns (3,h,w): G, R, NIR-proxy.
    NIR proxy: vegetation (green-dominant, darker red) -> high NIR;
    built/road (low saturation, bright) -> low-mid NIR."""
    r, g, b = rgb_small[..., 0], rgb_small[..., 1], rgb_small[..., 2]
    exg = np.clip(2 * g - r - b, 0, None)          # excess-green vegetation proxy
    brightness = (r + g + b) / 3
    nir = np.clip(0.15 + 1.8 * exg + 0.3 * brightness, 0, 1)
    return np.stack([g, r, nir]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with paired *_sat.jpg / *_mask.png (DeepGlobe layout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=int, default=12, help="downsample factor (50cm*12=6m ~ LISS-IV)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow required")

    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    for split in ("train", "val", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)

    sats = sorted(src.glob("*_sat.jpg"))
    if not sats:
        sys.exit(f"no *_sat.jpg found in {src}")
    rng = np.random.default_rng(0)
    # region-ish split: sort by id and split contiguously (DeepGlobe ids cluster spatially)
    n = len(sats)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    split_of = {}
    for i, p in enumerate(sats):
        split_of[p] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    kept = skipped = 0
    for p in sats:
        mask_p = src / p.name.replace("_sat.jpg", "_mask.png")
        if not mask_p.exists():
            skipped += 1
            continue
        rgb = np.asarray(Image.open(p), dtype=np.float32) / 255.0
        mask = (np.asarray(Image.open(mask_p).convert("L")) > 127).astype(np.uint8)
        img_s = simulate_grnir(downsample_image(rgb, args.factor))
        mask_s = downsample_mask(mask, args.factor)
        if mask_s.mean() < 0.002:  # nearly empty tile: keep a few, drop most
            if rng.random() > 0.1:
                skipped += 1
                continue
        np.savez_compressed(out / split_of[p] / (p.stem.replace("_sat", "") + ".npz"),
                            image=img_s, mask=mask_s)
        kept += 1
    print(f"kept {kept}, skipped {skipped}; output -> {out}")


if __name__ == "__main__":
    main()
