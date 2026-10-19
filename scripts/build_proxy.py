import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


def downsample_image(img, factor):
    h, w, c = img.shape
    hh, ww = h // factor * factor, w // factor * factor
    img = img[:hh, :ww]
    return img.reshape(hh // factor, factor, ww // factor, factor, c).mean(axis=(1, 3))


def downsample_mask(mask, factor, keep_frac=0.15):
    h, w = mask.shape
    hh, ww = h // factor * factor, w // factor * factor
    m = mask[:hh, :ww].reshape(hh // factor, factor, ww // factor, factor)
    return (m.mean(axis=(1, 3)) >= keep_frac).astype(np.uint8)


def simulate_grnir(rgb_small):
    r, g, b = rgb_small[..., 0], rgb_small[..., 1], rgb_small[..., 2]
    exg = np.clip(2 * g - r - b, 0, None)
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
        if mask_s.mean() < 0.002:
            if rng.random() > 0.1:
                skipped += 1
                continue
        np.savez_compressed(out / split_of[p] / (p.stem.replace("_sat", "") + ".npz"),
                            image=img_s, mask=mask_s)
        kept += 1
    print(f"kept {kept}, skipped {skipped}; output -> {out}")


if __name__ == "__main__":
    main()
