import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.draw import line as draw_line

from targets import make_targets


def _random_road_network(size, rng, n_trunks=3, n_branches=4):
    mask = np.zeros((size, size), dtype=np.uint8)
    pts = []
    for _ in range(n_trunks):
        side = rng.integers(0, 2)
        a = (rng.integers(0, size), 0) if side else (0, rng.integers(0, size))
        b = (rng.integers(0, size), size - 1) if side else (size - 1, rng.integers(0, size))
        prev = a
        n_seg = rng.integers(3, 6)
        for i in range(1, n_seg + 1):
            t = i / n_seg
            mid = (int(a[0] + (b[0] - a[0]) * t + rng.integers(-size // 8, size // 8 + 1)),
                   int(a[1] + (b[1] - a[1]) * t + rng.integers(-size // 8, size // 8 + 1)))
            mid = (np.clip(mid[0], 0, size - 1), np.clip(mid[1], 0, size - 1))
            rr, cc = draw_line(*prev, *mid)
            mask[rr, cc] = 1
            pts.extend(zip(rr[::4], cc[::4]))
            prev = mid
    for _ in range(n_branches):
        if not pts:
            break
        start = pts[rng.integers(0, len(pts))]
        end = (rng.integers(0, size), rng.integers(0, size))
        rr, cc = draw_line(*start, *end)
        mask[rr, cc] = 1
    return mask


def _spectral_render(mask_w, size, rng):
    base = gaussian_filter(rng.random((size, size)), 8)
    veg = (base - base.min()) / (np.ptp(base) + 1e-6)
    g = 0.18 + 0.10 * veg + 0.04 * rng.standard_normal((size, size))
    r = 0.16 - 0.06 * veg + 0.10 * (1 - veg) * 0.5 + 0.04 * rng.standard_normal((size, size))
    nir = 0.25 + 0.35 * veg + 0.05 * rng.standard_normal((size, size))
    road = mask_w.astype(np.float32)
    road_soft = gaussian_filter(road, 0.6)
    g = g * (1 - road_soft) + (0.30 + 0.05 * rng.standard_normal((size, size))) * road_soft
    r = r * (1 - road_soft) + (0.32 + 0.05 * rng.standard_normal((size, size))) * road_soft
    nir = nir * (1 - road_soft) + (0.22 + 0.05 * rng.standard_normal((size, size))) * road_soft
    img = np.stack([g, r, nir]).astype(np.float32)
    return np.clip(img, 0, 1)


class NpzTileDataset(Dataset):

    def __init__(self, root, size=256, augment=False, with_topo_targets=True, seed=0,
                 preload=False, major_masks_dir=None, paths=None, zero_bands=None):
        import pathlib, re
        self.paths = sorted(paths) if paths is not None else sorted(pathlib.Path(root).glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no .npz tiles in {root or paths}")
        self.size, self.augment, self.with_topo = size, augment, with_topo_targets
        self.seed = seed
        self.zero_bands = zero_bands or []
        self.major = None
        if major_masks_dir:
            md = pathlib.Path(major_masks_dir)
            self.major = {p.stem: np.load(p)["major"] for p in md.glob("*.npz")}
            self._re = re
            if not self.major:
                raise FileNotFoundError(f"no major masks in {major_masks_dir}")
        self.cache = None
        if preload:
            self.cache = []
            for p in self.paths:
                d = np.load(p)
                item = {"image": d["image"].astype(np.float16),
                        "mask": d["mask"].astype(np.uint8)}
                if self.with_topo and "skel" in d.files:
                    item["skel"] = d["skel"].astype(np.uint8)
                    item["width"] = d["width"].astype(np.float16)
                    item["aff"] = d["aff"].astype(np.uint8)
                self.cache.append(item)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.cache is not None:
            d = self.cache[i]
            keys = d.keys()
        else:
            d = np.load(self.paths[i])
            keys = d.files
        img, mask = d["image"].astype(np.float32), d["mask"].astype(np.uint8)
        if img.shape[0] != 3:
            img = img.transpose(2, 0, 1)
        for b in self.zero_bands:
            img[b] = 0.0
        if self.major is not None:
            name = self.paths[i].stem
            scene, tile = name.split("__")
            y = int(self._re.search(r"y(\d+)", tile).group(1))
            x = int(self._re.search(r"x(\d+)", tile).group(1))
            mj = self.major.get(scene)
            if mj is not None:
                crop = mj[y:y + mask.shape[0], x:x + mask.shape[1]]
                if crop.shape == mask.shape:
                    mask = crop.astype(np.uint8)
            keys = ["image", "mask"]
        cached = self.with_topo and "skel" in keys
        if cached:
            skel = d["skel"].astype(np.float32)
            width = d["width"].astype(np.float32)
            aff = d["aff"].astype(np.float32)
        _, h, w = img.shape
        s = self.size
        if h < s or w < s:
            ph, pw = max(0, s - h), max(0, s - w)
            img = np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
            if cached:
                skel = np.pad(skel, ((0, ph), (0, pw)), mode="reflect")
                width = np.pad(width, ((0, ph), (0, pw)), mode="reflect")
                aff = np.pad(aff, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            _, h, w = img.shape
        if h > s or w > s:
            y0, x0 = (h - s) // 2, (w - s) // 2
            img = img[:, y0:y0 + s, x0:x0 + s]
            mask = mask[y0:y0 + s, x0:x0 + s]
            if cached:
                skel = skel[y0:y0 + s, x0:x0 + s]
                width = width[y0:y0 + s, x0:x0 + s]
                aff = aff[:, y0:y0 + s, x0:x0 + s]
        if self.augment:
            rng = np.random.default_rng()
            k = int(rng.integers(0, 4))
            do_flip = rng.random() < 0.5
            img = np.rot90(img, k, axes=(1, 2))
            mask = np.rot90(mask, k)
            if do_flip:
                img = img[:, :, ::-1]
                mask = mask[:, ::-1]
            if cached:
                skel = np.rot90(skel, k)
                width = np.rot90(width, k)
                aff = np.stack([np.rot90(aff[(j - 2 * k) % 8], k) for j in range(8)])
                if do_flip:
                    skel = skel[:, ::-1]
                    width = width[:, ::-1]
                    aff = np.stack([aff[(4 - j) % 8][:, ::-1] for j in range(8)])
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)
        sample = {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
        }
        if self.with_topo:
            if not cached:
                t = make_targets(mask)
                skel, width, aff = t["skel"], t["width"], t["aff"]
            sample["skel"] = torch.from_numpy(np.ascontiguousarray(skel)[None].astype(np.float32))
            sample["width"] = torch.from_numpy(np.ascontiguousarray(width)[None].astype(np.float32))
            sample["aff"] = torch.from_numpy(np.ascontiguousarray(aff).astype(np.float32))
        return sample


class SyntheticLISSRoads(Dataset):
    def __init__(self, n=256, size=256, seed=0, with_topo_targets=True):
        self.n, self.size, self.seed = n, size, seed
        self.with_topo = with_topo_targets

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 100003 + i)
        skel_mask = _random_road_network(self.size, rng)
        width_px = rng.integers(1, 4)
        mask = binary_dilation(skel_mask, iterations=int(width_px) - 1) if width_px > 1 else skel_mask
        mask = mask.astype(np.uint8)
        img = _spectral_render(mask, self.size, rng)
        sample = {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
        }
        if self.with_topo:
            t = make_targets(mask)
            sample["skel"] = torch.from_numpy(t["skel"][None])
            sample["width"] = torch.from_numpy(t["width"][None])
            sample["aff"] = torch.from_numpy(t["aff"])
        return sample
