import argparse
import json
import pathlib
import sys
import time
import urllib.request
import zipfile

import numpy as np

OVERPASS = "https://overpass-api.de/api/interpreter"
OSM_WIDTHS = {
    "motorway": 20, "trunk": 16, "primary": 12, "secondary": 10,
    "tertiary": 8, "unclassified": 6, "residential": 6, "service": 5,
    "motorway_link": 12, "trunk_link": 10, "primary_link": 8, "secondary_link": 8,
    "track": 4,
}


def find_band_files(scene_dir):
    tifs = sorted(scene_dir.rglob("*.tif")) + sorted(scene_dir.rglob("*.TIF"))
    bands = {}
    for t in tifs:
        up = t.name.upper()
        for key, name in (("BAND2", "G"), ("BAND3", "R"), ("BAND4", "N")):
            if key in up:
                bands[name] = t
    if len(bands) == 3:
        return bands, None
    for t in tifs:
        import rasterio
        with rasterio.open(t) as src:
            if src.count >= 3:
                return None, t
    return None, None


OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def fetch_osm_roads(bounds_wgs84, retries=4):
    import urllib.parse
    s, w, n, e = bounds_wgs84
    q = f'[out:json][timeout:180];way["highway"]({s},{w},{n},{e});out geom;'
    body = urllib.parse.urlencode({"data": q}).encode()
    for attempt in range(retries):
        url = OVERPASS_MIRRORS[attempt % len(OVERPASS_MIRRORS)]
        try:
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "LISS-TopoRoadNet-research/1.0 (academic road-extraction study)",
            })
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read()).get("elements", [])
        except Exception as exc:
            print(f"  overpass attempt {attempt+1} ({url.split('/')[2]}) failed: {exc}", flush=True)
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("Overpass failed after retries")


def rasterize_roads(elements, transform, crs, shape):
    from rasterio.features import rasterize
    from pyproj import Transformer
    from shapely.geometry import LineString
    tfm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    px_size = abs(transform.a)
    shapes = []
    for el in elements:
        hw = el.get("tags", {}).get("highway", "")
        if hw not in OSM_WIDTHS or "geometry" not in el:
            continue
        pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(pts) < 2:
            continue
        xs, ys = tfm.transform(*zip(*pts))
        line = LineString(zip(xs, ys))
        half_w = max(OSM_WIDTHS[hw] / 2.0, px_size * 0.55)
        shapes.append(line.buffer(half_w))
    if not shapes:
        return np.zeros(shape, dtype=np.uint8)
    return rasterize(((g, 1) for g in shapes), out_shape=shape,
                     transform=transform, fill=0, dtype="uint8")


def normalize(band):
    lo, hi = np.percentile(band[band > 0], [2, 98]) if (band > 0).any() else (0, 1)
    return np.clip((band.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)


def process_scene(scene_dir, out_dir, tile=256, do_osm=True, keep_empty=0.1):
    import rasterio
    bands, stack = find_band_files(scene_dir)
    if bands is None and stack is None:
        print(f"  !! no usable TIFFs in {scene_dir.name}, skipping"); return 0
    if bands:
        arrs, ref = [], None
        for name in ("G", "R", "N"):
            with rasterio.open(bands[name]) as src:
                arrs.append(src.read(1))
                ref = ref or (src.transform, src.crs, src.bounds)
        transform, crs, bnds = ref
    else:
        with rasterio.open(stack) as src:
            data = src.read()
            arrs = [data[0], data[1], data[2]]
            transform, crs, bnds = src.transform, src.crs, src.bounds
    img = np.stack([normalize(a) for a in arrs])
    h, w = img.shape[1:]

    mask = None
    if do_osm:
        from pyproj import Transformer
        t2 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        (w_lon, s_lat) = t2.transform(bnds.left, bnds.bottom)
        (e_lon, n_lat) = t2.transform(bnds.right, bnds.top)
        elements = fetch_osm_roads((s_lat, w_lon, n_lat, e_lon))
        mask = rasterize_roads(elements, transform, crs, (h, w))
        print(f"  OSM: {len(elements)} ways, road fraction {mask.mean():.4f}", flush=True)

    rng = np.random.default_rng(0)
    n_saved = 0
    sdir = out_dir / scene_dir.name
    sdir.mkdir(parents=True, exist_ok=True)
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            chip = img[:, y:y + tile, x:x + tile]
            if (chip.sum(0) == 0).mean() > 0.3:
                continue
            m = mask[y:y + tile, x:x + tile] if mask is not None else None
            if m is not None and m.mean() < 0.001 and rng.random() > keep_empty:
                continue
            out = {"image": chip.astype(np.float32)}
            if m is not None:
                out["mask"] = m
            np.savez_compressed(sdir / f"y{y}_x{x}.npz", **out)
            n_saved += 1
    print(f"  {scene_dir.name}: {n_saved} tiles", flush=True)
    return n_saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/liss4_raw")
    ap.add_argument("--out", default="data/liss4_tiles")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--no-osm", action="store_true")
    args = ap.parse_args()
    raw = pathlib.Path(args.raw)
    out = pathlib.Path(args.out)
    for z in raw.glob("*.zip"):
        mark = raw / (z.stem + ".extracted")
        if not mark.exists():
            print(f"extracting {z.name}", flush=True)
            zipfile.ZipFile(z).extractall(raw / z.stem)
            mark.touch()
    scenes = [d for d in raw.iterdir() if d.is_dir()]
    if not scenes:
        sys.exit(f"no scene dirs in {raw}")
    total = 0
    for s in scenes:
        if (out / s.name).exists() and any((out / s.name).glob("*.npz")):
            print(f"scene {s.name}: already processed, skipping", flush=True)
            continue
        print(f"scene {s.name}", flush=True)
        total += process_scene(s, out, args.tile, do_osm=not args.no_osm)
    print(f"TOTAL {total} tiles -> {out}")


if __name__ == "__main__":
    main()
