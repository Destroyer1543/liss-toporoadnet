import pathlib
import sys
import zipfile

import numpy as np
import rasterio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from liss4_prepare import OSM_WIDTHS, fetch_osm_roads, rasterize_roads

RAW = pathlib.Path("data/liss4_raw")
OUT = pathlib.Path("data/eval_masks")
TEST_SCENES = ["R2F24AUG2023064073010100066SSANSTUC00GTDD",
               "RAF23OCT2023035685010100064SSANSTUC00GTDA"]
ALL_SCENES = sorted(p.stem for p in RAW.glob("*.zip"))
MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary",
         "motorway_link", "trunk_link", "primary_link", "secondary_link",
         "unclassified", "residential"}
HIGHWAY = {"motorway", "trunk", "primary", "secondary",
           "motorway_link", "trunk_link", "primary_link", "secondary_link"}


def band2_vsizip(scene):
    zp = RAW / f"{scene}.zip"
    inner = [n for n in zipfile.ZipFile(zp).namelist() if n.upper().endswith("BAND2.TIF")][0]
    return f"/vsizip/{zp}/{inner}"


def main():
    import sys as _sys
    scenes = ALL_SCENES if "--all" in _sys.argv else TEST_SCENES
    highway_only = "--highway-only" in _sys.argv
    keep = HIGHWAY if highway_only else MAJOR
    out_dir = (RAW.parent / "highway_masks") if highway_only else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        out_p = out_dir / f"{scene}.npz"
        if out_p.exists():
            print(f"{scene}: exists, skip"); continue
        with rasterio.open(band2_vsizip(scene)) as src:
            transform, crs, bnds, shape = src.transform, src.crs, src.bounds, (src.height, src.width)
        from pyproj import Transformer
        t2 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        w_lon, s_lat = t2.transform(bnds.left, bnds.bottom)
        e_lon, n_lat = t2.transform(bnds.right, bnds.top)
        elements = fetch_osm_roads((s_lat, w_lon, n_lat, e_lon))
        major_el = [e for e in elements if e.get("tags", {}).get("highway") in keep]
        minor_el = [e for e in elements
                    if e.get("tags", {}).get("highway") in OSM_WIDTHS
                    and e.get("tags", {}).get("highway") not in keep]
        major = rasterize_roads(major_el, transform, crs, shape)
        minor = rasterize_roads(minor_el, transform, crs, shape)
        ignore = (minor & ~major).astype(np.uint8)
        np.savez_compressed(out_p, major=major.astype(np.uint8), ignore=ignore)
        print(f"{scene}: {len(major_el)} major ways (frac {major.mean():.4f}), "
              f"ignore frac {ignore.mean():.4f} -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
