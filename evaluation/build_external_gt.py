import pathlib, sys, time, zipfile
import numpy as np, rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString
from shapely.ops import transform as shp_tf
sys.path.insert(0, "scripts")
from liss4_prepare import OSM_WIDTHS
from major_masks import MAJOR

import os
PBF = "data/osm/north-eastern-zone.osm.pbf"
PBFS = {"R2F24AUG2023064073010100066SSANSTUC00GTDD": ["data/osm/tamil_nadu-latest.osm.pbf"],
        "RAF23OCT2023035685010100064SSANSTUC00GTDA": ["data/osm/tamil_nadu-latest.osm.pbf",
                                                      "data/osm/andhra_pradesh-latest.osm.pbf"]}
OUT = pathlib.Path(os.environ.get("REB_GT_OUT", "data/rebuttal/external"))
SCENES = ["R2F13DEC2025076029011100053SSANSTUC00GTDA", "R2F25JAN2026076640011000053SSANSTUC00GTDD",
          "RAF18JAN2026047300011100053SSANSTUC00GTDC", "RAF20DEC2025046888011000052SSANSTUC00GTDD"]
CLASSES = sorted(OSM_WIDTHS, key=lambda c: -OSM_WIDTHS[c])
PAVED = {"paved", "asphalt", "concrete", "concrete:plates", "paving_stones", "sett", "metal"}
UNPAVED = {"unpaved", "dirt", "earth", "ground", "gravel", "fine_gravel", "compacted", "mud",
           "sand", "grass", "pebblestone", "soil"}
T = 256


def meta(scene):
    zp = next(p for p in (pathlib.Path(f"data/sih_raw/{scene}.zip"),
                          pathlib.Path(f"data/liss4_raw/{scene}.zip")) if p.exists())
    inner = [n for n in zipfile.ZipFile(zp).namelist() if n.upper().endswith("BAND2.TIF")][0]
    with rasterio.open(f"/vsizip/{zp}/{inner}") as s:
        H, W = (s.height // T) * T, (s.width // T) * T
        return s.transform, s.crs, H, W, s.bounds


def main(scenes=SCENES):
    from pyrosm import OSM
    OUT.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        if (OUT / f"{scene}.npz").exists():
            print(scene, "exists"); continue
        t0 = time.time()
        tf, crs, H, W, b = meta(scene)
        w, s, e, n = transform_bounds(crs, "EPSG:4326", *b)
        import pandas as pd
        parts_gdf = []
        for pbf in PBFS.get(scene, [PBF]):
            g = OSM(pbf, bounding_box=[w, s, e, n]).get_data_by_custom_criteria(
                custom_filter={"highway": list(OSM_WIDTHS)}, filter_type="keep", keep_nodes=False,
                keep_ways=True, keep_relations=False)
            if g is not None and len(g):
                parts_gdf.append(g)
        gdf = pd.concat(parts_gdf, ignore_index=True)
        if "id" in gdf.columns:
            gdf = gdf.drop_duplicates(subset="id")
        gdf = gdf[gdf["highway"].isin(OSM_WIDTHS)]
        to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
        px = abs(tf.a)
        layers = {k: [] for k in ("t1", "major", "minor", "t1_cl", "major_cl")}
        cls_shapes, surf_shapes = [], []
        has_surf = "surface" in gdf.columns
        for _, r in gdf.iterrows():
            geom = r.geometry
            parts = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
            hw = r["highway"]; half = max(OSM_WIDTHS[hw] / 2.0, px * 0.55)
            sv = str(r["surface"]) if has_surf and r["surface"] is not None else ""
            sc = 1 if sv in PAVED else 2 if sv in UNPAVED else 0
            for p in parts:
                if not isinstance(p, LineString) or len(p.coords) < 2:
                    continue
                pu = shp_tf(to_utm, p); buf = pu.buffer(half)
                layers["t1"].append(buf); layers["t1_cl"].append(pu)
                if hw in MAJOR:
                    layers["major"].append(buf); layers["major_cl"].append(pu)
                else:
                    layers["minor"].append(buf)
                cls_shapes.append((buf, CLASSES.index(hw)))
                if sc:
                    surf_shapes.append((buf, sc))
        out = {}
        for k, shp in layers.items():
            m = rasterize(((g, 1) for g in shp), out_shape=(H, W), transform=tf, fill=0, dtype="uint8") \
                if shp else np.zeros((H, W), np.uint8)
            out[k] = m.astype(bool)
        cls = rasterize(sorted(cls_shapes, key=lambda t: -t[1]), out_shape=(H, W), transform=tf,
                        fill=255, dtype="uint8") if cls_shapes else np.full((H, W), 255, np.uint8)
        surf = rasterize(surf_shapes, out_shape=(H, W), transform=tf, fill=0, dtype="uint8") \
            if surf_shapes else np.zeros((H, W), np.uint8)
        ignore = out["minor"] & ~out["major"]
        pk = lambda a: np.packbits(a, axis=-1)
        np.savez_compressed(OUT / f"{scene}.npz", t1=pk(out["t1"]), major=pk(out["major"]),
                            ignore=pk(ignore), t1_cl=pk(out["t1_cl"]), major_cl=pk(out["major_cl"]),
                            cls=cls, surf=surf, shape=np.array([H, W]), classes=np.array(CLASSES))
        vc = gdf["highway"].value_counts()
        nsurf = int(gdf["surface"].notna().sum()) if has_surf else 0
        print(f"{scene}: {len(gdf)} ways ({', '.join(f'{k}:{v}' for k, v in vc.head(6).items())}); "
              f"t1 frac {out['t1'].mean():.4f} major {out['major'].mean():.4f} ignore {ignore.mean():.4f}; "
              f"surface-tagged ways {nsurf}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else SCENES)
