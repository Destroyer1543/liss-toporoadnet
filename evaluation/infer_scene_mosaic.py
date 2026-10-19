import json, os, pathlib, sys, time, zipfile
import numpy as np, rasterio, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from evaluate_twotier import build_model

RUNS = pathlib.Path("runs")
OUT = pathlib.Path("data/rebuttal/mosaics")
ZIPDIRS = [pathlib.Path("data/liss4_raw"), pathlib.Path("data/sih_raw")]
T = 256


def normalize(b):
    lo, hi = np.percentile(b[b > 0], [2, 98]) if (b > 0).any() else (0, 1)
    return np.clip((b.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)


CACHE = pathlib.Path("data/rebuttal/scene_cache")


def load_scene(scene):
    ci, cn, cm = CACHE / f"{scene}_img.npy", CACHE / f"{scene}_nodata.npy", CACHE / f"{scene}_meta.json"
    if ci.exists() and cn.exists() and cm.exists():
        import rasterio.crs
        meta = json.loads(cm.read_text())
        return (np.load(ci, mmap_mode="r"), np.load(cn), rasterio.Affine(*meta["tf"]),
                rasterio.crs.CRS.from_wkt(meta["crs"]))
    img, nodata, tf, crs = _load_scene_raw(scene)
    if os.environ.get("REB_SCENE_CACHE") != "1":
        return img, nodata, tf, crs
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(ci, img); np.save(cn, nodata)
    cm.write_text(json.dumps({"tf": list(tf)[:6], "crs": crs.to_wkt()}))
    return img, nodata, tf, crs


def _load_scene_raw(scene):
    zp = next(d / f"{scene}.zip" for d in ZIPDIRS if (d / f"{scene}.zip").exists())
    names = zipfile.ZipFile(zp).namelist()
    arrs = []
    for k in ("BAND2", "BAND3", "BAND4"):
        inner = [n for n in names if n.upper().endswith(f"{k}.TIF")][0]
        with rasterio.open(f"/vsizip/{zp}/{inner}") as s:
            arrs.append(s.read(1)); tf, crs = s.transform, s.crs
    raw = np.stack(arrs)
    H, W = (raw.shape[1] // T) * T, (raw.shape[2] // T) * T
    nodata = (raw[:, :H, :W].sum(0) == 0)
    img = np.stack([normalize(a) for a in arrs])[:, :H, :W].astype(np.float16)
    return img, nodata, tf, crs


def main(gpu, scenes, runs, bs=8):
    dev = torch.device(f"cuda:{gpu}")
    for scene in scenes:
        od = OUT / scene; od.mkdir(parents=True, exist_ok=True)
        todo = [r for r in runs if not (od / f"{r}.npz").exists()]
        if not todo:
            print(f"{scene}: all done", flush=True); continue
        t0 = time.time()
        img, nodata, tf, crs = load_scene(scene)
        H, W = nodata.shape
        if not (od / "meta.npz").exists():
            np.savez_compressed(od / "meta.npz", transform=np.array(tf)[:6], crs=crs.to_wkt(),
                                shape=np.array([H, W]), nodata=np.packbits(nodata, axis=-1))
        coords = [(y, x) for y in range(0, H, T) for x in range(0, W, T)]
        print(f"{scene}: {H}x{W}, {len(coords)} tiles, loaded in {time.time()-t0:.0f}s", flush=True)
        for rn in todo:
            cfg = json.loads((RUNS / rn / "config.json").read_text())
            zb = [int(b) for b in str(cfg.get("zero_bands") or "").split(",") if b != ""]
            model = build_model(cfg.get("model", "toporoad"), dev, cfg)
            ck = torch.load(RUNS / rn / "best.pt", map_location=dev, weights_only=False)
            model.load_state_dict(ck["model"]); model.eval()
            mos = np.zeros((H, W), bool)
            with torch.no_grad():
                for i in range(0, len(coords), bs):
                    cc = coords[i:i + bs]
                    x = torch.from_numpy(np.stack([img[:, y:y + T, x0:x0 + T] for y, x0 in cc]).astype(np.float32)).to(dev)
                    for b in zb:
                        x[:, b] = 0.0
                    out = model(x)
                    logit = out["fused_logit"] if "fused_logit" in out else out["mask_logit"]
                    pr = (torch.sigmoid(logit)[:, 0] >= 0.5).cpu().numpy()
                    for k, (y, x0) in enumerate(cc):
                        mos[y:y + T, x0:x0 + T] = pr[k]
            mos &= ~nodata
            np.savez_compressed(od / f"{rn}.npz", mask=np.packbits(mos, axis=-1))
            print(f"  {rn}: road-frac {mos[~nodata].mean():.4f}  [{time.time()-t0:.0f}s]", flush=True)
            del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2].split(","), sys.argv[3].split(","))
