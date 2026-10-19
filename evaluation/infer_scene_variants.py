import json, os, pathlib, sys, time
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from infer_scene_mosaic import load_scene, build_model, RUNS, OUT, T


def variants(out, is_ours):
    if is_ours:
        f = torch.sigmoid(out["fused_logit"])[:, 0]
        m = torch.sigmoid(out["mask_logit"])[:, 0]
        if "skel_logit" not in out:
            return {"F05": f >= 0.5, "M05": m >= 0.5}
        s = torch.sigmoid(out["skel_logit"])[:, 0]
        return {"F05": f >= 0.5, "M05": m >= 0.5, "S03": s >= 0.3, "S05": s >= 0.5, "S07": s >= 0.7,
                "FS": (f >= 0.5) & (s >= 0.3)}
    p = torch.sigmoid(out["fused_logit"] if "fused_logit" in out else out["mask_logit"])[:, 0]
    return {"T03": p >= 0.3, "T05": p >= 0.5, "T07": p >= 0.7}


def main(gpu, scene, runs, bs=8):
    dev = torch.device(f"cuda:{gpu}")
    od = OUT / scene; od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    img, nodata, tf, crs = load_scene(scene)
    H, W = nodata.shape
    if not (od / "meta.npz").exists():
        np.savez_compressed(od / "meta.npz", transform=np.array(tf)[:6], crs=crs.to_wkt(),
                            shape=np.array([H, W]), nodata=np.packbits(nodata, axis=-1))
    coords = [(y, x) for y in range(0, H, T) for x in range(0, W, T)]
    print(f"{scene}: {H}x{W} loaded [{time.time()-t0:.0f}s]", flush=True)
    for rn in runs:
        cfg = json.loads((RUNS / rn / "config.json").read_text())
        is_ours = cfg.get("model", "toporoad") == "toporoad"
        model = build_model(cfg.get("model", "toporoad"), dev, cfg)
        model.load_state_dict(torch.load(RUNS / rn / os.environ.get("REB_CKPT", "best.pt"), map_location=dev, weights_only=False)["model"])
        model.eval()
        mos = None
        with torch.no_grad():
            for i in range(0, len(coords), bs):
                cc = coords[i:i + bs]
                x = torch.from_numpy(np.stack([img[:, y:y + T, x0:x0 + T] for y, x0 in cc]).astype(np.float32)).to(dev)
                v = {k: a.cpu().numpy() for k, a in variants(model(x), is_ours).items()}
                if mos is None:
                    mos = {k: np.zeros((H, W), bool) for k in v}
                for k, a in v.items():
                    for j, (y, x0) in enumerate(cc):
                        mos[k][y:y + T, x0:x0 + T] = a[j]
        for k, m in mos.items():
            m &= ~nodata
            np.savez_compressed(od / f"{rn}__{k}.npz", mask=np.packbits(m, axis=-1))
            print(f"  {rn}__{k}: road-frac {m[~nodata].mean():.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    print(f"done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3].split(","))
