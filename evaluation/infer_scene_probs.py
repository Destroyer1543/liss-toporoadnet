import json, os, pathlib, sys, time
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent)); sys.path.insert(0, "src")
from infer_scene_mosaic import load_scene, build_model, RUNS, OUT, T
from model import reconstruct_mask


def main(gpu, scene, runs, bs=8):
    dev = torch.device(f"cuda:{gpu}")
    od = OUT / scene; od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    img, nodata, tf, crs = load_scene(scene)
    H, W = nodata.shape
    coords = [(y, x) for y in range(0, H, T) for x in range(0, W, T)]
    print(f"{scene}: loaded [{time.time()-t0:.0f}s]", flush=True)
    for rn in runs:
        cfg = json.loads((RUNS / rn / "config.json").read_text())
        ours = cfg.get("model", "toporoad") == "toporoad"
        model = build_model(cfg.get("model", "toporoad"), dev, cfg)
        model.load_state_dict(torch.load(RUNS / rn / os.environ.get("REB_CKPT", "best.pt"), map_location=dev, weights_only=False)["model"])
        model.eval()
        if ours:
            maps = {k: np.zeros((H, W), np.uint8) for k in ("P_mask", "P_skel")}
            bins = {k: np.zeros((H, W), bool) for k in ("FH", "FW1", "FWC")}
        else:
            maps = {"P_out": np.zeros((H, W), np.uint8)}
            bins = {k: np.zeros((H, W), bool) for k in ("T03", "T05", "T07")} if cfg.get("model") == "roadformer" else {}
        with torch.no_grad():
            for i in range(0, len(coords), bs):
                cc = coords[i:i + bs]
                x = torch.from_numpy(np.stack([img[:, y:y + T, x0:x0 + T] for y, x0 in cc]).astype(np.float32)).to(dev)
                out = model(x)
                if ours:
                    a, b = model.alpha, model.beta
                    pm, ps = torch.sigmoid(out["mask_logit"]), torch.sigmoid(out["skel_logit"])
                    lg = lambda m: torch.logit(m.clamp(1e-4, 1 - 1e-4))
                    rec = reconstruct_mask(ps, out["width"], k=model.recon_k)
                    r1 = reconstruct_mask(ps, torch.ones_like(out["width"]), k=model.recon_k)
                    rc = reconstruct_mask(ps, out["width"].clamp(0.5, 1.5), k=model.recon_k)
                    v = {"FH": a * out["mask_logit"] + 0.5 * b * lg(rec) >= 0,
                         "FW1": a * out["mask_logit"] + b * lg(r1) >= 0,
                         "FWC": a * out["mask_logit"] + b * lg(rc) >= 0}
                    p = {"P_mask": pm, "P_skel": ps}
                else:
                    po = torch.sigmoid(out["fused_logit"] if "fused_logit" in out else out["mask_logit"])
                    p = {"P_out": po}
                    v = {k: po >= float(k[1:]) / 10 for k in bins}
                p = {k: (t[:, 0].float().cpu().numpy() * 255).round().astype(np.uint8) for k, t in p.items()}
                v = {k: t[:, 0].cpu().numpy() for k, t in v.items()}
                for j, (y, x0) in enumerate(cc):
                    for k in maps:
                        maps[k][y:y + T, x0:x0 + T] = p[k][j]
                    for k in bins:
                        bins[k][y:y + T, x0:x0 + T] = v[k][j]
        for k, m in maps.items():
            m[nodata] = 0
            np.savez_compressed(od / f"{rn}__{k}.npz", prob=m)
        for k, m in bins.items():
            m &= ~nodata
            np.savez_compressed(od / f"{rn}__{k}.npz", mask=np.packbits(m, axis=-1))
            print(f"  {rn}__{k}: road-frac {m[~nodata].mean():.4f}", flush=True)
        print(f"  {rn}: probs saved [{time.time()-t0:.0f}s]", flush=True)
        del model; torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3].split(","))
