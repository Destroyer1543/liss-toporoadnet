import json, os, pathlib, sys, time
import numpy as np, torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from evaluate_twotier import build_model

RUNS = pathlib.Path("runs")
OUT = pathlib.Path("data/rebuttal/preds_test32")
TILES = sorted(pathlib.Path("data/liss4_tiles/test").glob("*.npz"))
OURS = ["exp15_toporoad_full", "exp22_toporoad_seed1", "exp24_toporoad_seed2"]


def main(gpu, runs, bs=8):
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(f"cuda:{gpu}")
    t0 = time.time()
    imgs = np.stack([np.load(t)["image"].astype(np.float32) for t in TILES])
    masks = np.stack([np.load(t)["mask"] for t in TILES]).astype(bool)
    print(f"loaded {len(imgs)} tiles fp32 [{time.time()-t0:.0f}s]", flush=True)
    width_stats = {}
    for rn in runs:
        cfg = json.loads((RUNS / rn / "config.json").read_text())
        zb = [int(b) for b in str(cfg.get("zero_bands") or "").split(",") if b != ""]
        model = build_model(cfg.get("model", "toporoad"), dev, cfg)
        ck = torch.load(RUNS / rn / os.environ.get("REB_CKPT", "best.pt"), map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); model.eval()
        packed = np.zeros((len(imgs), 256, 32), np.uint8)
        errs, tgts = [], []
        with torch.no_grad():
            for i in range(0, len(imgs), bs):
                x = torch.from_numpy(imgs[i:i + bs]).to(dev)
                for b in zb:
                    x[:, b] = 0.0
                out = model(x)
                logit = out["fused_logit"] if "fused_logit" in out else out["mask_logit"]
                packed[i:i + bs] = np.packbits((torch.sigmoid(logit)[:, 0] >= 0.5).cpu().numpy(), axis=-1)
                if rn in OURS and "width" in out:
                    w = out["width"][:, 0].float().cpu().numpy()
                    for k in range(len(w)):
                        m = masks[i + k]
                        if m.sum() < 10:
                            continue
                        sk = skeletonize(m)
                        t = distance_transform_edt(m)[sk]
                        errs.append(w[k][sk] - t); tgts.append(t)
        np.save(OUT / f"{rn}.npy", packed)
        if errs:
            e, t = np.concatenate(errs), np.concatenate(tgts)
            st = {"n_px": int(len(e)), "mae_px": float(np.abs(e).mean()), "mae_m": float(np.abs(e).mean() * 5.8),
                  "bias_px": float(e.mean()), "target_mean_px": float(t.mean())}
            for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 99)):
                sel = (t >= lo) & (t < hi)
                if sel.any():
                    st[f"mae_px_t{lo}-{hi}"] = float(np.abs(e[sel]).mean()); st[f"bias_px_t{lo}-{hi}"] = float(e[sel].mean())
                    st[f"n_t{lo}-{hi}"] = int(sel.sum())
            width_stats[rn] = st
            print(f"  width {rn}: {st}", flush=True)
        print(f"{rn}: done [{time.time()-t0:.0f}s]", flush=True)
        del model; torch.cuda.empty_cache()
    if width_stats:
        pathlib.Path("data/rebuttal/width_head.json").write_text(json.dumps(width_stats, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2].split(","))
