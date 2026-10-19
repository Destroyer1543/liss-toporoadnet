import json, pathlib, sys
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from evaluate_twotier import build_model
RUNS = pathlib.Path("runs"); OUT = pathlib.Path("data/rebuttal/preds_test_thr")
TILES = sorted(pathlib.Path("data/liss4_tiles/test").glob("*.npz"))
THR = [0.3, 0.4, 0.6, 0.7]
def main(gpu, runs, bs=16):
    OUT.mkdir(parents=True, exist_ok=True); dev = torch.device(f"cuda:{gpu}")
    imgs = np.stack([np.load(t)["image"].astype(np.float32) for t in TILES]); print("loaded", flush=True)
    for rn in runs:
        cfg = json.loads((RUNS / rn / "config.json").read_text())
        m = build_model(cfg.get("model", "toporoad"), dev, cfg)
        m.load_state_dict(torch.load(RUNS / rn / "best.pt", map_location=dev, weights_only=False)["model"]); m.eval()
        packs = {t: np.zeros((len(imgs), 256, 32), np.uint8) for t in THR}
        with torch.no_grad():
            for i in range(0, len(imgs), bs):
                out = m(torch.from_numpy(imgs[i:i + bs]).to(dev))
                p = torch.sigmoid(out["fused_logit"] if "fused_logit" in out else out["mask_logit"])[:, 0].cpu().numpy()
                for t in THR:
                    packs[t][i:i + bs] = np.packbits(p >= t, axis=-1)
        for t in THR:
            np.save(OUT / f"{rn}__t{t}.npy", packs[t])
        print(rn, "done", flush=True); del m; torch.cuda.empty_cache()
if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2].split(","))
