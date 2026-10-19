import json, pathlib, sys, time
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from evaluate_twotier import build_model

RUNS = pathlib.Path("runs")
OUT = pathlib.Path("data/rebuttal/preds_test")
TILES = sorted(pathlib.Path("data/liss4_tiles/test").glob("*.npz"))

def main(run_names, gpu, bs=16):
    dev = torch.device(f"cuda:{gpu}")
    names = [t.name for t in TILES]
    (OUT / "tiles.json").write_text(json.dumps(names))
    t0 = time.time()
    imgs = np.stack([np.load(t)["image"].astype(np.float16) for t in TILES])
    print(f"loaded {len(imgs)} tiles in {time.time()-t0:.0f}s", flush=True)
    for rn in run_names:
        cfg = json.loads((RUNS / rn / "config.json").read_text())
        zb = [int(b) for b in str(cfg.get("zero_bands") or "").split(",") if b != ""]
        variants = [("", zb)] + ([("__nozero", [])] if zb else [])
        model = build_model(cfg.get("model", "toporoad"), dev, cfg)
        ck = torch.load(RUNS / rn / "best.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); model.eval()
        for suffix, zero in variants:
            outp = OUT / f"{rn}{suffix}.npy"
            if outp.exists():
                print(f"{rn}{suffix}: exists, skip", flush=True); continue
            packed = np.zeros((len(imgs), 256, 32), np.uint8)
            with torch.no_grad():
                for i in range(0, len(imgs), bs):
                    x = torch.from_numpy(imgs[i:i+bs].astype(np.float32)).to(dev)
                    for b in zero:
                        x[:, b] = 0.0
                    out = model(x)
                    logit = out["fused_logit"] if "fused_logit" in out else out["mask_logit"]
                    pr = (torch.sigmoid(logit)[:, 0] >= 0.5).cpu().numpy()
                    packed[i:i+bs] = np.packbits(pr, axis=-1)
            np.save(outp, packed)
            print(f"{rn}{suffix}: done (ckpt epoch {ck.get('epoch')}, zero={zero}) "
                  f"road-frac {np.unpackbits(packed,axis=-1).mean():.4f}  [{time.time()-t0:.0f}s]", flush=True)
        del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    gpu = int(sys.argv[1]); runs = sys.argv[2].split(",")
    main(runs, gpu)
