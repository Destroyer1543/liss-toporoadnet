import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import NpzTileDataset
from graph_metrics import graph_metrics_batch
from metrics import raster_metrics, cldice_metric


def build_model(name):
    if name == "toporoad":
        from model import LISSTopoRoadNet
        return LISSTopoRoadNet()
    from baselines import BASELINES
    return BASELINES[name]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=None, help="default: read from run config.json")
    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--graph-items", type=int, default=200, help="tiles for graph metrics (slow)")
    args = ap.parse_args()

    ckpt_path = pathlib.Path(args.ckpt)
    model_name = args.model
    cfg_p = ckpt_path.parent / "config.json"
    if model_name is None:
        model_name = json.loads(cfg_p.read_text()).get("model", "toporoad") if cfg_p.exists() else "toporoad"

    device = torch.device(f"cuda:{args.gpu}")
    model = build_model(model_name).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    ds = NpzTileDataset(pathlib.Path(args.data_root) / args.split, size=args.tile,
                        augment=False, with_topo_targets=False)
    dl = DataLoader(ds, batch_size=64, num_workers=8)
    probs, masks = [], []
    with torch.no_grad():
        for batch in dl:
            out = model(batch["image"].to(device))
            probs.append(torch.sigmoid(out["fused_logit"]).float().cpu().numpy())
            masks.append(batch["mask"].numpy())
    probs = np.concatenate(probs)
    masks = np.concatenate(masks)

    m = raster_metrics(probs, masks)
    m["cldice"] = cldice_metric(probs, masks)
    m.update(graph_metrics_batch(probs, masks, max_items=args.graph_items))
    m = {k: round(float(v), 4) for k, v in m.items()}
    print(json.dumps({"ckpt": str(ckpt_path), "model": model_name,
                      "split": args.split, "n_tiles": len(ds), **m}, indent=2))


if __name__ == "__main__":
    main()
