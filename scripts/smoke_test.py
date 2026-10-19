import argparse
import sys
import time

sys.path.insert(0, "src")

import numpy as np
import torch

from data import SyntheticLISSRoads
from losses import TopoRoadLoss
from metrics import raster_metrics, cldice_metric
from model import LISSTopoRoadNet, count_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    ds = SyntheticLISSRoads(n=4, size=192, seed=42)
    batch = {k: torch.stack([ds[i][k] for i in range(4)]).to(device) for k in ds[0]}
    road_frac = batch["mask"].mean().item()
    print(f"synthetic batch: image {tuple(batch['image'].shape)}, road fraction {road_frac:.4f}")
    assert 0.001 < road_frac < 0.2, "road fraction implausible"
    assert batch["aff"].shape[1] == 8

    model = LISSTopoRoadNet().to(device)
    print(f"model params: {count_params(model)/1e6:.2f}M")
    out = model(batch["image"])
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
        expect = tuple(n // 4 for n in batch["image"].shape[-2:]) if k == "aux_logit" else tuple(batch["image"].shape[-2:])
        assert tuple(v.shape[-2:]) == expect, f"{k} wrong spatial size"

    criterion = TopoRoadLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t0 = time.time()
    first_loss = None
    for step in range(args.steps):
        stage = "warmup" if step < 30 else "full"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch["image"])
            loss, parts = criterion(out, batch, stage=stage)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        if step % 25 == 0 or step == args.steps - 1:
            print(f"step {step:4d} [{stage}] " +
                  " ".join(f"{k}={v:.4f}" for k, v in parts.items()), flush=True)
    print(f"overfit time: {time.time()-t0:.1f}s, "
          f"mem: {torch.cuda.max_memory_allocated(device)/2**30:.2f} GiB")

    with torch.no_grad():
        prob = torch.sigmoid(model(batch["image"])["fused_logit"]).float().cpu().numpy()
    m = raster_metrics(prob, batch["mask"].cpu().numpy())
    m["cldice"] = cldice_metric(prob, batch["mask"].cpu().numpy())
    print("overfit metrics:", {k: round(v, 4) for k, v in m.items()})
    assert loss.item() < first_loss, "loss did not decrease"
    assert m["iou"] > 0.3, f"overfit IoU too low ({m['iou']:.3f}) — pipeline may be broken"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
