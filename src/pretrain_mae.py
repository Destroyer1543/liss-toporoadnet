"""MAE-style self-supervised pretraining for the LISS-TopoRoadNet encoder.

Masks random 16x16 patches of the G/R/NIR input (mask ratio 0.5) and trains
stem+encoder+light decoder to reconstruct the masked pixels (MSE on masked
area only). Pretrains on TRAIN-scene tiles only (no test-scene leakage).
Saves {"stem": ..., "encoder": ...} state dicts for train.py --init.

Usage:
  python pretrain_mae.py --out ../runs/exp27_mae_pretrain --gpu 1 --epochs 40
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))

from data import NpzTileDataset
from model import ConvBNAct, DualPathEncoder, SpectralStem


class MAENet(nn.Module):
    def __init__(self, stem_ch=32, c4=64):
        super().__init__()
        self.stem = SpectralStem(stem_ch)
        self.encoder = DualPathEncoder(stem_ch)
        self.dec = nn.Sequential(
            ConvBNAct(c4, c4), nn.Conv2d(c4, 3 * 16, 1), nn.PixelShuffle(4)
        )  # f4 (1/4 scale) -> 3ch full res

    def forward(self, x):
        f4, _, _ = self.encoder(self.stem(x))
        return self.dec(f4)


def random_patch_mask(b, h, w, patch=16, ratio=0.5, device="cuda"):
    gh, gw = h // patch, w // patch
    keep = (torch.rand(b, 1, gh, gw, device=device) > ratio).float()
    return F.interpolate(keep, scale_factor=patch, mode="nearest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0)

    print("preloading train tiles...", flush=True)
    ds = NpzTileDataset("/data/road_liss4/liss4_tiles/train", size=256,
                        augment=True, with_topo_targets=False, preload=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, drop_last=True)
    model = MAENet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))

    start_epoch = 0
    resume = os.path.join(args.out, "mae_last.pt")
    if os.path.exists(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start_epoch = ck["epoch"] + 1
        print(f"resumed at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time(); tot = 0.0; n = 0
        for batch in dl:
            x = batch["image"].to(device, non_blocking=True)
            keep = random_patch_mask(x.shape[0], x.shape[2], x.shape[3],
                                     ratio=args.mask_ratio, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec = model(x * keep)
            rec = rec.float()
            masked = 1 - keep
            loss = ((rec - x) ** 2 * masked).sum() / (masked.sum() * 3 + 1e-6)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if torch.isfinite(gn):
                opt.step()
            sched.step()
            tot += loss.item(); n += 1
        print(f"ep {epoch:03d} mae_loss {tot/max(n,1):.5f} ({time.time()-t0:.0f}s)", flush=True)
        tmp = resume + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch}, tmp)
        os.replace(tmp, resume)
    torch.save({"stem": model.stem.state_dict(), "encoder": model.encoder.state_dict()},
               os.path.join(args.out, "mae_encoder.pt"))
    print("DONE -> mae_encoder.pt", flush=True)


if __name__ == "__main__":
    main()
