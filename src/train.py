import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))

from data import SyntheticLISSRoads
from losses import TopoRoadLoss
from metrics import raster_metrics, cldice_metric
from model import LISSTopoRoadNet, count_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gpu", type=int, default=None, help="CUDA device index (as seen by nvidia-smi)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--warmup-epochs", type=int, default=5, help="seg-loss-only epochs")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--data-root", default=None,
                   help="proxy/LISS tiles dir with train/ val/ subdirs of .npz; default: synthetic")
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", default="toporoad",
                   choices=["toporoad", "unet", "dlinknet", "roadformer", "roadformer_s2", "unet_ds"],
                   help="baselines train with the region loss only (their standard setup)")
    p.add_argument("--l-cldice", type=float, default=0.5)
    p.add_argument("--l-skel", type=float, default=1.0)
    p.add_argument("--l-width", type=float, default=0.2)
    p.add_argument("--l-aff", type=float, default=0.5)
    p.add_argument("--l-cons", type=float, default=0.3)
    p.add_argument("--skel-pos-weight", type=float, default=20.0)
    p.add_argument("--l-aux", type=float, default=0.4, help="deep supervision weight; 0 disables")
    p.add_argument("--l-blob", type=float, default=0.0, help="anti-blob edge-band FP penalty; 0 disables")
    p.add_argument("--major-masks-dir", default=None,
                   help="train on visible major-class roads only (dir of per-scene major masks)")
    p.add_argument("--transfer-init", default=None,
                   help="cross-resolution transfer: init full model from a proxy-trained checkpoint")
    p.add_argument("--holdout-scene", default=None,
                   help="leave-one-region-out: scene prefix to hold out for val/test; train on the rest")
    p.add_argument("--zero-bands", default=None,
                   help="band ablation: comma channel indices to zero (0=G,1=R,2=NIR)")
    p.add_argument("--init", default=None,
                   help="path to mae_encoder.pt: init stem+encoder from MAE pretraining")
    p.add_argument("--baseline-topo-loss", action="store_true",
                   help="rebuttal control: baselines also get soft-clDice after warmup "
                        "(their outputs have no skel/width/aff heads, so only clDice applies)")
    p.add_argument("--no-spectral-stem", action="store_true")
    p.add_argument("--no-topology-heads", action="store_true")
    p.add_argument("--drop-topology-heads", action="store_true",
                   help="true topology-head ablation: no centreline/width/affinity outputs or losses")
    p.add_argument("--no-reconstruction", action="store_true")
    p.add_argument("--tversky-alpha", type=float, default=0.7, help="focal-Tversky FN weight (region loss)")
    p.add_argument("--tversky-beta", type=float, default=0.3, help="focal-Tversky FP weight (region loss)")
    p.add_argument("--const-lr", action="store_true", help="constant LR (continuation fine-tunes) instead of cosine")
    p.add_argument("--accum", type=int, default=1,
                   help="gradient accumulation micro-batches per optimizer step (same effective batch, less memory)")
    p.add_argument("--c4", type=int, default=64, help="encoder width at 1/4 (track D capacity-matched: 90)")
    p.add_argument("--c8", type=int, default=128)
    p.add_argument("--c16", type=int, default=256)
    p.add_argument("--dec-ch", type=int, default=48)
    p.add_argument("--hr-refine", action="store_true", help="track C: full-resolution refinement ResBlock")
    p.add_argument("--learned-down2", action="store_true", help="track C: learned stride-2 stem skip")
    p.add_argument("--sep-heads", action="store_true", help="track C: separate mask / topology head branches")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, masks = [], []
    for batch in loader:
        x = batch["image"].to(device)
        out = model(x)
        probs.append(torch.sigmoid(out["fused_logit"]).float().cpu().numpy())
        masks.append(batch["mask"].numpy())
    probs = np.concatenate(probs)
    masks = np.concatenate(masks)
    m = raster_metrics(probs, masks)
    m["cldice"] = cldice_metric(probs, masks)
    model.train()
    return m


def main():
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")

    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.data_root:
        from data import NpzTileDataset
        import glob as _glob
        zb = [int(x) for x in args.zero_bands.split(",")] if args.zero_bands else None
        print("preloading tiles into RAM...", flush=True)
        if args.holdout_scene:
            holds = [h for h in args.holdout_scene.split(",") if h]
            allp = []
            for sub in ("train", "val", "test"):
                allp += _glob.glob(os.path.join(args.data_root, sub, "*.npz"))
            def _held(p):
                b = os.path.basename(p)
                return any(b.startswith(h) for h in holds)
            tr = [p for p in allp if not _held(p)]
            ho = [p for p in allp if _held(p)]
            if not ho:
                raise SystemExit(f"holdout '{args.holdout_scene}' matched 0 tiles")
            train_ds = NpzTileDataset(None, size=args.tile, augment=True, seed=args.seed,
                                      with_topo_targets=(args.model == "toporoad"),
                                      preload=True, paths=tr, zero_bands=zb)
            val_ds = NpzTileDataset(None, size=args.tile, augment=False, seed=args.seed,
                                    with_topo_targets=False, preload=True, paths=ho, zero_bands=zb)
            print(f"LORO holdout={args.holdout_scene}: {len(train_ds)} train / {len(val_ds)} heldout tiles", flush=True)
        else:
            train_ds = NpzTileDataset(os.path.join(args.data_root, "train"),
                                      size=args.tile, augment=True, seed=args.seed,
                                      with_topo_targets=(args.model == "toporoad"),
                                      preload=True, major_masks_dir=args.major_masks_dir, zero_bands=zb)
            val_ds = NpzTileDataset(os.path.join(args.data_root, "val"),
                                    size=args.tile, augment=False, seed=args.seed,
                                    with_topo_targets=False, preload=True,
                                    major_masks_dir=args.major_masks_dir, zero_bands=zb)
            print(f"data: {len(train_ds)} train / {len(val_ds)} val tiles from {args.data_root}", flush=True)
    else:
        train_ds = SyntheticLISSRoads(n=args.n_train, size=args.tile, seed=args.seed)
        val_ds = SyntheticLISSRoads(n=args.n_val, size=args.tile, seed=args.seed + 999)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=False, drop_last=True,
                          prefetch_factor=1 if args.workers > 0 else None)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        num_workers=min(2, args.workers))

    if args.model == "toporoad":
        model = LISSTopoRoadNet(
            c4=args.c4, c8=args.c8, c16=args.c16, dec_ch=args.dec_ch,
            use_spectral_stem=not args.no_spectral_stem,
            use_topology_heads=not args.no_topology_heads,
            use_reconstruction=not args.no_reconstruction,
            drop_topology_heads=args.drop_topology_heads,
            hr_refine=args.hr_refine, learned_down2=args.learned_down2, sep_heads=args.sep_heads,
        ).to(device)
    else:
        from baselines import BASELINES
        model = BASELINES[args.model]().to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        model.stem.load_state_dict(ck["stem"])
        model.encoder.load_state_dict(ck["encoder"])
        print(f"initialized stem+encoder from {args.init}", flush=True)
    if args.transfer_init:
        ck = torch.load(args.transfer_init, map_location=device, weights_only=False)
        sd = ck.get("model", ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"transfer-init full model from {args.transfer_init} "
              f"(missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
    print(f"params: {count_params(model)/1e6:.2f}M", flush=True)

    criterion = TopoRoadLoss(l_cldice=args.l_cldice, l_skel=args.l_skel,
                             l_width=args.l_width, l_aff=args.l_aff, l_cons=args.l_cons,
                             l_aux=args.l_aux, l_blob=args.l_blob,
                             skel_pos_weight=args.skel_pos_weight,
                             tv_alpha=args.tversky_alpha, tv_beta=args.tversky_beta)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = (torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0) if args.const_lr else
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl) // args.accum))

    start_epoch = 0
    best_cldice = -1.0
    resume_path = os.path.join(args.run_dir, "last.pt")
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        if "opt" in ck:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_epoch = ck["epoch"] + 1
            best_cldice = ck.get("best_cldice", -1.0)
            print(f"resumed from {resume_path} at epoch {start_epoch}", flush=True)
        else:
            print("last.pt has no optimizer state (pre-resume format); starting fresh", flush=True)

    log_path = os.path.join(args.run_dir, "log.csv")
    log_f = open(log_path, "a", newline="")
    logger = csv.writer(log_f)
    logger.writerow(["epoch", "stage", "loss", "iou", "f1", "cldice", "lr", "sec"])

    for epoch in range(start_epoch, args.epochs):
        if args.model != "toporoad" and not args.baseline_topo_loss:
            stage = "warmup"
        else:
            stage = "warmup" if epoch < args.warmup_epochs else "full"
        t0 = time.time()
        ep_loss, n_it = 0.0, 0
        for it, batch in enumerate(train_dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch["image"])
            out = {k: v.float() for k, v in out.items()}
            loss, parts = criterion(out, batch, stage=stage)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                print(f"WARN ep{epoch}: non-finite loss, batch skipped", flush=True)
                continue
            if it % args.accum == 0:
                opt.zero_grad(set_to_none=True)
            (loss / args.accum).backward()
            if (it + 1) % args.accum != 0:
                ep_loss += loss.item()
                n_it += 1
                continue
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gnorm):
                opt.zero_grad(set_to_none=True)
                print(f"WARN ep{epoch}: non-finite grad norm, step skipped", flush=True)
                continue
            opt.step()
            sched.step()
            ep_loss += loss.item()
            n_it += 1
        m = evaluate(model, val_dl, device)
        dt = time.time() - t0
        row = [epoch, stage, round(ep_loss / max(n_it, 1), 4),
               round(m["iou"], 4), round(m["f1"], 4), round(m["cldice"], 4),
               f"{sched.get_last_lr()[0]:.2e}", round(dt, 1)]
        logger.writerow(row)
        log_f.flush()
        print(f"ep {epoch:03d} [{stage}] loss {row[2]} iou {row[3]} f1 {row[4]} clDice {row[5]} ({dt:.0f}s)",
              flush=True)
        if m["cldice"] > best_cldice:
            best_cldice = m["cldice"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": m},
                       os.path.join(args.run_dir, "best.pt"))
        tmp = os.path.join(args.run_dir, "last.pt.tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch,
                    "best_cldice": best_cldice}, tmp)
        os.replace(tmp, os.path.join(args.run_dir, "last.pt"))
    log_f.close()


if __name__ == "__main__":
    main()
