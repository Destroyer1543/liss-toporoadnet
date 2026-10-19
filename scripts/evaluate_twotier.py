import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from graph_metrics import apls_raster, junction_recall
from metrics import cldice_metric, raster_metrics

TILES = pathlib.Path("data/liss4_tiles/test")
EVAL_MASKS = pathlib.Path("data/eval_masks")
RUNS = pathlib.Path(__file__).resolve().parent.parent / "runs"
EPS = 1e-6


def build_model(name, device, cfg=None):
    if name == "toporoad":
        from model import LISSTopoRoadNet
        cfg = cfg or {}
        return LISSTopoRoadNet(
            c4=cfg.get("c4", 64), c8=cfg.get("c8", 128), c16=cfg.get("c16", 256), dec_ch=cfg.get("dec_ch", 48),
            use_spectral_stem=not cfg.get("no_spectral_stem", False),
            use_topology_heads=not cfg.get("no_topology_heads", False),
            use_reconstruction=not cfg.get("no_reconstruction", False),
            drop_topology_heads=cfg.get("drop_topology_heads", False),
            hr_refine=cfg.get("hr_refine", False), learned_down2=cfg.get("learned_down2", False),
            sep_heads=cfg.get("sep_heads", False),
        ).to(device)
    from baselines import BASELINES
    return BASELINES[name]().to(device)


def raster_with_ignore(prob, gt, ignore, thr=0.5):
    keep = ignore < 0.5
    p = (prob >= thr) & keep
    t = (gt >= 0.5) & keep
    tp = float((p & t).sum()); fp = float((p & ~t).sum()); fn = float((~p & t).sum())
    iou = tp / (tp + fp + fn + EPS)
    prec = tp / (tp + fp + EPS); rec = tp / (tp + fn + EPS)
    f1 = 2 * prec * rec / (prec + rec + EPS)
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec}


def load_eval_masks():
    m = {}
    for p in EVAL_MASKS.glob("*.npz"):
        d = np.load(p)
        m[p.stem] = (d["major"], d["ignore"])
    return m


def parse_tile(name):
    scene, tile = name.split("__")
    y = int(re.search(r"y(\d+)", tile).group(1))
    x = int(re.search(r"x(\d+)", tile).group(1))
    return scene, y, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--graph-items", type=int, default=0,
                    help="score graph metrics on the first N tiles only; 0 = all tiles (default)")
    ap.add_argument("--runs", default=None, help="comma list; default = all runs with best.pt")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    emasks = load_eval_masks()
    if not emasks:
        sys.exit("no eval_masks — run scripts/major_masks.py first")
    tiles = sorted(TILES.glob("*.npz"))
    print(f"{len(tiles)} test tiles, {len(emasks)} scenes with visible-road masks", flush=True)

    if args.runs:
        run_names = args.runs.split(",")
    else:
        run_names = sorted(d.name for d in RUNS.iterdir()
                           if (d / "best.pt").exists() and (d / "config.json").exists())

    tile = 256
    results = {}
    for rn in run_names:
        cfg = json.loads((RUNS / rn / "config.json").read_text())
        model_name = cfg.get("model", "toporoad")
        model = build_model(model_name, device, cfg)
        ck = torch.load(RUNS / rn / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); model.eval()

        acc = {1: [], 2: []}
        cl = {1: [], 2: []}
        graph = {1: [], 2: []}
        gcount = 0
        with torch.no_grad():
            for tp in tiles:
                scene, y, x = parse_tile(tp.name)
                if scene not in emasks:
                    continue
                d = np.load(tp)
                img = torch.from_numpy(d["image"].astype(np.float32))[None].to(device)
                gt_all = d["mask"].astype(np.uint8)
                major, ignore = emasks[scene]
                mj = major[y:y + tile, x:x + tile]
                ig = ignore[y:y + tile, x:x + tile]
                if mj.shape != gt_all.shape:
                    continue
                prob = torch.sigmoid(model(img)["fused_logit"])[0, 0].float().cpu().numpy()
                zero_ig = np.zeros_like(ig)
                acc[1].append(raster_with_ignore(prob, gt_all, zero_ig, args.thr))
                acc[2].append(raster_with_ignore(prob, mj, ig, args.thr))
                if args.graph_items <= 0 or gcount < args.graph_items:
                    cl[1].append(cldice_metric(prob[None, None], gt_all[None, None], args.thr))
                    prob2 = prob * (ig < 0.5)
                    cl[2].append(cldice_metric(prob2[None, None], (mj * (ig < 0.5))[None, None], args.thr))
                    for tier, g, ig_arr in ((1, gt_all, zero_ig), (2, mj, ig)):
                        keep = ig_arr < 0.5
                        a = apls_raster(prob * keep, g * keep, thr=args.thr)
                        j = junction_recall(prob * keep, g * keep, thr=args.thr)
                        if not np.isnan(a):
                            graph[tier].append((a, j))
                    gcount += 1

        def agg(lst, key):
            v = [r[key] for r in lst]
            return float(np.mean(v)) if v else float("nan")

        row = {"model": model_name}
        for tier in (1, 2):
            row[f"t{tier}_iou"] = agg(acc[tier], "iou")
            row[f"t{tier}_f1"] = agg(acc[tier], "f1")
            row[f"t{tier}_cldice"] = float(np.nanmean([c for c in cl[tier] if not np.isnan(c)])) if cl[tier] else float("nan")
            gv = graph[tier]
            row[f"t{tier}_apls"] = float(np.mean([a for a, _ in gv])) if gv else float("nan")
            row[f"t{tier}_jrec"] = float(np.mean([j for _, j in gv])) if gv else float("nan")
        results[rn] = row
        print(f"{rn:30s} T1 IoU {row['t1_iou']:.4f} clD {row['t1_cldice']:.4f} APLS {row['t1_apls']:.4f} "
              f"| T2 IoU {row['t2_iou']:.4f} clD {row['t2_cldice']:.4f} APLS {row['t2_apls']:.4f}", flush=True)

    out = RUNS / "test_twotier_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
