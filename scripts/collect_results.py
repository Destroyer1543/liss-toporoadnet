import argparse
import csv
import json
import pathlib

RUNS = pathlib.Path(__file__).resolve().parent.parent / "runs"


def _num(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def best_row(log_path, key="cldice"):
    with open(log_path) as f:
        rows = [r for r in csv.DictReader(f) if _num(r, key) is not None]
    if not rows:
        return None
    return max(rows, key=lambda r: float(r[key]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", default="cldice")
    args = ap.parse_args()
    print(f"{'run':38s} {'model':10s} {'ep':>3s} {'iou':>7s} {'f1':>7s} {'cldice':>7s}")
    results = []
    for d in sorted(RUNS.iterdir()):
        log = d / "log.csv"
        if not log.exists():
            continue
        cfg = {}
        cfg_p = d / "config.json"
        if cfg_p.exists():
            cfg = json.loads(cfg_p.read_text())
        r = best_row(log, args.sort)
        if r is None:
            continue
        results.append((d.name, cfg.get("model", "toporoad"), r))
    for name, model, r in results:
        print(f"{name:38s} {model:10s} {r['epoch']:>3s} {r['iou']:>7s} {r['f1']:>7s} {r['cldice']:>7s}")


if __name__ == "__main__":
    main()
