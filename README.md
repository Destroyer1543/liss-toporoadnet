# LISS-TopoRoadNet

Code for topology-aware road extraction from Resourcesat-2/2A LISS-IV imagery at
5.8 m. This repository holds the model, the training and evaluation code, the
two-tier evaluation protocol, the OpenStreetMap label and mask construction, the
metric implementations, and the human-observability annotation app.

The imagery itself is not redistributed here. LISS-IV scenes are distributed by the
National Remote Sensing Centre (NRSC) through the Bhoonidhi portal and cannot be
re-shared. We provide the scene identifiers and acquisition dates in `scenes.csv`
and the scripts needed to reproduce the tiles and labels from the source scenes.

## What is in this repository

```
src/                 model, losses, targets, data, metrics, training
  model.py           LISSTopoRoadNet (spectral stem, cross-scale encoder,
                     topology decoder, four task heads, reconstruction)
  baselines.py       U-Net, D-LinkNet, RoadFormer (SegFormer-style)
  losses.py          FocalTversky, Dice, soft-clDice, skeleton, width,
                     orientation, consistency, reconstruction, deep supervision
  targets.py         mask -> skeleton / width / orientation targets
  data.py            tile dataset, band ablation, cross-region splits
  metrics.py         IoU, F1, clDice
  train.py           config-driven trainer (warmup then topology curriculum)
  graph_metrics.py   APLS_r (raster APLS proxy) and junction recall
  pretrain_mae.py    masked-autoencoder pretraining (ablation)
  postprocess.py     affinity graph post-processing (reported as a negative result)
scripts/             data preparation, evaluation, and experiment scripts
annotation/          human observability study app (stdlib server + web canvas)
figures/             scripts that render the paper figures
scenes.csv           the ten LISS-IV scenes, satellites, dates, and splits
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
# install PyTorch for your CUDA build from pytorch.org, then:
pip install -r requirements.txt
```

A smoke test on synthetic data checks that the model, targets, losses, and
reconstruction run end to end:

```bash
python scripts/smoke_test.py
```

## Reproducing the benchmark

The scenes in `scenes.csv` are ordered from the Bhoonidhi portal. NRSC requires a
registered account and, for the API, IP whitelisting. `scripts/bhoonidhi_fetch.py`
implements the API workflow and reads credentials from `BHOONIDHI_USER` and
`BHOONIDHI_PASS`. Manual portal download also works.

Once the scene zips are in a working directory, build tiles and OSM labels, then the
splits:

```bash
# 1. tile each scene, normalise bands, rasterise OSM roads as bootstrap labels
python scripts/liss4_prepare.py --raw <scenes_dir> --out <tiles_dir>

# 2. whole-scene holdout splits (train / val / test per scenes.csv)
python scripts/make_splits.py --tiles <tiles_dir>

# 3. precompute topology targets into the tiles (skips per-epoch skeletonisation)
python scripts/precompute_targets.py --tiles <tiles_dir>

# 4. build the two-tier evaluation masks for the test scenes
python scripts/major_masks.py --all
```

Train the full model and the baselines on the same split and budget:

```bash
python src/train.py --run-dir runs/toporoad_full --data-root <tiles_dir> \
  --tile 256 --batch-size 16 --epochs 60 --warmup-epochs 5 \
  --l-cldice 1.0 --l-aff 0.25 --l-cons 0.1 --skel-pos-weight 10
python src/train.py --run-dir runs/unet --data-root <tiles_dir> --model unet --epochs 60
```

Evaluate on the held-out test scenes at both tiers:

```bash
python scripts/evaluate_twotier.py --runs "toporoad_full,unet,dlinknet,roadformer"
```

## Ablations and experiments

- Component ablations use the flags `--no-spectral-stem`, `--no-topology-heads`,
  `--no-reconstruction`, and `--l-aux 0` (deep supervision).
- Band ablation uses `--zero-bands` (0=green, 1=red, 2=NIR).
- Scene-level 5-fold cross-validation is in `scripts/run_folds.sh`.
- APLS_r is compared against a junction-graph APLS in `scripts/apls_validate.py`
  and `scripts/apls_ranking.py`.
- MAE pretraining is `src/pretrain_mae.py` with `src/train.py --init`.

## Human observability study

`annotation/` is a self-contained app. `server.py` uses only the Python standard
library and serves `index.html`, a canvas where annotators shade the roads they can
see on a tile. The 24 study tiles (`tiles.json`) are not included, because they embed
LISS-IV crops. Regenerate them from your own prepared tiles, then:

```bash
cd annotation && python3 server.py        # serves on 0.0.0.0:8000
python scripts/score_annotations.py annotation/submissions
```

Scoring reports human recall against road width. The ground-truth masks used for
scoring are never shown to annotators.

## Citation

To be added on acceptance.

## License

MIT, see `LICENSE`.
