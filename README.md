# LISS-TopoRoadNet: road extraction at 5.8 m from Resourcesat LISS-IV

Code and benchmark tools for

> Adithya Annamalai J K and Vijayalakshmi A. **Road Extraction at the 5.8 m Limit. A LISS-IV Benchmark with Two-Tier,
> Human-Grounded Evaluation.** In *Proceedings of the 17th Indian Conference on Computer Vision, Graphics and Image
> Processing (ICVGIP 2026)*, Kolkata, India.

At 5.8 m a two-lane road is one or two pixels wide. This repository holds the benchmark construction (tiling, OSM
labels, the two-tier visible-road masks, splits and the external Assam test set), the evaluation (tile metrics,
the raster APLS proxy APLS_r, strict symmetric APLS on stitched scenes and a bridge to the official SpaceNet APLS
code), the LISS-TopoRoadNet model and its baselines, every training configuration from the paper, and the app used
for the human observability study.

The imagery is not redistributed. LISS-IV scenes come from the National Remote Sensing Centre (NRSC) through the
Bhoonidhi portal. `scenes.csv` lists the 14 scene identifiers, satellites, dates, path/row and split, and the scripts
rebuild tiles and labels from the source scenes. Road labels are derived from OpenStreetMap data, (c) OpenStreetMap
contributors.

## Repository layout

```
src/            model (LISSTopoRoadNet), baselines (U-Net, D-LinkNet, RoadFormer), losses, targets, data, training,
                tile metrics and APLS_r (graph_metrics.py)
scripts/        benchmark construction (liss4_prepare, make_splits, precompute_targets, major_masks), tile-level
                evaluation (evaluate_twotier), human-study scoring, Bhoonidhi download helper
evaluation/     paper evaluation: whole-scene inference and mosaics, strict symmetric APLS (vector_apls*.py),
                official SpaceNet APLS bridge (official_apls_*.py), external-set labels (build_external_gt.py),
                strata, human-traced check, localisation diagnostics, aggregation with spatial block bootstrap,
                and the pre-registered decoder study and training-change gates
configs/        the training configuration of every run in the paper (see configs/README.md)
annotation/     human observability study app (standard-library server + canvas)
scenes.csv      benchmark and external scenes
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # install PyTorch for your CUDA build first
python scripts/smoke_test.py         # needs a CUDA GPU; synthetic end-to-end check of model, losses and targets
```

Run all commands from the repository root. Data paths default to `data/` (for example `data/liss4_tiles`).

## Rebuilding the benchmark

1. Download the scenes in `scenes.csv` from Bhoonidhi. `scripts/bhoonidhi_fetch.py` reads the account from the
   `BHOONIDHI_USER` and `BHOONIDHI_PASS` environment variables (the API needs IP whitelisting), and manual download
   also works.
2. Tile, normalise and label each scene, then build splits, topology targets and the Tier-2 masks.

```bash
python scripts/liss4_prepare.py --raw <scenes_dir> --out data/liss4_tiles
python scripts/make_splits.py --tiles data/liss4_tiles
python scripts/precompute_targets.py --tiles data/liss4_tiles
python scripts/major_masks.py --all
```

3. For the external Assam set, `evaluation/build_external_gt.py` applies the same labelling recipe to a regional
   OpenStreetMap extract.

## Training

Each file in `configs/` holds the full argument set of one run. For example, the full model and U-Net:

```bash
python src/train.py --run-dir runs/toporoad_full --data-root data/liss4_tiles --tile 256 --batch-size 16 \
  --epochs 60 --warmup-epochs 5 --l-cldice 1.0 --l-aff 0.25 --l-cons 0.1 --skel-pos-weight 10
python src/train.py --run-dir runs/unet --data-root data/liss4_tiles --model unet --epochs 60
```

Ablation flags are `--no-spectral-stem`, `--no-reconstruction`, `--drop-topology-heads` and `--l-aux 0` (deep
supervision), band ablation is `--zero-bands` (0 green, 1 red, 2 NIR), and the U-Net controls use `--model unet` or
`--model unet_ds` with `--baseline-topo-loss`.

## Evaluation

- Tile metrics (IoU, clDice, APLS_r, junction recall) at both tiers on all test tiles:
  `python scripts/evaluate_twotier.py --runs toporoad_full,unet`. The paper numbers come from
  `evaluation/infer_test_tiles.py` and `evaluation/tile_metrics.py`, which do the same on all 6,076 tiles.
- Strict symmetric APLS on stitched scenes: `evaluation/infer_scene_probs.py`, `evaluation/infer_scene_mosaic.py`,
  then `evaluation/vector_apls_run.py SCENE RUN1,RUN2`, and `evaluation/aggregate_families.py` for model comparisons
  with the spatial block bootstrap.
- Official SpaceNet APLS: clone https://github.com/CosmiQ/apls to `third_party/apls` (or set `APLS_REF`), then run
  `evaluation/official_apls_full.py`. `evaluation/stubs.py` stubs out GDAL, which only the official raster helpers use.

## Main results

Tier 2 (visible roads). IoU and APLS_r on all 6,076 test tiles, mean and standard deviation over three seeds
(RoadFormer one run). Official APLS on 1.5 km windows of whole stitched scenes, mean over runs.

| Model | Params | IoU | APLS_r | Official APLS, test scenes | Official APLS, Assam scenes |
|---|---|---|---|---|---|
| U-Net | 7.76 M | 0.103 ± 0.003 | 0.241 ± 0.012 | 0.071 ± 0.004 | 0.072 ± 0.006 |
| D-LinkNet | 5.25 M | 0.098 ± 0.006 | 0.205 ± 0.021 | 0.055 ± 0.007 | 0.065 ± 0.020 |
| RoadFormer | 2.25 M | 0.087 | 0.147 | 0.040 | 0.043 |
| LISS-TopoRoadNet (ours) | 3.97 M | 0.099 ± 0.004 | 0.260 ± 0.002 | 0.068 ± 0.007 | 0.082 ± 0.015 |

U-Net is slightly ahead on the original test scenes under the official APLS, and our model leads on the unseen Assam
scenes. Every model falls towards zero as windows grow to whole scenes. See the paper for intervals, ablations,
strata and the human-confirmed recall.

The version of the paper that went to review computed APLS_r and clDice on the first 250 test tiles only, because of
the old `--graph-items 250` default in `scripts/evaluate_twotier.py`. The default now scores all tiles.

## Human observability study

`annotation/server.py` (standard library only) serves `annotation/index.html`, where annotators shade the roads they
can see on a tile. The study tiles are not included because they contain LISS-IV crops. Score submissions with
`python scripts/score_annotations.py annotation/submissions`.

## Citation

```bibtex
@inproceedings{annamalai2026road,
  title     = {Road Extraction at the 5.8\,m Limit: A {LISS-IV} Benchmark with Two-Tier, Human-Grounded Evaluation},
  author    = {{Adithya Annamalai J K} and {Vijayalakshmi A}},
  booktitle = {Proceedings of the 17th Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP)},
  year      = {2026}
}
```

## License

MIT, see `LICENSE`. LISS-IV imagery is subject to NRSC terms, and OpenStreetMap data to the ODbL.
