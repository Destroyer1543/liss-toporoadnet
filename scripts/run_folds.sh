#!/bin/bash
# Clean equal-budget scene-level 5-fold CV of the full model. Each fold holds out
# two scenes (all 10 covered once), trains on the other eight at the SAME 40-epoch
# budget, sequentially (one job at a time) to stay light on the shared GPUs.
# Resume-safe: skips folds that already reached epoch 39.
cd /root/Road/src
OURS="--l-cldice 1.0 --l-aff 0.25 --l-cons 0.1 --skel-pos-weight 10 --workers 0"
EP=40

declare -A FOLD=(
 [cvfold1]="R2F24AUG2023064073010100066SSANSTUC00GTDD,RAF23OCT2023035685010100064SSANSTUC00GTDA"
 [cvfold2]="R2F12JUN2023063029009600051SSANSTUC00GTDC,RAF03JUN2024038861009300049SSANSTUC00GTDD"
 [cvfold3]="R2F04JAN2026076342010100056SSANSTUC00GTDD,RAF13MAY2025043748010900055SSANSTUC00GTDA"
 [cvfold4]="R2F10APR2026077706010100066SSANSTUC00GTDD,RAF21APR2025043436009500051SSANSTUC00GTDD"
 [cvfold5]="R2F08FEB2024066453010100064SSANSTUC00GTDA,RAF28MAR2025043095009500049SSANSTUC00GTDA"
)

for name in cvfold1 cvfold2 cvfold3 cvfold4 cvfold5; do
  d=../runs/exp_${name}; mkdir -p "$d"
  done_ep=$(grep -c '^ep ' "$d/train.log" 2>/dev/null || echo 0)
  if [ -f "$d/log.csv" ] && [ "$(tail -1 "$d/log.csv" 2>/dev/null | cut -d, -f1)" -ge 39 ] 2>/dev/null; then
    echo "$(date +%H:%M) $name already complete, skip" >> /root/Road/runs/folds.log; continue
  fi
  # wait until a GPU has >= 8 GB free, then pick it
  while true; do
    read G GF < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | tr ',' ' ')
    [ "${GF:-0}" -ge 8000 ] && break
    sleep 120
  done
  echo "$(date +%H:%M) launching $name on GPU $G (holdout ${FOLD[$name]})" >> /root/Road/runs/folds.log
  TMPDIR=/data/road_liss4/tmp OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 train.py \
    --run-dir "$d" --data-root /data/road_liss4/liss4_tiles --tile 256 --batch-size 16 \
    --epochs $EP --warmup-epochs 5 --gpu $G $OURS --holdout-scene "${FOLD[$name]}" \
    > "$d/train.log" 2>&1
  echo "$(date +%H:%M) $name finished" >> /root/Road/runs/folds.log
done
echo "$(date +%H:%M) ALL FOLDS DONE" >> /root/Road/runs/folds.log
