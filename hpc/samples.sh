#!/bin/bash
#BSUB -q gpuv100
#BSUB -J phase3_samples
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o hpc/logs/samples.out
#BSUB -e hpc/logs/samples.err

set -eu
cd "${LS_SUBCWD:-$PWD}"
source hpc/env.sh
nvidia-smi --query-gpu=name --format=csv,noheader || true

echo "== controls"
uv run python eval/make_controls.py --num 256 --out results/phase3/sequences

echo "== judge classifier"
uv run python src/protgen/models/judge.py

echo "== classifier guidance, n=256"
for L in 0 0.5 1 2 4; do
  uv run python src/protgen/models/train_CG.py --mode sample --family AcyP --guidance $L --n 256
done
# train_CG.py is Phase 2's and hardcodes results/sequences/; move its output into ours.
mkdir -p results/phase3/sequences
mv results/sequences/cg_AcyP_*.csv results/phase3/sequences/

echo "== classifier-free guidance"
uv run python src/protgen/models/train_cfg.py --mode sample --family AcyP --w 0 0.5 1 2 3
uv run python src/protgen/models/train_cfg.py --mode sample --family null
uv run python src/protgen/models/train_cfg.py --mode sample --family HPr --w 2
uv run python src/protgen/models/train_cfg.py --mode sample --family HU --w 2

echo "== family membership"
uv run python src/protgen/models/score_family.py --predictor guide
uv run python src/protgen/models/score_family.py --predictor judge

echo "== sequence metrics"
for f in results/phase3/sequences/{control_,cg_AcyP_,cfg_}*.csv; do
  uv run python eval/eval.py --samples "$f" --out "results/phase3/eval/$(basename "${f%.csv}")"
done
