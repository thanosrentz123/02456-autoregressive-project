#!/bin/bash
#BSUB -q gpuv100
#BSUB -J train_cfg
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/train_cfg.out
#BSUB -e hpc/logs/train_cfg.err

# Phase 3: train the label-conditioned Transformer.   bsub < hpc/train_cfg.sh

set -eu
cd "${LS_SUBCWD:-$PWD}"
source hpc/env.sh

echo "commit: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader || true

uv run python src/protgen/models/train_cfg.py --mode train --epochs 200
