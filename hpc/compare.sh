#!/bin/bash
#BSUB -q hpc
#BSUB -J compare_structures
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:00
#BSUB -o hpc/logs/compare.out
#BSUB -e hpc/logs/compare.err

# Structural comparison of the Phase 3 sets. CPU only.   bsub < hpc/compare.sh

set -eu
cd "${LS_SUBCWD:-$PWD}"
source hpc/env.sh

uv run --group eval python eval/compare_structures.py \
    --sets real=results/phase3/eval/control_real \
           shuffled=results/phase3/eval/control_shuffled \
           unguided=results/phase3/eval/cfg_null \
           cfg_w2=results/phase3/eval/cfg_AcyP_w2 \
           cg_l4=results/phase3/eval/cg_AcyP_lambda4 \
    --reference real \
    --out results/phase3/eval/comparison