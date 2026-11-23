#!/bin/sh
#BSUB -q gpuv100
#BSUB -J eval
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/eval.out
#BSUB -e hpc/logs/eval.err

# Score sample sets on a GPU, structures included:   bsub < hpc/eval.sh
#
# Run `bash hpc/setup_env.sh` once before this, to install the environment and download the
# ESMFold weights. Add your own sample CSVs to the list below; a name with no CSV is skipped with
# a message rather than failing the job, so this can be submitted before every model has been
# sampled. Each set lands in results/phase3/eval/<name>/.
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
##BSUB -u <EMAIL_ADDRESS>
##BSUB -B
##BSUB -N

set -eu
cd "${LS_SUBCWD:-$PWD}"

SETS="control_real control_shuffled \
      cg_AcyP_lambda0 cg_AcyP_lambda0.5 cg_AcyP_lambda1 cg_AcyP_lambda2 cg_AcyP_lambda4 \
      cfg_null cfg_AcyP_w0 cfg_AcyP_w0.5 cfg_AcyP_w1 cfg_AcyP_w2 cfg_AcyP_w3 \
      cfg_HPr_w2 cfg_HU_w2"


for NAME in $SETS; do
    SAMPLES="results/phase3/sequences/$NAME.csv"
    if [ ! -f "$SAMPLES" ]; then
        echo "== $NAME: no $SAMPLES, skipping"
        continue
    fi
    echo "== $NAME"
    bash hpc/run_eval.sh "$SAMPLES" "results/phase3/eval/$NAME"
done

echo "Done. Each set is in results/phase3/eval/<name>/"
