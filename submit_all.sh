#!/bin/bash
# Submit the full IPPO/SRPO sweep. Defaults: 4 models x 2 methods x 3 seeds = 24 jobs.
# Override via env (space-separated):
#   METHODS="ippo srpo_main"             # also: srpo_reward_only
#   MODELS="q05b q06b q3b q4b"
#   SEEDS="1 2 3"
#
# Usage:
#   bash submit_all.sh                              # default sweep
#   MODELS="q05b q06b" bash submit_all.sh           # only small models
#   METHODS=srpo_main SEEDS=1 bash submit_all.sh    # one method, one seed
#   DRY_RUN=1 bash submit_all.sh                    # print sbatch lines, do not submit
set -e

METHODS=${METHODS:-"ippo srpo_main"}
MODELS=${MODELS:-"q05b q06b q3b q4b"}
SEEDS=${SEEDS:-"1 2 3"}
DRY_RUN=${DRY_RUN:-0}

if [ "$DRY_RUN" = "1" ]; then
    SBATCH="echo [DRY-RUN] sbatch"
else
    SBATCH="sbatch"
fi

for model in $MODELS; do
    case "$model" in
        q05b|q06b) NODES=2; GPN=1 ;;
        q3b|q4b)   NODES=2; GPN=2 ;;
        *) echo "[ERROR] Unknown MODEL=$model; expected q05b|q06b|q3b|q4b" >&2; exit 2 ;;
    esac
    for method in $METHODS; do
        for seed in $SEEDS; do
            METHOD=$method MODEL=$model SEED=$seed \
                $SBATCH --nodes=$NODES --gpus-per-node=$GPN --gres=gpu:$GPN \
                --job-name=mappo_${method}_${model}_s${seed} train.slurm
        done
    done
done
