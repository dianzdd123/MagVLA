#!/bin/bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=MagVLA-v13-text
#SBATCH --output=slurm-v13-text-%j.out
#SBATCH --error=slurm-v13-text-%j.err
set -euo pipefail

# Submit from this variant directory. Slurm copies the script to a spool directory.
variant_dir="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "$0")" && pwd)}"
: "${MARSS_DATA_DIR:?Set MARSS_DATA_DIR to the absolute dataset path}"
cd "$variant_dir/codebase"
exec torchrun --standalone --nnodes=1 --nproc_per_node="${MARSS_GPUS:-4}" train_ddp.py "$@"
