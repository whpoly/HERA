#!/bin/bash
#SBATCH --job-name=hera_hypergraph
#SBATCH --output=/home/wuhao/job_logs/hera_hypergraph_%j.out
#SBATCH --error=/home/wuhao/job_logs/hera_hypergraph_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --chdir=/home/wuhao

set -euo pipefail

# Running this file with bash submits it once after ensuring that Slurm's
# output directory exists. Running it with sbatch executes the job normally.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p /home/wuhao/job_logs
    exec sbatch "$0"
fi

CONDA_PATH="/home/wuhao/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="hera"
PROJECT_PARENT="/home/wuhao"
RUN_ROOT="HERA/hypergraph/four_gpu"

source "$CONDA_PATH"
conda activate "$ENV_NAME"
cd "$PROJECT_PARENT"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start time: $(date --iso-8601=seconds)"
echo "Run root: $PROJECT_PARENT/$RUN_ROOT"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi

datasets=(2dmd_low 2dmd_high native och)
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if (( ${#allocated_gpus[@]} < ${#datasets[@]} )); then
    echo "Expected 4 allocated GPUs, but CUDA_VISIBLE_DEVICES contains: ${CUDA_VISIBLE_DEVICES:-<not set>}" >&2
    exit 1
fi

pids=()
dataset_logs=()

for index in "${!datasets[@]}"; do
    dataset="${datasets[$index]}"
    gpu_id="${allocated_gpus[$index]}"
    dataset_run_dir="$RUN_ROOT/$dataset"
    dataset_log="/home/wuhao/job_logs/hera_hypergraph_${SLURM_JOB_ID}_${dataset}.log"

    echo "Launching dataset=$dataset on allocated GPU=$gpu_id"
    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        echo "Dataset: $dataset"
        echo "Visible GPU: $CUDA_VISIBLE_DEVICES (PyTorch device cuda:0)"
        echo "Start time: $(date --iso-8601=seconds)"
        nvidia-smi

        python -m HERA.main \
            --model all \
            --dataset "$dataset" \
            --mode hypergraph \
            --hypergraph-radius 3.0 \
            --epochs 500 \
            --device cuda:0 \
            --seed 123 \
            --atom-init HERA/atom_init.json \
            --run-dir "$dataset_run_dir" \
            --resume

        echo "End time: $(date --iso-8601=seconds)"
    ) >"$dataset_log" 2>&1 &

    pids+=("$!")
    dataset_logs+=("$dataset_log")
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "Completed ${datasets[$index]} successfully; log=${dataset_logs[$index]}"
    else
        exit_code=$?
        echo "Failed ${datasets[$index]} with exit code $exit_code; log=${dataset_logs[$index]}" >&2
        status=1
    fi
done

echo "End time: $(date --iso-8601=seconds)"
exit "$status"
