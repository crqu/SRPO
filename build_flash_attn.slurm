#!/bin/bash
#SBATCH --job-name=build_flash_attn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=nvl,l40s,h100
#SBATCH --output=logs/build_flash_attn_%j.log
#SBATCH --error=logs/build_flash_attn_%j.log

set -exo pipefail

REPO_DIR=/weka/scratch/lshi40_llm/mallm/SRPO
BUILD_DIR=/tmp/flash-attn-build-${SLURM_JOB_ID}
WHEELS_DIR=${REPO_DIR}/wheels

mkdir -p logs "${WHEELS_DIR}"

# Load environment (do NOT load system gcc — conda env ships its own GCC 14 toolchain;
# loading system gcc/9.3.0 pollutes CFLAGS with incompatible AVX512 headers)
module reset
module load anaconda3/2024.02-1
conda activate srpo

# Verify CUDA is accessible
echo "[INFO] CUDA version:"
nvcc --version

echo "[INFO] PyTorch CUDA:"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

# Clone flash-attn
echo "[INFO] Cloning flash-attn..."
git clone --depth=1 https://github.com/Dao-AILab/flash-attention.git "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Build configuration
export TORCH_CUDA_ARCH_LIST="9.0"   # H100 only — reduces compilation units ~3x
export MAX_JOBS=4                   # stay well within 64GB RAM (each cicc job ~4-8GB)
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

echo "[INFO] Building wheel (this takes ~30-60 min)..."
pip wheel . --no-build-isolation -w "${WHEELS_DIR}"

echo "[INFO] Installing wheel..."
pip install "${WHEELS_DIR}"/flash_attn-*.whl --no-deps --force-reinstall

# Smoke test
echo "[INFO] Smoke test:"
python -c "import flash_attn; print('flash_attn version:', flash_attn.__version__)"

echo "[INFO] Build and install complete. Wheel saved to ${WHEELS_DIR}/"
ls -lh "${WHEELS_DIR}"/flash_attn-*.whl

# Cleanup build dir
rm -rf "${BUILD_DIR}"
