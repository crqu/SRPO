#!/bin/bash
#SBATCH --job-name=build_flash_attn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=h100
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

# Clone flash-attn — pin to a release compatible with torch 2.8.x.
# HEAD (v2.8.4) declares a dependency on torch>=2.11.0 which conflicts with the
# installed 2.8.0+cu128, and its build system ignores TORCH_CUDA_ARCH_LIST.
echo "[INFO] Cloning flash-attn v2.7.4..."
git clone --depth=1 --branch v2.7.4 https://github.com/Dao-AILab/flash-attention.git "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Build configuration
# TORCH_CUDA_ARCH_LIST: restrict to H100 (sm_90) only — avoids compiling for
# sm_80/100/101/120, which multiplies compile time and RAM usage ~6x.
export TORCH_CUDA_ARCH_LIST="9.0"

# MAX_JOBS=4: with sm_90 only, each file spawns 1 cicc process × 4 nvcc threads
# = 16 concurrent cicc processes × ~4GB each = ~64GB, just within our RAM limit.
export MAX_JOBS=4

export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

# NOTE: do NOT unset CFLAGS/CXXFLAGS. Conda's ~cuda-nvcc_activate.sh injects
# -I.../targets/x86_64-linux/include into them, which is where cuda_runtime_api.h
# lives. Without it, the host C++ compiler can't find CUDA headers (job 1272800).
# The old AVX512 errors were caused by 'module load gcc/9.3.0' (now removed),
# not by the conda-set CFLAGS.

echo "[INFO] Building wheel (this takes ~30-90 min for sm_90 only)..."
# -v: stream build output in real-time so the log shows nvcc commands and any
#     cicc OOM kills instead of appearing silently stuck for hours.
pip wheel . --no-build-isolation -v -w "${WHEELS_DIR}"

echo "[INFO] Installing wheel..."
pip install "${WHEELS_DIR}"/flash_attn-*.whl --no-deps --force-reinstall

# Smoke test
echo "[INFO] Smoke test:"
python -c "import flash_attn; print('flash_attn version:', flash_attn.__version__)"

echo "[INFO] Build and install complete. Wheel saved to ${WHEELS_DIR}/"
ls -lh "${WHEELS_DIR}"/flash_attn-*.whl

# Cleanup build dir
rm -rf "${BUILD_DIR}"
