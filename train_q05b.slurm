#!/bin/bash
# Usage:
#   sbatch train_q05b.slurm              # IPPO (default)
#   METHOD=srpo sbatch train_q05b.slurm  # SRPO
#
#SBATCH --job-name=mappo_q05b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --partition=nvl,h100
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.output
#SBATCH --error=logs/%x_%j.error
#SBATCH --export=ALL,ROCR_VISIBLE_DEVICES=
#SBATCH --exclude=l02,l04,l05,l06

set -x

# Training method: ippo (default) or srpo
METHOD=${METHOD:-ippo}

# Paths
REPO_DIR=/weka/scratch/lshi40_llm/mallm/SRPO
CKPT_DIR=${REPO_DIR}/checkpoints/${METHOD}_q05b
DATA_DIR=${REPO_DIR}/data/gsm8k
GSM8K_TRAIN=${DATA_DIR}/train.parquet
GSM8K_TEST=${DATA_DIR}/test.parquet
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

mkdir -p logs "${CKPT_DIR}"

# Load environment
module reset
module load gcc/9.3.0
module load anaconda3/2024.02-1
conda activate srpo

cd "${REPO_DIR}"

echo "[INFO] Cleaning up old Ray processes on all nodes..."
srun --nodes=${SLURM_JOB_NUM_NODES} --ntasks=${SLURM_JOB_NUM_NODES} bash -c "ray stop -f" || true

# Download GSM8K data if needed
if [ ! -f "${GSM8K_TRAIN}" ]; then
    echo "[INFO] Downloading GSM8K data..."
    mkdir -p "${DATA_DIR}"
    python examples/data_preprocess/gsm8k.py --local_save_dir "${DATA_DIR}"
fi

# Fix ROCm/CUDA conflict
unset ROCR_VISIBLE_DEVICES

# Environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=4
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=2404c5a258f9f0623cde0ef3041df0a8bc425bb6
export WANDB_MODE=offline
# Prevent Ray from setting CUDA_VISIBLE_DEVICES="" for zero-GPU processes (driver,
# TaskRunner). Without this, Ray's current behavior hides the GPU from the head
# node's processes, which then propagates to WorkerDict actors on the same node
# causing "Flash Attention 2 not available on CPU" errors for agent_pool_1.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_BACKEND_LOG_LEVEL=debug
export VLLM_LOGGING_LEVEL=DEBUG
nvidia-smi

NUM_NODES=${SLURM_JOB_NUM_NODES:-2}
GPUS_PER_NODE=1
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}

# Build Ray cluster
mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "[INFO] IPv6 detected; using IPv4: $head_node_ip"
fi

port=6380
ip_head="${head_node_ip}:${port}"
export ip_head
echo "[INFO] Ray head: $ip_head"

# Start Ray head
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "unset ROCR_VISIBLE_DEVICES; \
        ray start --head --node-ip-address=${head_node_ip} --port=${port} \
        --num-cpus=${CPUS_PER_TASK} --num-gpus=${GPUS_PER_NODE} --block" &
sleep 20

# Start Ray workers
worker_num=$((NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "[INFO] Starting worker $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash -c "unset ROCR_VISIBLE_DEVICES; \
            ray start --address=${ip_head} \
            --num-cpus=${CPUS_PER_TASK} --num-gpus=${GPUS_PER_NODE} --block" &
done

# Poll until all NUM_NODES nodes have registered with the Ray head.
# A fixed sleep is a race condition: the worker node needs time to load
# its environment, run `ray start`, and complete the TCP handshake.
echo "[INFO] Waiting for all ${NUM_NODES} Ray nodes to register..."
for ((attempt = 1; attempt <= 24; attempt++)); do
    ALIVE=$(python -c "
import ray
ray.init(address='${ip_head}', ignore_reinit_error=True)
print(sum(1 for n in ray.nodes() if n['Alive']))
" 2>/dev/null || echo 0)
    if [ "${ALIVE:-0}" -ge "${NUM_NODES}" ]; then
        echo "[INFO] All ${NUM_NODES} nodes registered after $((attempt * 5))s"
        break
    fi
    echo "[INFO] ${ALIVE:-0}/${NUM_NODES} nodes registered, retrying in 5s..."
    sleep 5
done
if [ "${ALIVE:-0}" -lt "${NUM_NODES}" ]; then
    echo "[ERROR] Only ${ALIVE:-0}/${NUM_NODES} nodes registered after 120s; aborting."
    exit 1
fi

# Method-specific config
case "$METHOD" in
    srpo_main)
        TRAINER_TYPE="risk_averse"
        KL_COEF=0.1          # adversary KL regularization = 1/tau, tau=10
        EXP_NAME="srpo_main_q05b"
        ENTROPY_OVERRIDE="+multi_agent.agents.0.actor.actor.entropy_coeff=0.05"
        ;;
    srpo_reward_only)
        TRAINER_TYPE="risk_averse"
        KL_COEF=0.1
        EXP_NAME="srpo_reward_only_q05b"
        ENTROPY_OVERRIDE=""
        ;;
    ippo|*)
        TRAINER_TYPE="mappo"
        KL_COEF=0.001        # standard PPO default (not applied to rewards by default)
        EXP_NAME="ippo_q05b"
        ENTROPY_OVERRIDE=""
        ;;
esac

# Cross-pair probe partner: optional. Default empty (probe disabled).
# To enable, export PARTNER_CKPT_DIR=/path/to/sibling/arm/ckpt before invocation.
PARTNER_CKPT_DIR="${PARTNER_CKPT_DIR:-}"

# Debug info
echo "[DEBUG] hostname=$(hostname)"
echo "[DEBUG] pwd=$(pwd)"
echo "[DEBUG] which python=$(which python)"
echo "[DEBUG] python version=$(python -V 2>&1)"
echo "[DEBUG] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[DEBUG] NCCL_DEBUG=${NCCL_DEBUG}"
echo "[DEBUG] VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}"
echo "[DEBUG] HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR}"
nvidia-smi
python - <<'PY'
import torch, vllm, ray
print("[DEBUG] torch", torch.__version__)
print("[DEBUG] cuda available", torch.cuda.is_available())
print("[DEBUG] device count", torch.cuda.device_count())
print("[DEBUG] vllm", vllm.__version__)
print("[DEBUG] ray", ray.__version__)
PY


# Launch training
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "unset ROCR_VISIBLE_DEVICES; \
        python -m verl.trainer.main_mappo \
        data.train_files=${GSM8K_TRAIN} \
        data.val_files=${GSM8K_TEST} \
        data.train_batch_size=256 \
        data.max_prompt_length=2048 \
        data.max_response_length=512 \
        data.seed=42 \
        multi_agent.trainer_type=${TRAINER_TYPE} \
        multi_agent.agents.0.actor.model.path=${MODEL_NAME} \
        multi_agent.agents.1.actor.model.path=${MODEL_NAME} \
        multi_agent.agents.0.n_gpus_per_node=${GPUS_PER_NODE} \
        multi_agent.agents.1.n_gpus_per_node=${GPUS_PER_NODE} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.max_model_len=2048 \
        actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        actor_rollout_ref.rollout.enforce_eager=true \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=NONE \
        actor_rollout_ref.model.path=${MODEL_NAME} \
        actor_rollout_ref.model.lora_rank=32 \
        actor_rollout_ref.model.lora_alpha=32 \
        actor_rollout_ref.model.target_modules=all-linear \
        critic.model.path=${MODEL_NAME} \
        critic.optim.lr=1e-5 \
        critic.ppo_micro_batch_size_per_gpu=32 \
        algorithm.kl_ctrl.kl_coef=${KL_COEF} \
        trainer.total_training_steps=200 \
        trainer.test_freq=10 \
        trainer.save_freq=50 \
        trainer.nnodes=${NUM_NODES} \
        trainer.n_gpus_per_node=${GPUS_PER_NODE} \
        trainer.default_local_dir=${CKPT_DIR} \
        trainer.project_name=srpo_experiments \
        trainer.experiment_name=${EXP_NAME} \
        trainer.logger=[console,wandb] \
        trainer.val_before_train=True \
        multi_agent.cross_pair_probe.partner_ckpt_dir=${PARTNER_CKPT_DIR} \
        ${ENTROPY_OVERRIDE}" \
    2>&1 | tee "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

echo "[INFO] Training finished: METHOD=${METHOD}"
