#!/bin/bash
# Usage:
#   sbatch train_q05b.slurm              # IPPO (default)
#   METHOD=srpo sbatch train_q05b.slurm  # SRPO
#
#SBATCH --job-name=mappo_q05b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --partition=nvl,h100,l40s
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j_${METHOD:-ippo}.output
#SBATCH --error=logs/%x_%j_${METHOD:-ippo}.error
#SBATCH --export=ALL,ROCR_VISIBLE_DEVICES=

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
module load cuda/12.3.0
module load anaconda3/2024.02-1
conda activate srpo

cd "${REPO_DIR}"

echo "[INFO] Cleaning up old Ray processes..."
ray stop -f || true

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
    sleep 5
done

# Method-specific config
if [ "$METHOD" = "srpo" ]; then
    TRAINER_TYPE="risk_averse"
    KL_COEF=0.1          # adversary KL regularization = 1/tau, tau=10
    EXP_NAME="srpo_q05b"
else
    TRAINER_TYPE="mappo"
    KL_COEF=0.001        # standard PPO default (not applied to rewards by default)
    EXP_NAME="ippo_q05b"
fi

# Launch training
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "unset ROCR_VISIBLE_DEVICES; \
        python -m verl.trainer.main_mappo \
        data.train_files=${GSM8K_TRAIN} \
        data.val_files=${GSM8K_TEST} \
        data.train_batch_size=256 \
        data.max_prompt_length=2048 \
        data.max_response_length=512 \
        multi_agent.trainer_type=${TRAINER_TYPE} \
        multi_agent.agents.0.actor.model.path=${MODEL_NAME} \
        multi_agent.agents.1.actor.model.path=${MODEL_NAME} \
        multi_agent.agents.0.n_gpus_per_node=${GPUS_PER_NODE} \
        multi_agent.agents.1.n_gpus_per_node=${GPUS_PER_NODE} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.max_model_len=2048 \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.model.lora_rank=32 \
        actor_rollout_ref.model.lora_alpha=32 \
        actor_rollout_ref.model.target_modules=all-linear \
        critic.optim.lr=1e-5 \
        critic.ppo_micro_batch_size_per_gpu=4 \
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
        trainer.val_before_train=True" \
    2>&1 | tee "logs/${EXP_NAME}_${SLURM_JOB_ID}.log"

echo "[INFO] Training finished: METHOD=${METHOD}"
