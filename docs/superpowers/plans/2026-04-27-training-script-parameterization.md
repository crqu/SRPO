# Training Script Parameterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `train_q05b.slurm` with a single parameterized `train.slurm` plus a `submit_all.sh` helper that drives the full IPPO×SRPO×{Q0.5B, Q0.6B, Q3B, Q4B}×3-seed sweep.

**Architecture:** Carry over the existing slurm scaffold (sbatch header, env, Ray bootstrap, `case "$METHOD"` block) verbatim and add a `case "$MODEL"` block before it that sets `MODEL_NAME`, `NNODES`, `GPUS_PER_NODE`, `TP_SIZE`, `MICRO_BSZ`, `GMU`, `TOTAL_STEPS`. Critic gets a LoRA adapter with an explicit Qwen attention/MLP module list (NOT `all-linear`, which collides with the PEFT TOKEN_CLS value head). `submit_all.sh` issues one `sbatch` per `(model, method, seed)` triple with `--nodes` / `--gpus-per-node` chosen per model.

**Tech Stack:** bash, SLURM (`sbatch`, `srun`), Ray, verl, vLLM, PEFT, hydra-core. Compute: H100/NVL partitions for production; 2× L40S for local smoke validation.

**Spec:** [`docs/superpowers/specs/2026-04-26-training-script-parameterization-design.md`](../specs/2026-04-26-training-script-parameterization-design.md)

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `train.slurm` | **Create** | Single parameterized SLURM script — `case "$MODEL"` (sizing) + `case "$METHOD"` (trainer wiring) + hydra invocation |
| `submit_all.sh` | **Create** | One-shot submission helper that loops over `MODELS × METHODS × SEEDS` and issues `sbatch` calls with the right `--nodes` / `--gpus-per-node` |
| `train_q05b.slurm` | **Keep as-is** | Historical reference; spec §"Out of scope" mandates we do not delete it |
| `verl/workers/fsdp_workers.py` | **Untouched** | Spec §"Out of scope" — the config-only critic LoRA fix removes the need for the `exclude_modules` patch |

---

## Task 0: Close the critic-LoRA ckpt-save validation gap

> **STATUS: DONE 2026-04-27.** Smoke `/tmp/smoke_srpo_critic_lora.sh` with `trainer.save_freq=1` ran to exit code 0 on 2× L40S. Critic FSDP shards confirmed on disk under `checkpoints/srpo_main_q05b_critic_lora_smoke/global_step_1/critic/{0,1}/`. Spec §"Validation evidence" updated. Steps below kept as a record; an executor with no doubt about the verification can mark them complete and proceed to Task 1.

The spec §"Validation evidence" originally listed one unverified path: critic LoRA checkpoint save was never exercised (the prior smoke was preempted before `save_freq=50` fired). Closed before building on top.

**Files:**
- Modify: `/tmp/smoke_srpo_critic_lora.sh` — change `trainer.save_freq=50` → `trainer.save_freq=1`
- Read: `${REPO_DIR}/checkpoints/srpo_main_q05b_critic_lora_smoke/global_step_1/critic/` after the run

- [ ] **Step 1: Edit the smoke script's save_freq**

```bash
sed -i 's/trainer.save_freq=50/trainer.save_freq=1/' /tmp/smoke_srpo_critic_lora.sh
grep "save_freq" /tmp/smoke_srpo_critic_lora.sh
# Expected: trainer.save_freq=1 \
```

- [ ] **Step 2: Wipe the smoke ckpt dir to avoid silent auto-resume**

```bash
rm -rf /weka/scratch/lshi40_llm/mallm/SRPO/checkpoints/srpo_main_q05b_critic_lora_smoke
```

- [ ] **Step 3: Run the smoke to completion**

```bash
bash /tmp/smoke_srpo_critic_lora.sh
```

Expected:
- Exit code `0`
- Log line `global_steps: 1`
- Log line containing `save_checkpoint` for the critic
- Wandb run created at `https://wandb.ai/chengruiqu-caltech/srpo_experiments/runs/<id>`

- [ ] **Step 4: Verify the critic checkpoint files landed on disk**

```bash
ls -la /weka/scratch/lshi40_llm/mallm/SRPO/checkpoints/srpo_main_q05b_critic_lora_smoke/global_step_1/
ls -la /weka/scratch/lshi40_llm/mallm/SRPO/checkpoints/srpo_main_q05b_critic_lora_smoke/global_step_1/critic/
```

Expected: at minimum a `model_world_size_*_rank_*.pt` shard or PEFT `adapter_config.json` + `adapter_model.safetensors` under `critic/`.

If this fails, do NOT proceed to Task 1 — revisit the spec §"Failure modes" Critic LoRA row and consider the `verl/workers/fsdp_workers.py` patch from `project_critic_lora_target_modules_collision.md` step 2.

- [ ] **Step 5: Update the memory with verified status**

Edit `~/.claude/projects/-weka-scratch-lshi40-llm-mallm-SRPO/memory/project_critic_lora_target_modules_collision.md` — replace the "Final critic ckpt save with PEFT was NOT exercised" sentence with a line documenting that ckpt save is now verified end-to-end on `<date>` with the explicit Qwen module list.

- [ ] **Step 6: No commit** — `/tmp/smoke_srpo_critic_lora.sh` lives outside the repo.

---

## Task 1: Create `train.slurm`

Build the parameterized script from `train_q05b.slurm:1-249` plus the spec's `case "$MODEL"` block, critic LoRA hydra args, and parameterized `--nodes` / `--gpus-per-node` reading from SLURM env.

**Files:**
- Create: `/weka/scratch/lshi40_llm/mallm/SRPO/train.slurm`

- [ ] **Step 1: Copy `train_q05b.slurm` as the starting point**

```bash
cp /weka/scratch/lshi40_llm/mallm/SRPO/train_q05b.slurm /weka/scratch/lshi40_llm/mallm/SRPO/train.slurm
```

- [ ] **Step 2: Update the SBATCH header to be model-agnostic**

In `train.slurm` lines 16-28, replace:

```
#SBATCH --job-name=mappo_q05b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --partition=nvl,h100
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.output
#SBATCH --error=logs/%x_%j.error
#SBATCH --export=ALL,ROCR_VISIBLE_DEVICES=
#SBATCH --exclude=l02,l04,l05,l06
```

with:

```
#SBATCH --job-name=mappo
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --partition=nvl,h100
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.output
#SBATCH --error=logs/%x_%j.error
#SBATCH --export=ALL,ROCR_VISIBLE_DEVICES=
#SBATCH --exclude=l02,l04,l05,l06
```

The `--nodes`, `--gres`, and `--gpus-per-node` directives are removed because `submit_all.sh` passes them at submission time per-model (different sizes for q05b/q06b vs. q3b/q4b).

- [ ] **Step 3: Update the usage banner at the top**

Replace the comment block at lines 1-14 with:

```
#!/bin/bash
# Usage:
#   METHOD=ippo MODEL=q05b SEED=1 sbatch --nodes=2 --gpus-per-node=1 train.slurm
#   METHOD=srpo_main MODEL=q3b SEED=2 sbatch --nodes=2 --gpus-per-node=2 train.slurm
#
# METHOD: ippo (default) | srpo_main | srpo_reward_only
# MODEL : q05b (default) | q06b | q3b | q4b
# SEED  : integer (default 42)
#
# Usually invoked via submit_all.sh which loops over the full sweep.
#
# Cross-pair probe is OFFLINE (separate invocation on saved checkpoints):
#   python -m verl.trainer.main_mappo \
#     +multi_agent.cross_pair_probe.run_only=true \
#     multi_agent.cross_pair_probe.partner_ckpt_dir=<sibling-arm>/global_step_N \
#     multi_agent.cross_pair_probe.out_path=logs/probe_<method>_step_N.json \
#     trainer.resume_mode=resume_path \
#     trainer.resume_from_path=<this-arm>/global_step_N \
#     <all other config args from below...>
```

- [ ] **Step 4: Add MODEL/SEED env defaults next to METHOD**

In the script, find the line `METHOD=${METHOD:-ippo}` and replace it with:

```bash
# Sweep coordinates
METHOD=${METHOD:-ippo}
MODEL=${MODEL:-q05b}
SEED=${SEED:-42}
```

- [ ] **Step 5: Replace the hardcoded `MODEL_NAME` block with the per-model `case` block**

Find:

```
CKPT_DIR=${REPO_DIR}/checkpoints/${METHOD}_q05b
DATA_DIR=${REPO_DIR}/data/gsm8k
GSM8K_TRAIN=${DATA_DIR}/train.parquet
GSM8K_TEST=${DATA_DIR}/test.parquet
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
```

and replace with:

```bash
DATA_DIR=${REPO_DIR}/data/gsm8k
GSM8K_TRAIN=${DATA_DIR}/train.parquet
GSM8K_TEST=${DATA_DIR}/test.parquet

# Per-model sizing — chosen to fit LoRA-actor + LoRA-critic + colocated vLLM
# on 80 GB H100 with margin. Total GPUs = NNODES * GPUS_PER_NODE.
case "$MODEL" in
    q05b)
        MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
        TOTAL_STEPS=250
        NNODES=2; GPUS_PER_NODE=1; TP_SIZE=1
        MICRO_BSZ=32; GMU=0.6
        ;;
    q06b)
        MODEL_NAME="Qwen/Qwen3-0.6B"
        TOTAL_STEPS=250
        NNODES=2; GPUS_PER_NODE=1; TP_SIZE=1
        MICRO_BSZ=32; GMU=0.6
        ;;
    q3b)
        MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
        TOTAL_STEPS=250
        NNODES=2; GPUS_PER_NODE=2; TP_SIZE=2
        MICRO_BSZ=16; GMU=0.5
        ;;
    q4b)
        MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
        TOTAL_STEPS=250
        NNODES=2; GPUS_PER_NODE=2; TP_SIZE=2
        MICRO_BSZ=8; GMU=0.5
        ;;
    *)
        echo "[ERROR] Unknown MODEL=${MODEL}; expected one of q05b q06b q3b q4b" >&2
        exit 2
        ;;
esac

CKPT_DIR=${REPO_DIR}/checkpoints/${METHOD}_${MODEL}_seed${SEED}
```

- [ ] **Step 6: Replace the hardcoded `GPUS_PER_NODE=1` line**

In the Ray-cluster section (around line 86-88), find:

```
NUM_NODES=${SLURM_JOB_NUM_NODES:-2}
GPUS_PER_NODE=1
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
```

and replace with:

```bash
# NUM_NODES and GPUS_PER_NODE come from SLURM at runtime; fall back to the
# per-model defaults from the case block above when running outside slurm.
NUM_NODES=${SLURM_JOB_NUM_NODES:-${NNODES}}
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-${GPUS_PER_NODE}}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
```

- [ ] **Step 7: Update the `case "$METHOD"` block to drop `_q05b` from `EXP_NAME`**

In the existing `case "$METHOD"` block, change every `EXP_NAME="<method>_q05b"` line to use the seed-scoped name:

```bash
case "$METHOD" in
    srpo_main)
        TRAINER_TYPE="risk_averse"
        KL_COEF=0.1          # 1/tau, tau=10
        EXP_NAME="srpo_main_${MODEL}_seed${SEED}"
        ENTROPY_OVERRIDE="+multi_agent.agents.0.actor.actor.entropy_coeff=0.05"
        ADV_KL_TO_HERO=true
        ;;
    srpo_reward_only)
        TRAINER_TYPE="risk_averse"
        KL_COEF=0.1
        EXP_NAME="srpo_reward_only_${MODEL}_seed${SEED}"
        ENTROPY_OVERRIDE=""
        ADV_KL_TO_HERO=true
        ;;
    ippo|*)
        TRAINER_TYPE="mappo"
        KL_COEF=0.001        # standard PPO default (not applied to rewards by default)
        EXP_NAME="ippo_${MODEL}_seed${SEED}"
        ENTROPY_OVERRIDE=""
        ADV_KL_TO_HERO=false
        ;;
esac
```

- [ ] **Step 8: Parameterize the hydra invocation (the long `srun` block)**

In the final `srun ... python -m verl.trainer.main_mappo ...` block, change:

  - `data.seed=42` → `data.seed=${SEED}`
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32` → `=${MICRO_BSZ}`
  - `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32` → `=${MICRO_BSZ}`
  - `actor_rollout_ref.rollout.tensor_model_parallel_size=1` → `=${TP_SIZE}`
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.6` → `=${GMU}`
  - `critic.ppo_micro_batch_size_per_gpu=32` → `=${MICRO_BSZ}`
  - `trainer.total_training_steps=200` → `=${TOTAL_STEPS}`

Then **insert** the critic LoRA args after the `critic.model.path=${MODEL_NAME} \` line and before `critic.optim.lr=1e-5 \`:

```
        critic.model.lora_rank=32 \
        critic.model.lora_alpha=32 \
        "critic.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
```

Finally, add `multi_agent.num_rounds=3 \` somewhere in the multi_agent block (pin explicitly so silent config drift can't break the experiment).

The complete final hydra block should look like:

```bash
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "unset ROCR_VISIBLE_DEVICES; \
        python -m verl.trainer.main_mappo \
        data.train_files=${GSM8K_TRAIN} \
        data.val_files=${GSM8K_TEST} \
        data.train_batch_size=256 \
        data.max_prompt_length=2048 \
        data.max_response_length=512 \
        data.seed=${SEED} \
        multi_agent.trainer_type=${TRAINER_TYPE} \
        multi_agent.adversary_kl_to_hero=${ADV_KL_TO_HERO} \
        multi_agent.num_rounds=3 \
        multi_agent.agents.0.actor.model.path=${MODEL_NAME} \
        multi_agent.agents.1.actor.model.path=${MODEL_NAME} \
        multi_agent.agents.0.n_gpus_per_node=${GPUS_PER_NODE} \
        multi_agent.agents.1.n_gpus_per_node=${GPUS_PER_NODE} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ} \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
        actor_rollout_ref.rollout.gpu_memory_utilization=${GMU} \
        actor_rollout_ref.rollout.max_model_len=2560 \
        actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        actor_rollout_ref.rollout.enforce_eager=true \
        actor_rollout_ref.rollout.free_cache_engine=false \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=NONE \
        actor_rollout_ref.model.path=${MODEL_NAME} \
        actor_rollout_ref.model.lora_rank=32 \
        actor_rollout_ref.model.lora_alpha=32 \
        actor_rollout_ref.model.target_modules=all-linear \
        critic.model.path=${MODEL_NAME} \
        critic.model.lora_rank=32 \
        critic.model.lora_alpha=32 \
        \"critic.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]\" \
        critic.optim.lr=1e-5 \
        critic.ppo_micro_batch_size_per_gpu=${MICRO_BSZ} \
        algorithm.kl_ctrl.kl_coef=${KL_COEF} \
        trainer.total_training_steps=${TOTAL_STEPS} \
        trainer.test_freq=20 \
        trainer.save_freq=50 \
        trainer.resume_mode=auto \
        trainer.nnodes=${NUM_NODES} \
        trainer.n_gpus_per_node=${GPUS_PER_NODE} \
        trainer.default_local_dir=${CKPT_DIR} \
        trainer.project_name=srpo_experiments \
        trainer.experiment_name=${EXP_NAME} \
        trainer.logger=[console,wandb] \
        trainer.val_before_train=false \
        ${ENTROPY_OVERRIDE}" \
    2>&1 | tee "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
```

Note: the inner double-quotes around the `target_modules=[...]` argument must be escaped (`\"`) because the whole hydra invocation lives inside a `bash -c "..."` heredoc.

- [ ] **Step 9: Add the critic-LoRA-collision warning comment**

Right above the `critic.model.lora_rank=32 \` line in the hydra block, insert (inside the `bash -c "..."` string, indented to match):

```
        # NOTE: target_modules MUST be the explicit Qwen list, not all-linear.
        # all-linear collides with PEFT's TaskType.TOKEN_CLS modules_to_save=[score]
        # — see memory project_critic_lora_target_modules_collision.md.
```

(Bash `#` inside a `-c` double-quoted string is a literal `#`, not a comment, but `srun bash -c` will pass it to the inner shell where `#` after a space is parsed as a comment. Verify with the dry-print in Step 10.)

- [ ] **Step 10: Dry-print to verify variable expansion**

Run a no-execute parse of the script with stub env vars:

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
METHOD=srpo_main MODEL=q3b SEED=7 bash -n train.slurm
echo "Exit: $?"
```

Expected: exit `0` (syntax check passes; `-n` does not execute).

Then verify the case-block expansion:

```bash
METHOD=srpo_main MODEL=q3b SEED=7 bash -c '
    METHOD=${METHOD:-ippo}; MODEL=${MODEL:-q05b}; SEED=${SEED:-42}
    case "$MODEL" in
        q05b) MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"; NNODES=2; GPUS_PER_NODE=1; TP_SIZE=1; MICRO_BSZ=32; GMU=0.6 ;;
        q06b) MODEL_NAME="Qwen/Qwen3-0.6B"; NNODES=2; GPUS_PER_NODE=1; TP_SIZE=1; MICRO_BSZ=32; GMU=0.6 ;;
        q3b)  MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"; NNODES=2; GPUS_PER_NODE=2; TP_SIZE=2; MICRO_BSZ=16; GMU=0.5 ;;
        q4b)  MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"; NNODES=2; GPUS_PER_NODE=2; TP_SIZE=2; MICRO_BSZ=8; GMU=0.5 ;;
    esac
    echo "MODEL_NAME=$MODEL_NAME NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE TP_SIZE=$TP_SIZE MICRO_BSZ=$MICRO_BSZ GMU=$GMU"
'
```

Expected: `MODEL_NAME=Qwen/Qwen2.5-3B-Instruct NNODES=2 GPUS_PER_NODE=2 TP_SIZE=2 MICRO_BSZ=16 GMU=0.5`

- [ ] **Step 11: Stage the file (do not commit yet)**

```bash
git -C /weka/scratch/lshi40_llm/mallm/SRPO add train.slurm
git -C /weka/scratch/lshi40_llm/mallm/SRPO status
```

Expected: `train.slurm` shown under "Changes to be committed". No commit yet — Task 5 covers the combined commit.

---

## Task 2: Local 1-step IPPO smoke of `train.slurm` on 2× L40S

`train.slurm` cannot be `sbatch`-tested on the compute node. Instead, run a python invocation that mirrors its hydra args, with the standard L40S sizing overrides from `project_l40s_sizing_for_h100_configs.md`. Goal: prove the new variable wiring (MODEL=q05b → MODEL_NAME, MICRO_BSZ, GMU, etc.) and the critic LoRA hydra args produce a functioning IPPO run.

**Files:**
- Create: `/tmp/smoke_train_slurm_ippo_q05b.sh` (ephemeral; not in repo)
- Read: log under `${REPO_DIR}/logs/`

- [ ] **Step 1: Write the L40S smoke script**

Create `/tmp/smoke_train_slurm_ippo_q05b.sh` with:

```bash
#!/bin/bash
# Local L40S smoke of train.slurm METHOD=ippo MODEL=q05b for 1 step.
# Mirrors train.slurm's hydra block but with L40S sizing overrides.
set -x
set -o pipefail

REPO_DIR=/weka/scratch/lshi40_llm/mallm/SRPO
CKPT_DIR=${REPO_DIR}/checkpoints/ippo_q05b_seed1_smoke_l40s
DATA_DIR=${REPO_DIR}/data/gsm8k
GSM8K_TRAIN=${DATA_DIR}/train.parquet
GSM8K_TEST=${DATA_DIR}/test.parquet
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

mkdir -p "${REPO_DIR}/logs" "${CKPT_DIR}"
rm -rf "${CKPT_DIR}"  # smoke: no auto-resume from prior runs

module reset >/dev/null 2>&1 || true
module load gcc/9.3.0 anaconda3/2024.02-1 >/dev/null 2>&1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate srpo

cd "${REPO_DIR}"
ray stop -f 2>&1 | tail -3 || true

unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=2404c5a258f9f0623cde0ef3041df0a8bc425bb6
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HYDRA_FULL_ERROR=1

NUM_NODES=2; GPUS_PER_NODE=1; CPUS_PER_TASK=8

ray start --head --node-ip-address=127.0.0.1 --port=6380 \
    --num-cpus=$((NUM_NODES * CPUS_PER_TASK)) \
    --num-gpus=$((NUM_NODES * GPUS_PER_NODE)) \
    --include-dashboard=false 2>&1 | tail -20
for i in 1 2 3 4 5; do ray status >/dev/null 2>&1 && break; sleep 2; done
ray status

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/smoke_train_slurm_ippo_q05b_${TS}.log"
echo "[INFO] Logging to ${LOG}"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_mappo \
    data.train_files=${GSM8K_TRAIN} \
    data.val_files=${GSM8K_TEST} \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.seed=1 \
    multi_agent.trainer_type=mappo \
    multi_agent.adversary_kl_to_hero=false \
    multi_agent.num_rounds=3 \
    multi_agent.agents.0.actor.model.path=${MODEL_NAME} \
    multi_agent.agents.1.actor.model.path=${MODEL_NAME} \
    multi_agent.agents.0.n_gpus_per_node=${GPUS_PER_NODE} \
    multi_agent.agents.1.n_gpus_per_node=${GPUS_PER_NODE} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_model_len=2560 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=false \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=NONE \
    actor_rollout_ref.model.path=${MODEL_NAME} \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    critic.model.path=${MODEL_NAME} \
    critic.model.lora_rank=32 \
    critic.model.lora_alpha=32 \
    "critic.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.total_training_steps=1 \
    trainer.test_freq=10 \
    trainer.save_freq=1 \
    trainer.resume_mode=disable \
    trainer.nnodes=${NUM_NODES} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.default_local_dir=${CKPT_DIR} \
    trainer.project_name=srpo_experiments \
    trainer.experiment_name=ippo_q05b_seed1_smoke_l40s \
    trainer.logger=[console,wandb] \
    trainer.val_before_train=false \
    2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}
echo "[INFO] Exit code: ${RC}"
ray stop -f 2>&1 | tail -3 || true
exit ${RC}
```

```bash
chmod +x /tmp/smoke_train_slurm_ippo_q05b.sh
```

- [ ] **Step 2: Run the smoke**

```bash
bash /tmp/smoke_train_slurm_ippo_q05b.sh
```

Expected:
- Exit `0`
- `global_steps: 1` in log
- Critic checkpoint files under `checkpoints/ippo_q05b_seed1_smoke_l40s/global_step_1/critic/`
- Wandb run created

If OOM at actor backward: confirm `gpu_memory_utilization=0.4` and `ppo_micro_batch_size_per_gpu=8` are set per `project_l40s_sizing_for_h100_configs.md`.

If `TypeError: modules_to_save cannot be applied`: the critic `target_modules` got reset to `all-linear`; verify Step 1 has the explicit Qwen list.

---

## Task 3: Local 1-step SRPO smoke of `train.slurm` on 2× L40S

Same as Task 2 but with the SRPO trainer wiring (TRAINER_TYPE=risk_averse, ADV_KL_TO_HERO=true, KL_COEF=0.1, entropy override on agent 0). Confirms the `case "$METHOD"` branch flips correctly.

**Files:**
- Create: `/tmp/smoke_train_slurm_srpo_main_q05b.sh` (ephemeral; not in repo)

- [ ] **Step 1: Copy + edit the IPPO smoke**

```bash
cp /tmp/smoke_train_slurm_ippo_q05b.sh /tmp/smoke_train_slurm_srpo_main_q05b.sh
sed -i 's|ippo_q05b_seed1_smoke_l40s|srpo_main_q05b_seed1_smoke_l40s|g' /tmp/smoke_train_slurm_srpo_main_q05b.sh
sed -i 's|multi_agent.trainer_type=mappo|multi_agent.trainer_type=risk_averse|' /tmp/smoke_train_slurm_srpo_main_q05b.sh
sed -i 's|multi_agent.adversary_kl_to_hero=false|multi_agent.adversary_kl_to_hero=true|' /tmp/smoke_train_slurm_srpo_main_q05b.sh
sed -i 's|algorithm.kl_ctrl.kl_coef=0.001|algorithm.kl_ctrl.kl_coef=0.1|' /tmp/smoke_train_slurm_srpo_main_q05b.sh
```

Then add the entropy override at the end of the hydra invocation (just before the final `2>&1 | tee` line). Edit the file to insert this line:

```
    +multi_agent.agents.0.actor.actor.entropy_coeff=0.05 \
```

- [ ] **Step 2: Run the smoke**

```bash
bash /tmp/smoke_train_slurm_srpo_main_q05b.sh
```

Expected:
- Exit `0`
- `global_steps: 1` in log
- Critic checkpoint files under `checkpoints/srpo_main_q05b_seed1_smoke_l40s/global_step_1/critic/`
- Wandb run created (look for `back_propogate_reward` in log to confirm SRPO branch hit)

---

## Task 4: Create `submit_all.sh`

**Files:**
- Create: `/weka/scratch/lshi40_llm/mallm/SRPO/submit_all.sh`

- [ ] **Step 1: Write the file**

```bash
cat > /weka/scratch/lshi40_llm/mallm/SRPO/submit_all.sh <<'EOF'
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
EOF
chmod +x /weka/scratch/lshi40_llm/mallm/SRPO/submit_all.sh
```

- [ ] **Step 2: Dry-run to verify command shape**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
DRY_RUN=1 bash submit_all.sh | head -30
```

Expected output (24 lines total; first six lines):

```
[DRY-RUN] sbatch --nodes=2 --gpus-per-node=1 --gres=gpu:1 --job-name=mappo_ippo_q05b_s1 train.slurm
[DRY-RUN] sbatch --nodes=2 --gpus-per-node=1 --gres=gpu:1 --job-name=mappo_ippo_q05b_s2 train.slurm
[DRY-RUN] sbatch --nodes=2 --gpus-per-node=1 --gres=gpu:1 --job-name=mappo_ippo_q05b_s3 train.slurm
[DRY-RUN] sbatch --nodes=2 --gpus-per-node=1 --gres=gpu:1 --job-name=mappo_srpo_main_q05b_s1 train.slurm
[DRY-RUN] sbatch --nodes=2 --gpus-per-node=1 --gres=gpu:1 --job-name=mappo_srpo_main_q05b_s2 train.slurm
[DRY-RUN] sbatch --nodes=2 --gpus-per-node=1 --gres=gpu:1 --job-name=mappo_srpo_main_q05b_s3 train.slurm
```

Then verify Q3B/Q4B lines have `--gpus-per-node=2 --gres=gpu:2`:

```bash
DRY_RUN=1 bash submit_all.sh | grep "q3b\|q4b" | head -3
```

Expected:

```
[DRY-RUN] sbatch --nodes=2 --gpus-per-node=2 --gres=gpu:2 --job-name=mappo_ippo_q3b_s1 train.slurm
```

- [ ] **Step 3: Verify env-override paths**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
DRY_RUN=1 MODELS=q4b SEEDS=42 METHODS=srpo_main bash submit_all.sh
```

Expected: exactly one line:

```
[DRY-RUN] sbatch --nodes=2 --gpus-per-node=2 --gres=gpu:2 --job-name=mappo_srpo_main_q4b_s42 train.slurm
```

- [ ] **Step 4: Verify total job count**

```bash
DRY_RUN=1 bash /weka/scratch/lshi40_llm/mallm/SRPO/submit_all.sh | wc -l
```

Expected: `24` (4 models × 2 methods × 3 seeds).

- [ ] **Step 5: Stage the file**

```bash
git -C /weka/scratch/lshi40_llm/mallm/SRPO add submit_all.sh
```

---

## Task 5: Commit `train.slurm` + `submit_all.sh` + spec

**Files:**
- Modify: git index — commit the staged spec, plan, train.slurm, submit_all.sh

- [ ] **Step 1: Stage the spec and plan if not already staged**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
git add docs/superpowers/specs/2026-04-26-training-script-parameterization-design.md
git add docs/superpowers/plans/2026-04-27-training-script-parameterization.md
git status
```

Expected `Changes to be committed`:
- `docs/superpowers/plans/2026-04-27-training-script-parameterization.md` (new)
- `docs/superpowers/specs/2026-04-26-training-script-parameterization-design.md` (new)
- `submit_all.sh` (new)
- `train.slurm` (new)

- [ ] **Step 2: Commit**

```bash
git -C /weka/scratch/lshi40_llm/mallm/SRPO commit -m "$(cat <<'EOF'
feat(slurm): parameterized train.slurm + submit_all.sh for IPPO/SRPO sweep

Replaces train_q05b.slurm-style per-model scripts with a single
parameterized SLURM script driven by METHOD/MODEL/SEED env vars,
and a one-shot submission helper that loops over the
4-model x 2-method x 3-seed sweep.

Critic uses LoRA with explicit Qwen attention/MLP target_modules
(NOT all-linear, which collides with PEFT's TaskType.TOKEN_CLS
modules_to_save=[score]).

train_q05b.slurm kept as historical reference per spec.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify commit**

```bash
git -C /weka/scratch/lshi40_llm/mallm/SRPO log -1 --stat
```

Expected: 4 files changed (spec, plan, train.slurm, submit_all.sh).

---

## Task 6: First real-cluster validation — submit one Q0.5B IPPO seed

**This task requires SLURM access (NOT runnable from the L40S compute node).** It is the first end-to-end validation of `train.slurm` against the real H100/NVL partition. Submit only one job (smallest model, one seed) before authorizing the full sweep.

**Files:**
- Read: `logs/mappo_ippo_q05b_s1_<jobid>.output` after submission

- [ ] **Step 1: From a SLURM-capable login node, submit the smallest single arm**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
METHODS=ippo MODELS=q05b SEEDS=1 bash submit_all.sh
```

Expected: one `Submitted batch job <jobid>` line.

- [ ] **Step 2: Watch the queue until the job starts**

```bash
squeue -u $USER -n mappo_ippo_q05b_s1
```

Expected: ST=R within a few minutes once nvl/h100 has 2 nodes free.

- [ ] **Step 3: Monitor the log for early failure signatures**

```bash
JOBID=<id from step 1>
tail -F logs/mappo_ippo_q05b_s1_${JOBID}.output | grep -E "global_steps|Traceback|Error|FAILED|OOM|Killed|elapsed_steps|save_checkpoint"
```

Expected:
- Within ~5 min: Ray cluster registration logs, vLLM engine startup
- Within ~10 min: first `global_steps: 1` line
- At step 50: `save_checkpoint` line for `checkpoints/ippo_q05b_seed1/global_step_50/`
- No `Traceback` or `OOM`

If a `target_modules` TypeError appears: you regressed Task 1 Step 8 — the critic `target_modules` reverted to `all-linear`.

- [ ] **Step 4: Once `global_steps: 50` lands and ckpt saves, cancel and authorize the full sweep**

```bash
scancel $JOBID  # optional — let it run to completion if budget allows
```

- [ ] **Step 5: (Optional) Submit the full sweep**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
bash submit_all.sh
squeue -u $USER -n mappo
```

Expected: 24 jobs queued.

---

## Self-review

- **Spec coverage:** every spec section maps to a task — `case "$MODEL"` block (Task 1 Step 5), `case "$METHOD"` block (Task 1 Step 7), critic LoRA hydra args (Task 1 Step 8), submit_all.sh (Task 4), validation gap closure (Task 0).
- **Placeholder scan:** zero "TBD"/"appropriate"/"as needed" placeholders; every code step shows complete code; expected outputs given for every command.
- **Type/name consistency:** `MICRO_BSZ`, `GMU`, `TP_SIZE`, `NNODES`, `GPUS_PER_NODE`, `TOTAL_STEPS`, `MODEL_NAME`, `EXP_NAME`, `CKPT_DIR` are used consistently across Tasks 1-3; `submit_all.sh` job-name matches the SBATCH `--job-name=mappo` from Task 1 Step 2 (overridden per-job by Task 4 Step 1's `--job-name=mappo_<method>_<model>_s<seed>`).
- **Pitfalls covered:** L40S sizing (Tasks 2-3 use the documented overrides), PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 (not expandable_segments), critic LoRA target_modules collision (Step 8 + warning comment), resume_mode=disable for smokes vs. =auto for production, `--export=ALL,ROCR_VISIBLE_DEVICES=` carried over.
