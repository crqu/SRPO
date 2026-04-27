# Training script parameterization for IPPO/SRPO across 4 Qwen models

## Goal

Generate a single SLURM training script that supports the full SRPO experiment matrix from `docs/experiment_design.md`:

- **4 base models:** Qwen2.5-0.5B-Instruct (`q05b`), Qwen3-0.6B (`q06b`), Qwen2.5-3B-Instruct (`q3b`), Qwen3-4B-Instruct-2507 (`q4b`)
- **2 (+1) trainer methods:** `ippo`, `srpo_main` (entropy-asymmetric SRPO), `srpo_reward_only` (ablation)
- **≥3 seeds per (model, method)** → up to **24 training runs** for the 2-method core

Every arm must finish on h100/nvl partitions within the 24h SLURM cap with GPU sizing chosen so neither vLLM rollout nor FSDP actor/critic OOMs on the 80 GB H100.

## Non-goals

- **Cross-pair probe execution.** The training script's role ends at saving checkpoints. The header comment from the existing `train_q05b.slurm:7-14` documents the offline `cross_pair_probe.run_only=true` invocation; that stays an out-of-band command.
- **Untuned Llama partner evaluation.** Eval-only, runs on saved checkpoints, separate workflow.
- **Model download orchestration.** First job per model triggers the HF download via `transformers.AutoConfig`; we don't pre-prime the cache.
- **SLURM job-array support.** Single-seed-per-`sbatch` invocation only; the submission helper handles the for-loop over seeds.

## Decisions made during brainstorming

| # | Decision | Rationale |
|---|---|---|
| 1 | One parameterized `train.slurm` (replaces `train_q05b.slurm`); `case "$MODEL"` block for sizing, `case "$METHOD"` block for trainer wiring | Avoids drift across 4 per-model scripts; the 8 arms differ in only ~5 knobs |
| 2 | LoRA critic for all 4 models | Keeps GPU counts low; preserves matched-stack constraint from `CLAUDE.md` between IPPO/SRPO |
| 3 | 250 training steps for all 4 models | User override of paper's per-model anchors; flat budget simpler |
| 4 | Single seed per `sbatch` invocation, default `SEED=42`, scoped by `${METHOD}_${MODEL}_seed${SEED}` checkpoint dir | Matches existing `METHOD=...` env-var pattern; avoids slurm-array nuisance |
| 5 | GPU sizing per model: q05b/q06b → 2 GPUs total (1/agent, TP=1); q3b/q4b → 4 GPUs total (2/agent, TP=2) | LoRA actor + LoRA critic + colocated vLLM at `gmu=0.4-0.6` fits within 80 GB margin |
| 6 | Critic `target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` (NOT `all-linear`) | `all-linear` collides with PEFT's auto `modules_to_save=["score"]` for `TaskType.TOKEN_CLS` value head; verified empirically on Qwen2.5-0.5B (memory: `project_critic_lora_target_modules_collision.md`) |

## Architecture

Single bash script `train.slurm` at the repo root. Three top sections:

1. **SBATCH header** — minimum that's the same for all arms: partition, time, output dirs, exclude list, `--export=ALL,ROCR_VISIBLE_DEVICES=`. Per-arm `--nodes` and `--gpus-per-node` are passed at submission time via `sbatch --nodes=N --gpus-per-node=G` (avoids over-allocating GPUs for small models).
2. **Env + Ray cluster bootstrap** — carried over from `train_q05b.slurm:46-147` unchanged: conda activation, `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`, head/worker `srun`, registration polling.
3. **`case "$MODEL"` block** (new) followed by **`case "$METHOD"` block** (existing pattern, unchanged) → `python -m verl.trainer.main_mappo` with all hydra args.

Submission helper `submit_all.sh` issues 24 (or N×M×K) `sbatch` calls with the right `--nodes` / `--gpus-per-node` per model.

## Component details

### `case "$MODEL"` block (the new content)

```bash
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
```

### `case "$METHOD"` block (unchanged from current)

```bash
case "$METHOD" in
    srpo_main)
        TRAINER_TYPE="risk_averse"; KL_COEF=0.1
        ENTROPY_OVERRIDE="+multi_agent.agents.0.actor.actor.entropy_coeff=0.05"
        ADV_KL_TO_HERO=true ;;
    srpo_reward_only)
        TRAINER_TYPE="risk_averse"; KL_COEF=0.1
        ENTROPY_OVERRIDE=""; ADV_KL_TO_HERO=true ;;
    ippo|*)
        TRAINER_TYPE="mappo"; KL_COEF=0.001
        ENTROPY_OVERRIDE=""; ADV_KL_TO_HERO=false ;;
esac
```

### Hydra args common to all 24 arms

Carry over verbatim from `train_q05b.slurm:200-245`:

- `data.{train_files,val_files}` → GSM8K parquet paths
- `data.{train_batch_size=256, max_prompt_length=2048, max_response_length=512}`
- `actor_rollout_ref.actor.{optim.lr=1e-6, ppo_mini_batch_size=256}`
- `actor_rollout_ref.rollout.{name=vllm, max_model_len=2560, max_num_batched_tokens=8192, load_format=safetensors, calculate_log_probs=true, enforce_eager=true, free_cache_engine=false}`
- `+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=NONE` (vLLM LoRA cudagraph rank-mismatch lesson)
- `actor_rollout_ref.model.{lora_rank=32, lora_alpha=32, target_modules=all-linear}`
- `critic.optim.lr=1e-5`
- `multi_agent.num_rounds=3` *(YAML default already; pin explicitly so silent config drift can't break the experiment)*
- `trainer.{test_freq=20, save_freq=50, resume_mode=auto, val_before_train=false, project_name=srpo_experiments, logger=[console,wandb]}`

### Hydra args parameterized per (METHOD, MODEL, SEED)

```bash
data.seed=${SEED} \
multi_agent.trainer_type=${TRAINER_TYPE} \
multi_agent.adversary_kl_to_hero=${ADV_KL_TO_HERO} \
multi_agent.agents.0.actor.model.path=${MODEL_NAME} \
multi_agent.agents.1.actor.model.path=${MODEL_NAME} \
multi_agent.agents.0.n_gpus_per_node=${GPUS_PER_NODE} \
multi_agent.agents.1.n_gpus_per_node=${GPUS_PER_NODE} \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ} \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
actor_rollout_ref.rollout.gpu_memory_utilization=${GMU} \
actor_rollout_ref.model.path=${MODEL_NAME} \
critic.model.path=${MODEL_NAME} \
critic.model.lora_rank=32 \
critic.model.lora_alpha=32 \
"critic.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
critic.ppo_micro_batch_size_per_gpu=${MICRO_BSZ} \
algorithm.kl_ctrl.kl_coef=${KL_COEF} \
trainer.total_training_steps=${TOTAL_STEPS} \
trainer.nnodes=${NNODES} \
trainer.n_gpus_per_node=${GPUS_PER_NODE} \
trainer.default_local_dir=${REPO_DIR}/checkpoints/${METHOD}_${MODEL}_seed${SEED} \
trainer.experiment_name=${METHOD}_${MODEL}_seed${SEED} \
${ENTROPY_OVERRIDE}
```

The critic `target_modules` value MUST be the explicit list, not `all-linear`. A short `# NOTE: see project_critic_lora_target_modules_collision memory` comment will sit above this line in the script.

### Submission helper `submit_all.sh`

```bash
#!/bin/bash
# Submit the full sweep. Defaults: 4 models x 2 methods x 3 seeds = 24 jobs.
# Override via env: METHODS, MODELS, SEEDS (space-separated).
METHODS=${METHODS:-"ippo srpo_main"}
MODELS=${MODELS:-"q05b q06b q3b q4b"}
SEEDS=${SEEDS:-"1 2 3"}
for model in $MODELS; do
    case "$model" in
        q05b|q06b) NODES=2; GPN=1 ;;
        q3b|q4b)   NODES=2; GPN=2 ;;
        *) echo "Unknown MODEL=$model"; exit 2 ;;
    esac
    for method in $METHODS; do
        for seed in $SEEDS; do
            METHOD=$method MODEL=$model SEED=$seed \
                sbatch --nodes=$NODES --gpus-per-node=$GPN train.slurm
        done
    done
done
```

## Sizing summary

| Model | `--nodes` | `--gpus-per-node` | Total GPUs | TP | `MICRO_BSZ` | `GMU` | Est. wall time / 250 steps |
|---|---|---|---|---|---|---|---|
| Q0.5B | 2 | 1 | 2 | 1 | 32 | 0.6 | ~2 h |
| Q0.6B | 2 | 1 | 2 | 1 | 32 | 0.6 | ~2.5 h |
| Q3B | 2 | 2 | 4 | 2 | 16 | 0.5 | ~6-8 h |
| Q4B | 2 | 2 | 4 | 2 | 8 | 0.5 | ~10-14 h |

Total fleet cost for 24 runs (4 models × 2 methods × 3 seeds): ~300 GPU·hours.

## Failure modes and mitigations

| Risk | Mitigation |
|---|---|
| Critic LoRA crashes if someone "fixes" `target_modules` back to `all-linear` | Inline comment in `train.slurm` referencing the memory; design doc references the failure mode |
| `resume_mode=auto` silently no-ops on a stale ckpt dir | `${METHOD}_${MODEL}_seed${SEED}` checkpoint dir naming makes seeds isolated; if a job hits the 24h cap and retries, resume is intentional and works |
| Q4B at TP=2 hits a vLLM cumem inter-round wake/sleep bug (existing memory `project_mappo_colocated_interround_fix.md`) | First Q4B run treated as a smoke; if it crashes, fall back to TP=4 (4 GPUs/agent, 8 GPUs total) using `sbatch --gpus-per-node=4` override, no script change needed |
| Q0.5B baseline shifts vs. prior `train_q05b.slurm` numbers (because critic switched from full-FT to LoRA) | Acknowledged tradeoff per Q2 of brainstorming; existing `srpo_main_q05b` checkpoints stay valid as their own arm |
| Q0.6B/Q3B/Q4B not in HF cache → first job per model spends ~5-15 min downloading | Acceptable one-time cost; 24h slurm cap absorbs it |

## Validation evidence

- **SRPO pipeline already verified end-to-end** with critic full-FT on 2× L40S (`/tmp/smoke_srpo_l40s.sh`, exit 0, wandb run `mt58baz7`, 2 train steps + final GSM8K validation + ckpt save). The hydra wiring for `risk_averse` trainer + `adversary_kl_to_hero=true` is sound.
- **Critic LoRA target_modules validated** at the PEFT level: `all-linear` reproduces the `TypeError: modules_to_save cannot be applied to modules of type peft.tuners.lora.layer.Linear` at `peft/utils/other.py:297`; explicit Qwen module list works (3.4% trainable, value head + LoRA both `requires_grad=True`).
- **Critic LoRA at the verl integration level is fully validated** (2026-04-27 smoke on 2× L40S with explicit Qwen module list, `trainer.save_freq=1`, exit code 0): build OK, FSDP wrap OK, vLLM weight push OK, step 1 PPO update + critic backward OK, end-of-training validation OK, ckpt save OK. On disk: `checkpoints/.../global_step_1/critic/{0,1}/{model,optim,extra_state}_world_size_1_rank_0.pt` + `huggingface/` config. Asymmetry to flag: actor extracts a portable PEFT adapter under `actor/{0,1}/lora_adapter/`; critic does NOT (only FSDP shards). Resume works via FSDP; if anyone later wants a portable PEFT-only critic export, they will need to add it (out of scope for this design).

## Out of scope (for clarity)

- Code changes to `verl/workers/fsdp_workers.py` to add `exclude_modules` support to the critic LoraConfig. The config-only workaround removes the need.
- Changes to `train_q05b.slurm` itself. The new `train.slurm` supersedes it; the old script is **kept in place** as a historical reference (not deleted).
- Cron / auto-scheduling of jobs; the submission helper is one-shot.
