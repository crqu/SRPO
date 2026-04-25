# SRPO — Risk-Averse Multi-Agent PPO for LLMs

This repo backs the empirical claim in [arXiv:2602.21515](https://arxiv.org/html/2602.21515): **SRPO produces policies that generalize across partners better than IPPO** in multi-agent LLM training. Concretely, an adversary agent regularized toward the hero's policy via `(1/τ) · KL(π_adv ‖ π_hero)` finds harder collaborative settings, and training the hero against this adversary yields a partner-robust policy.

The codebase is a fork of [verl](https://github.com/volcengine/verl) extended with multi-agent rollout (`RayMAPPOTrainer`) and the SRPO adversarial-regularizer trainer (`RayRiskAverseTrainer`).

## What's where

| | path |
|---|---|
| Entry point | `verl/trainer/main_mappo.py` (Hydra; `multi_agent.trainer_type` chooses IPPO vs SRPO) |
| Core algorithm | `verl/trainer/ppo/mappo_trainer.py` |
| IPPO baseline | `RayMAPPOTrainer` |
| SRPO algorithm | `RayRiskAverseTrainer(RayMAPPOTrainer)` |
| Default config | `verl/trainer/config/mappo_trainer.yaml` |
| Multi-node slurm launch | `train_q05b.slurm` |
| Single-node 2-GPU debug | `debug_q05b_local.sh` |

`RayRiskAverseTrainer` overrides only what differs from IPPO:
- `back_propogate_reward` — propagates `−hero_return` to the adversary across rounds.
- `_apply_kl_penalty` — runs an FSDP forward of the hero on the adversary's rollout to compute `log π_hero(t_adv | s_adv)`, then subtracts `(1/τ) · KL(π_adv ‖ π_hero)` from the adversary's per-token reward.
- `_build_input_ids_from_histories` — substitutes a shared discussion prompt across agents.

The actor / critic / rollout stacks are shared with IPPO so any IPPO-vs-SRPO comparison is genuinely matched.

## Quickstart

### Prerequisites

```bash
module load anaconda3/2024.02-1
conda activate srpo
```

### Single-node 2-GPU debug

```bash
bash debug_q05b_local.sh             # SRPO (default in this script)
METHOD=ippo bash debug_q05b_local.sh # IPPO
```

The debug script trains 2 steps on GSM8K with Qwen2.5-0.5B-Instruct + LoRA, validates, and saves a checkpoint — useful for end-to-end smoke tests in ~5 minutes.

### Multi-node slurm

```bash
sbatch train_q05b.slurm              # IPPO (default)
METHOD=srpo sbatch train_q05b.slurm  # SRPO
```

Two nodes, one GPU per node, one agent per node. Adjust `GPUS_PER_NODE` / `--nodes` for larger runs.

## Configuring SRPO

Two knobs control the adversary's regularizer:

```yaml
multi_agent:
  adversary_kl_to_hero: true   # turn on the SRPO KL term
algorithm:
  kl_ctrl:
    kl_coef: 0.1               # = 1/τ in the paper; smaller -> weaker regularization
```

When `adversary_kl_to_hero=false`, the adversary trains on raw `−hero_return` with no KL — useful as an SRPO ablation. With it `true`, the trainer runs an extra FSDP forward of the hero per round; this is the dominant cost of SRPO over IPPO.

If you train with HYBRID rollout + LoRA, also set `actor_rollout_ref.rollout.free_cache_engine=false` to avoid vLLM's cumem sleep/wake state machine on small models — it's exposed by tight inter-round sleep cycles and not worth the GPU savings on Qwen-0.5B-class workloads.

## Reproducing the paper claim (work-in-progress)

The training setup in `train_q05b.slurm` is a starting point, not the final recipe. Several knobs still need tuning to make the SRPO > IPPO partner-generalization gap robust:

1. **Discussion prompt.** Currently a fixed string (`multi_agent.discussion_prompt`). The prompt shapes the state distribution the adversary explores; treat it as part of the algorithm and sweep it.
2. **τ schedule.** Fixed `1/τ = 0.1` is a reasonable starting point; an annealing schedule may help stability.
3. **num_rounds** vs **train_batch_size** trade-off: more rounds give the adversary more leverage but multiply rollout cost; partner-generalization metrics may be more sensitive to this than to step count.
4. **Partner distribution at evaluation.** A clear test of "partner generalization" needs a held-out set of partners that neither IPPO nor SRPO trained against — that pipeline is not in-tree yet.

Treat published numbers as preliminary until those are nailed down.

## Roadmap

- **GRPO-style trainer.** Replace the value-function critic with group-relative advantages (à la GRPO). Would subclass `RayMAPPOTrainer` similarly to how `RayRiskAverseTrainer` does today; the SRPO adversary regularizer is orthogonal and should compose.
- **Held-out partner evaluation harness.** A separate eval pipeline that loads N held-out partner checkpoints and reports per-partner success rates — needed to make the partner-generalization claim land cleanly.
- **Discussion-prompt sweep.** Treat the prompt as a hyperparameter; report SRPO-vs-IPPO gap as a function of prompt design.

## Notes

- vLLM 0.11 + LoRA + cumem sleep mode has known issues with rapid sleep/wake cycles. Default debug config disables `free_cache_engine` to sidestep them.
- For deeper debugging, `CUDA_LAUNCH_BLOCKING=1` (already on in `debug_q05b_local.sh`) makes async CUDA errors surface at the actual call site.
