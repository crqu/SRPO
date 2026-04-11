# MAPPO Training Speedup Design

**Date:** 2026-04-11  
**Status:** Approved  
**Scope:** `verl/trainer/ppo/mappo_trainer.py`, `train_q05b.slurm`

---

## Problem

`train_q05b.slurm` (Qwen2.5-0.5B, 2×H100, 200 steps, 3 discussion rounds) takes ~9 hours. The design must also remain efficient for 4B models on the same hardware.

### Root cause analysis (from training log `mappo_q05b_1274980.output`)

| Phase | Mean time | Total (114 samples) |
|-------|-----------|---------------------|
| `gen` (vLLM rollout) | 18.7s | 2,126s |
| `old_log_prob` (FSDP fwd pass) | **30.7s** | **3,498s** |
| `values` (critic fwd) | 13.3s | 1,516s |
| `update_critic` | 54.1s | 6,163s |
| `update_actor` | 225s | 25,664s |

Actor MFU: **1.06%** (H100 baseline: 40–60%).  
Average sequence length: **~418 tokens** despite `max_response_length=512`.

Two distinct bottlenecks:

1. **Micro-batch size too small (Approach A):** `ppo_micro_batch_size_per_gpu=4` produces only ~1,672 tokens per backward pass. The actor spends most of its time in kernel launch overhead, not computation.

2. **Redundant FSDP forward pass (Approach B):** After vLLM generates sequences and already computes per-token log probs, a separate FSDP forward pass is made to recompute the same `old_log_probs`. This costs ~30s × 3 rounds = ~90s/step and scales with model size.

---

## Solution

Implement both Approach A and Approach B in sequence.

---

## Approach A: Hyperparameter fixes (slurm only, no code changes)

### Changes to `train_q05b.slurm`

| Parameter | Current | Proposed | Reason |
|-----------|---------|----------|--------|
| `actor.ppo_micro_batch_size_per_gpu` | 4 | 32 | 8× fewer backward passes; average 418-token sequences fit easily |
| `actor.ppo_mini_batch_size` | 64 | 256 | Single mini-batch per epoch; eliminates outer loop overhead |
| `critic.ppo_micro_batch_size_per_gpu` | 4 | 32 | Same reasoning as actor |
| `rollout.log_prob_micro_batch_size_per_gpu` | 8 | 32 | Faster fallback log_prob path |
| `rollout.gpu_memory_utilization` | 0.5 | 0.7 | Larger KV cache → faster vLLM generation |

**Memory safety:** Current max allocated is 11.2GB on an 80GB H100. vLLM at 0.7 utilization reserves ~56GB. Remaining ~24GB covers FSDP weights + LoRA optimizer states + activations for 32 sequences of ~418 avg tokens with gradient checkpointing enabled.

**Expected speedup:** 4–6× on actor update alone (1% → ~5-10% MFU).

---

## Approach B: Bypass FSDP log_prob recomputation

### Background

The existing `ray_trainer.py` already implements a bypass mode (`algorithm.rollout_correction.bypass_mode`) that reuses `rollout_log_probs` from the vLLM rollout instead of recomputing them via FSDP. `mappo_trainer.py` has not been updated to use this pattern.

### Data flow

```
generate_sequences() [agent_loop.py]
  └─ vLLM generates tokens
       └─ if calculate_log_probs=True:
            output.batch["rollout_log_probs"] = per-token logprobs   ← already computed
                │
                ▼ (batch.union at mappo_trainer.py:1428)
       batch.batch["rollout_log_probs"] available
                │
     CURRENT:   ▼ FSDP forward pass (30s)
           compute_log_prob()
           → batch.batch["old_log_probs"]
                │
     PROPOSED:  ▼ tensor assignment (0s)
           batch["old_log_probs"] = batch["rollout_log_probs"]
```

### Entropy correctness

The entropy regularization term in the PPO loss is **not** sourced from the `old_log_prob` phase. It is recomputed from scratch during each `update_actor` forward pass in `dp_actor.py:586–651`:

```python
calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)  # True
outputs = self._forward_micro_batch(model_inputs, calculate_entropy=calculate_entropy)
entropy = outputs["entropys"]        # from current π_θ
policy_loss -= entropy_agg * entropy_coeff   # loss term
```

The `compute_log_prob` phase (which bypass mode eliminates) produces entropy only for a monitoring metric (`actor/entropy`), which is redundantly recomputed during `update_actor` anyway.

**Bypass mode has no effect on the entropy loss term.**

### Changes

**1. `train_q05b.slurm`** — add one flag:
```bash
actor_rollout_ref.rollout.calculate_log_probs=true
```

**2. `verl/trainer/ppo/mappo_trainer.py:_single_agent_rollout`** — replace lines ~1456–1468:

```python
# Current (expensive FSDP forward pass):
with marked_timer("old_log_prob", timing_raw, color="blue"):
    old_log_prob = self.actor_rollout_wgs[agent_key].compute_log_prob(batch)
    entropys = old_log_prob.batch["entropys"]
    ...
    batch = batch.union(old_log_prob)

# Proposed (zero-cost bypass with fallback):
if "rollout_log_probs" in batch.batch:
    # Reuse logprobs already computed by vLLM during generation
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
else:
    # Fallback: recompute via FSDP (backwards-compatible)
    with marked_timer("old_log_prob", timing_raw, color="blue"):
        old_log_prob = self.actor_rollout_wgs[agent_key].compute_log_prob(batch)
        entropys = old_log_prob.batch["entropys"]
        ...
        batch = batch.union(old_log_prob)
```

### PPO correctness

With `ppo_epochs=1` (current config), `old_log_probs` and `rollout_log_probs` represent the same policy checkpoint. The PPO ratio `π_θ / π_old` is numerically equivalent whether `π_old` comes from vLLM BF16 or FSDP BF16 (difference ≈ 1e-4, well within PPO clip ε=0.2).

For future multi-epoch PPO: the bypass mode produces standard PPO semantics (ratio anchored to rollout policy). The current "recompute" mode produces a 3-policy variant. Both are valid; the bypass mode is the standard PPO formulation.

### Scalability to 4B models

The `old_log_prob` phase cost scales with model size (8× more FLOPs for 4B vs 0.5B). At 4B, this phase would cost ~120–200s per round. Bypass mode eliminates this entirely regardless of model size — the vLLM log probs are a free byproduct of generation that the model was already computing.

### Tradeoffs

| | Bypass mode | Current |
|--|--|--|
| PPO loss entropy | Unchanged (from update_actor) | Unchanged |
| `actor/entropy` monitoring | From update_actor (post-update) | From old_log_prob (pre-update) |
| `old_log_probs` precision | vLLM BF16 (~1e-4 diff) | FSDP BF16 |
| Memory overhead | +~1MB/step | None |
| Fallback | Yes (if calculate_log_probs=False) | N/A |

---

## Combined expected speedup

| Change | Phase affected | Estimated savings |
|--------|---------------|-------------------|
| A: `ppo_micro_batch_size_per_gpu` 4→32 | update_actor | 4–6× faster |
| A: `ppo_mini_batch_size` 64→256 | update_actor | fewer loop iterations |
| A: `critic.ppo_micro_batch_size_per_gpu` 4→32 | update_critic | 4–6× faster |
| A: `gpu_memory_utilization` 0.5→0.7 | gen | ~20% faster |
| B: bypass old_log_prob | old_log_prob | ~90s/step eliminated |

**Projected total: 9h → ~1.5–2.5h** for the 0.5B run.  
**4B model benefit:** Approach B saves 600–900s/step (vs 90s for 0.5B), making it the dominant optimization at scale.

---

## Implementation order

1. **Approach A first** — slurm-only, no code change, immediately verifiable
2. **Approach B second** — 15 lines in `mappo_trainer.py` + 1 slurm flag, verify on a short run (10 steps)

---

## Files changed

- `train_q05b.slurm` — hyperparameter updates + `calculate_log_probs=true`
- `verl/trainer/ppo/mappo_trainer.py` — ~15 lines in `_single_agent_rollout`
