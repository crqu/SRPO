# MAPPO Training Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce MAPPO training time from ~9h to ~1.5–2.5h by fixing micro-batch size (Approach A) and bypassing redundant FSDP log-prob recomputation (Approach B).

**Architecture:** Approach A is pure hyperparameter tuning in the slurm script. Approach B extracts a `_set_old_log_probs` helper in `RayMAPPOTrainer` that assigns `old_log_probs` directly from the vLLM-generated `rollout_log_probs` when available, falling back to the existing FSDP forward pass otherwise.

**Tech Stack:** Python, PyTorch, verl (Ray + FSDP + vLLM), Hydra config, pytest, SLURM

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `train_q05b.slurm` | Modify | 5 hyperparameter values + 1 new flag |
| `verl/trainer/ppo/mappo_trainer.py` | Modify | Add `_set_old_log_probs` helper; replace inline block in `_single_agent_rollout` |
| `tests/trainer/ppo/test_mappo_bypass_logprobs.py` | Create | Unit tests for the new helper |

---

## Task 1: Approach A — Fix micro-batch hyperparameters in slurm

**Files:**
- Modify: `train_q05b.slurm`

- [ ] **Step 1: Update the 5 hyperparameters**

In `train_q05b.slurm`, inside the `srun ... python -m verl.trainer.main_mappo` block, make these replacements:

```bash
# FROM:
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
critic.ppo_micro_batch_size_per_gpu=4 \

# TO:
actor_rollout_ref.actor.ppo_mini_batch_size=256 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
critic.ppo_micro_batch_size_per_gpu=32 \
```

Rationale:
- `ppo_micro_batch_size_per_gpu` 4→32: fixes 1% actor MFU — average sequence length is ~418 tokens so 32 sequences ≈ 13K tokens/pass (well within memory)
- `ppo_mini_batch_size` 64→256: one mini-batch per epoch (= train_batch_size), removes outer loop
- `gpu_memory_utilization` 0.5→0.7: gives vLLM 56GB KV cache instead of 40GB

- [ ] **Step 2: Verify the diff is correct**

```bash
git diff train_q05b.slurm
```

Expected: exactly 5 changed lines, no other changes.

- [ ] **Step 3: Commit**

```bash
git add train_q05b.slurm
git commit -m "perf: increase ppo micro-batch size and vllm memory utilization"
```

---

## Task 2: Write failing tests for the bypass helper

**Files:**
- Create: `tests/trainer/ppo/test_mappo_bypass_logprobs.py`

- [ ] **Step 1: Create the test file**

```python
"""Unit tests for RayMAPPOTrainer._set_old_log_probs bypass logic."""
import torch
import pytest
from unittest.mock import MagicMock
from omegaconf import OmegaConf
from verl import DataProto


def _make_trainer():
    """Instantiate RayMAPPOTrainer without Ray/GPU init."""
    from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer
    trainer = object.__new__(RayMAPPOTrainer)
    trainer.config = OmegaConf.create({
        "actor_rollout_ref": {"actor": {"loss_agg_mode": "token-mean"}}
    })
    return trainer


def _make_batch(with_rollout_log_probs: bool):
    bsz, seq_len = 2, 8
    tensors = {
        "input_ids": torch.zeros(bsz, seq_len, dtype=torch.long),
        "response_mask": torch.ones(bsz, seq_len),
    }
    if with_rollout_log_probs:
        tensors["rollout_log_probs"] = torch.randn(bsz, seq_len)
    return DataProto.from_dict(tensors=tensors)


def test_bypass_sets_old_log_probs_from_rollout():
    """When rollout_log_probs present, old_log_probs is set to it without calling compute_log_prob."""
    trainer = _make_trainer()
    batch = _make_batch(with_rollout_log_probs=True)
    expected = batch.batch["rollout_log_probs"].clone()
    mock_wg = MagicMock()

    result = trainer._set_old_log_probs(batch, mock_wg, {}, {})

    mock_wg.compute_log_prob.assert_not_called()
    assert "old_log_probs" in result.batch
    assert torch.equal(result.batch["old_log_probs"], expected)


def test_bypass_fallback_calls_compute_log_prob():
    """When rollout_log_probs absent, compute_log_prob is called and old_log_probs is populated."""
    trainer = _make_trainer()
    batch = _make_batch(with_rollout_log_probs=False)
    bsz, seq_len = 2, 8

    fake_lp = DataProto.from_dict(tensors={
        "old_log_probs": torch.zeros(bsz, seq_len),
        "entropys": torch.ones(bsz, seq_len),
    })
    mock_wg = MagicMock()
    mock_wg.compute_log_prob.return_value = fake_lp
    metrics = {}

    result = trainer._set_old_log_probs(batch, mock_wg, {}, metrics)

    mock_wg.compute_log_prob.assert_called_once_with(batch)
    assert "old_log_probs" in result.batch
    assert torch.equal(result.batch["old_log_probs"], fake_lp.batch["old_log_probs"])
    assert "actor/entropy" in metrics


def test_bypass_fallback_does_not_leave_entropys_in_batch():
    """Fallback path pops entropys so they don't leak into downstream PPO steps."""
    trainer = _make_trainer()
    batch = _make_batch(with_rollout_log_probs=False)
    bsz, seq_len = 2, 8

    fake_lp = DataProto.from_dict(tensors={
        "old_log_probs": torch.zeros(bsz, seq_len),
        "entropys": torch.ones(bsz, seq_len),
    })
    mock_wg = MagicMock()
    mock_wg.compute_log_prob.return_value = fake_lp

    result = trainer._set_old_log_probs(batch, mock_wg, {}, {})

    assert "entropys" not in result.batch
```

- [ ] **Step 2: Run tests — verify they FAIL with AttributeError**

```bash
module load anaconda3/2024.02-1 && conda activate srpo
cd /weka/scratch/lshi40_llm/mallm/SRPO
python -m pytest tests/trainer/ppo/test_mappo_bypass_logprobs.py -v
```

Expected output:
```
FAILED ... AttributeError: '_set_old_log_probs'
```

All three tests should fail because `_set_old_log_probs` does not exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/trainer/ppo/test_mappo_bypass_logprobs.py
git commit -m "test: add failing tests for _set_old_log_probs bypass logic"
```

---

## Task 3: Implement `_set_old_log_probs` and wire it into `_single_agent_rollout`

**Files:**
- Modify: `verl/trainer/ppo/mappo_trainer.py`

- [ ] **Step 1: Add the `_set_old_log_probs` helper method**

Find the `_compute_reward` method in `mappo_trainer.py` (around line 1507). Add the new method immediately before it:

```python
def _set_old_log_probs(
    self,
    batch: DataProto,
    actor_rollout_wg,
    timing_raw: dict,
    metrics: dict,
) -> DataProto:
    """Populate batch with old_log_probs for PPO ratio computation.

    Bypass path (fast): if generate_sequences already returned rollout_log_probs
    (requires actor_rollout_ref.rollout.calculate_log_probs=true), reuse them
    directly — avoids a full FSDP forward pass (~30s per round).

    Fallback path: recompute via FSDP compute_log_prob (original behaviour).
    """
    if "rollout_log_probs" in batch.batch:
        batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
        return batch
    with marked_timer("old_log_prob", timing_raw, color="blue"):
        old_log_prob = actor_rollout_wg.compute_log_prob(batch)
        entropys = old_log_prob.batch["entropys"]
        response_masks = batch.batch["response_mask"]
        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
        entropy_agg = agg_loss(
            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
        )
        metrics["actor/entropy"] = entropy_agg.detach().item()
        old_log_prob.batch.pop("entropys")
        batch = batch.union(old_log_prob)
    return batch
```

- [ ] **Step 2: Replace the inline old_log_prob block in `_single_agent_rollout`**

Locate lines 1456–1468 in `_single_agent_rollout` (the block starting with `# recompute old_log_probs`). Replace the entire block:

```python
# REMOVE this block:
            # recompute old_log_probs
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_rollout_wgs[agent_key].compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(
                    loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                )
                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

# REPLACE WITH:
            batch = self._set_old_log_probs(
                batch, self.actor_rollout_wgs[agent_key], timing_raw, metrics
            )
```

- [ ] **Step 3: Run the tests — verify they PASS**

```bash
python -m pytest tests/trainer/ppo/test_mappo_bypass_logprobs.py -v
```

Expected output:
```
PASSED tests/trainer/ppo/test_mappo_bypass_logprobs.py::test_bypass_sets_old_log_probs_from_rollout
PASSED tests/trainer/ppo/test_mappo_bypass_logprobs.py::test_bypass_fallback_calls_compute_log_prob
PASSED tests/trainer/ppo/test_mappo_bypass_logprobs.py::test_bypass_fallback_does_not_leave_entropys_in_batch
3 passed
```

- [ ] **Step 4: Run the existing mappo tests to check for regressions**

```bash
python -m pytest tests/trainer/ppo/test_mappo_compute_reward.py -v
```

Expected: all tests pass (no regressions).

- [ ] **Step 5: Commit**

```bash
git add verl/trainer/ppo/mappo_trainer.py
git commit -m "perf: bypass FSDP old_log_prob recomputation when rollout_log_probs available"
```

---

## Task 4: Enable `calculate_log_probs` in the slurm script

**Files:**
- Modify: `train_q05b.slurm`

- [ ] **Step 1: Add the flag**

In `train_q05b.slurm`, inside the `srun ... python -m verl.trainer.main_mappo` block, add the following line after the existing rollout flags:

```bash
        actor_rollout_ref.rollout.calculate_log_probs=true \
```

A good place is right after `actor_rollout_ref.rollout.load_format=safetensors \`.

- [ ] **Step 2: Verify the diff**

```bash
git diff train_q05b.slurm
```

Expected: exactly 1 new line added.

- [ ] **Step 3: Commit**

```bash
git add train_q05b.slurm
git commit -m "perf: enable rollout calculate_log_probs to activate bypass mode"
```

---

## Task 5: Integration smoke test

**Files:** none (verification only)

- [ ] **Step 1: Run a short 10-step training job**

Edit a temporary copy of the slurm script (or override inline) to run only 10 steps:

```bash
# In a copy or by overriding at sbatch time:
sbatch --export=ALL,METHOD=ippo train_q05b.slurm
# or override steps:
# trainer.total_training_steps=10 trainer.test_freq=10 trainer.save_freq=10
```

The simplest approach — submit with a step override by temporarily editing the launch command:
```bash
trainer.total_training_steps=10 \
trainer.test_freq=10 \
trainer.save_freq=10 \
```

- [ ] **Step 2: Verify Approach B is active — old_log_prob timing absent**

Once the job produces output in `logs/`, check:

```bash
grep "timing_s/old_log_prob" logs/mappo_q05b_*.output
```

Expected: **no output** (the `old_log_prob` timer is only entered in the fallback path, which should not be triggered when `calculate_log_probs=true`).

- [ ] **Step 3: Verify Approach B is active — rollout_log_probs present**

```bash
grep "rollout_log_probs\|old_log_probs" logs/mappo_q05b_*.output | head -5
```

Expected: the batch should contain `old_log_probs` (assigned from rollout) and training should complete without error.

- [ ] **Step 4: Verify no OOM from Approach A**

```bash
grep -i "out of memory\|CUDA error\|OOM" logs/mappo_q05b_*.output
```

Expected: no output.

- [ ] **Step 5: Verify actor MFU improved**

```bash
grep "perf/mfu/actor:" logs/mappo_q05b_*.output | head -5
```

Expected: values noticeably above the previous `0.010611` (target: >0.05). Exact value depends on sequence packing, but should be at least 4× higher.

- [ ] **Step 6: Verify training runs to completion**

```bash
grep "Training Progress\|step:10" logs/mappo_q05b_*.output | tail -5
```

Expected: step 10 logged successfully.

- [ ] **Step 7: Commit verification note**

```bash
git commit --allow-empty -m "chore: verified 10-step smoke test passes with A+B speedups"
```

---

## Self-Review Checklist

- **Spec coverage:**
  - [x] Approach A: 5 hyperparameter changes — Task 1
  - [x] Approach B: `calculate_log_probs=true` — Task 4
  - [x] Approach B: bypass logic in `_single_agent_rollout` — Task 3
  - [x] Fallback path preserved — `_set_old_log_probs` fallback branch, tested in Task 2
  - [x] Entropy loss unaffected — no change to `dp_actor.py`; documented in spec
  - [x] Integration verification — Task 5

- **No placeholders:** all steps have exact commands or code blocks.

- **Type consistency:** `_set_old_log_probs(self, batch, actor_rollout_wg, timing_raw, metrics)` signature used consistently in Task 2 (tests) and Task 3 (implementation).
