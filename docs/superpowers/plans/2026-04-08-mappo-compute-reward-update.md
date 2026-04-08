# MAPPO compute_reward Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deleted `compute_reward`/`compute_reward_async` calls in `mappo_trainer.py` with a private `_compute_reward` helper method on `RayMAPPOTrainer`, dropping the async path entirely.

**Architecture:** Add one `_compute_reward(self, batch, reward_fn)` method to `RayMAPPOTrainer`. Because `RayRiskAverseTrainer` inherits from `RayMAPPOTrainer`, all three broken call sites are fixed with one method. The async branches (`launch_reward_fn_async`) are deleted — reward is always computed synchronously.

**Tech Stack:** Python, PyTorch, Ray, verl `DataProto`

---

## File Map

| File | Change |
|---|---|
| `verl/trainer/ppo/mappo_trainer.py` | Remove broken import (lines 52–56); add `_compute_reward` method; fix 3 call sites; remove async branches |
| `tests/trainer/ppo/test_mappo_compute_reward.py` | New — unit tests for `_compute_reward` |

---

### Task 1: Write a failing unit test for `_compute_reward`

**Files:**
- Create: `tests/trainer/ppo/test_mappo_compute_reward.py`

- [ ] **Step 1: Create the test directory if needed**

```bash
mkdir -p tests/trainer/ppo
touch tests/trainer/ppo/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/trainer/ppo/test_mappo_compute_reward.py`:

```python
"""Unit tests for RayMAPPOTrainer._compute_reward."""
import torch
import pytest
from unittest.mock import MagicMock
from verl import DataProto


def _make_trainer():
    """Construct a minimal RayMAPPOTrainer without Ray/GPU."""
    from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer
    trainer = object.__new__(RayMAPPOTrainer)
    return trainer


def _make_batch():
    batch = DataProto(
        batch={"input_ids": torch.zeros(2, 4, dtype=torch.long)},
        non_tensor_batch={},
        meta_info={},
    )
    return batch


def test_compute_reward_return_dict():
    """reward_fn returns dict with reward_tensor and reward_extra_info."""
    trainer = _make_trainer()
    batch = _make_batch()
    expected_tensor = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    expected_extra = {"score": [0.9, 0.4]}

    reward_fn = MagicMock(return_value={
        "reward_tensor": expected_tensor,
        "reward_extra_info": expected_extra,
    })

    reward_tensor, reward_extra_infos_dict = trainer._compute_reward(batch, reward_fn)

    reward_fn.assert_called_once_with(batch, return_dict=True)
    assert torch.equal(reward_tensor, expected_tensor)
    assert reward_extra_infos_dict == expected_extra


def test_compute_reward_missing_extra_info():
    """reward_fn returns dict without reward_extra_info — defaults to empty dict."""
    trainer = _make_trainer()
    batch = _make_batch()
    expected_tensor = torch.tensor([[1.0, 0.0]])

    reward_fn = MagicMock(return_value={"reward_tensor": expected_tensor})

    reward_tensor, reward_extra_infos_dict = trainer._compute_reward(batch, reward_fn)

    assert torch.equal(reward_tensor, expected_tensor)
    assert reward_extra_infos_dict == {}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
conda activate srpo
pytest tests/trainer/ppo/test_mappo_compute_reward.py -v
```

Expected: `AttributeError: type object 'RayMAPPOTrainer' has no attribute '_compute_reward'`

---

### Task 2: Remove the broken import

**Files:**
- Modify: `verl/trainer/ppo/mappo_trainer.py:52-56`

- [ ] **Step 1: Delete lines 52–56**

In `verl/trainer/ppo/mappo_trainer.py`, remove this entire block:

```python
try:
    from verl.trainer.ppo.reward import compute_reward, compute_reward_async
except ImportError:
    compute_reward = None  # type: ignore[assignment]
    compute_reward_async = None  # type: ignore[assignment]
```

After the edit, line 52 should be the `from verl.trainer.ppo.utils import ...` line.

- [ ] **Step 2: Verify the file parses cleanly**

```bash
python -c "import verl.trainer.ppo.mappo_trainer"
```

Expected: no output (no ImportError).

---

### Task 3: Add `_compute_reward` method to `RayMAPPOTrainer`

**Files:**
- Modify: `verl/trainer/ppo/mappo_trainer.py`

- [ ] **Step 1: Find the right insertion point**

Look for the end of `_get_gen_batch` or any small utility method early in `RayMAPPOTrainer`. Add `_compute_reward` as a new method just before `_compute_round`. Search for the line:

```python
    def _compute_round(
```

Insert the following method immediately before it:

```python
    def _compute_reward(self, batch: DataProto, reward_fn) -> tuple[torch.Tensor, dict]:
        """Call reward_fn synchronously and return (reward_tensor, reward_extra_infos_dict)."""
        reward_result = reward_fn(batch, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
        return reward_tensor, reward_extra_infos_dict

```

- [ ] **Step 2: Run the unit tests — they should now pass**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
conda activate srpo
pytest tests/trainer/ppo/test_mappo_compute_reward.py -v
```

Expected: both tests PASS.

---

### Task 4: Fix call site 1 — `_compute_round` reward block (~lines 1451–1465 and 1496–1500)

**Files:**
- Modify: `verl/trainer/ppo/mappo_trainer.py`

This call site has two parts: the async dispatch (in the `reward` timer) and the async resolution (in the `adv` timer).

- [ ] **Step 1: Replace the async dispatch block in the `reward` timer**

Find this block (inside `with marked_timer("reward", ...)`):

```python
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(data=batch, reward_fn=reward_fn)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
```

Replace with:

```python
                reward_tensor, reward_extra_infos_dict = self._compute_reward(batch, reward_fn)
```

- [ ] **Step 2: Remove the async resolution block in the `adv` timer**

Find this block (inside `with marked_timer("adv", ...)`):

```python
                reward_extra_infos_dict = reward_extra_infos_dict or {}
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                batch.batch["token_level_scores"] = reward_tensor
```

Replace with:

```python
                reward_extra_infos_dict = reward_extra_infos_dict or {}
                batch.batch["token_level_scores"] = reward_tensor
```

- [ ] **Step 3: Verify the file parses**

```bash
python -c "import verl.trainer.ppo.mappo_trainer"
```

Expected: no output.

---

### Task 5: Fix call site 2 — `_run_single_agent` reward block (~lines 1607–1620 and 1651–1655)

**Files:**
- Modify: `verl/trainer/ppo/mappo_trainer.py`

Same two-part pattern as Task 4, but inside `_run_single_agent`.

- [ ] **Step 1: Replace the async dispatch block in the `reward` timer**

Find this block (inside `_run_single_agent`, `with marked_timer("reward", ...)`):

```python
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(data=batch, reward_fn=reward_fn)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
```

Replace with:

```python
                reward_tensor, reward_extra_infos_dict = self._compute_reward(batch, reward_fn)
```

- [ ] **Step 2: Remove the async resolution block in the `adv` timer**

Find this block (inside `_run_single_agent`, `with marked_timer("adv", ...)`):

```python
                reward_extra_infos_dict = reward_extra_infos_dict or {}
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                batch.batch["token_level_scores"] = reward_tensor
```

Replace with:

```python
                reward_extra_infos_dict = reward_extra_infos_dict or {}
                batch.batch["token_level_scores"] = reward_tensor
```

- [ ] **Step 3: Verify the file parses**

```bash
python -c "import verl.trainer.ppo.mappo_trainer"
```

Expected: no output.

---

### Task 6: Fix call site 3 — `RayRiskAverseTrainer` reward block (~line 3069–3078)

**Files:**
- Modify: `verl/trainer/ppo/mappo_trainer.py`

`RayRiskAverseTrainer` inherits `_compute_reward` from `RayMAPPOTrainer` — no new method needed.

- [ ] **Step 1: Replace the reward block**

Find this block (inside `RayRiskAverseTrainer`, `with marked_timer("reward", ...)`):

```python
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
```

Replace with:

```python
                reward_tensor, reward_extra_infos_dict = self._compute_reward(batch, self.reward_fn)
```

Note: there is no deferred `ray.get` in `RayRiskAverseTrainer` — `reward_tensor` is used immediately on the next line (`scores = reward_tensor.sum(-1).cpu().tolist()`), so no `adv` timer cleanup is needed here.

- [ ] **Step 2: Verify no remaining references to `compute_reward` or `compute_reward_async`**

```bash
grep -n "compute_reward\b\|compute_reward_async\|future_reward\|launch_reward_fn_async" \
    verl/trainer/ppo/mappo_trainer.py | grep -v "^[0-9]*:#"
```

Expected: only commented-out lines (prefixed with `#`) should remain, all inside the large commented block around lines 2200–2310.

- [ ] **Step 3: Verify the file parses**

```bash
python -c "import verl.trainer.ppo.mappo_trainer"
```

Expected: no output.

---

### Task 7: Run all unit tests and commit

**Files:** none new

- [ ] **Step 1: Run the unit tests**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
conda activate srpo
pytest tests/trainer/ppo/test_mappo_compute_reward.py -v
```

Expected:
```
PASSED tests/trainer/ppo/test_mappo_compute_reward.py::test_compute_reward_return_dict
PASSED tests/trainer/ppo/test_mappo_compute_reward.py::test_compute_reward_missing_extra_info
2 passed
```

- [ ] **Step 2: Commit**

```bash
git add verl/trainer/ppo/mappo_trainer.py \
        tests/trainer/ppo/__init__.py \
        tests/trainer/ppo/test_mappo_compute_reward.py
git commit -m "$(cat <<'EOF'
fix(mappo): replace deprecated compute_reward with _compute_reward helper

Removes the broken try/except import of compute_reward/compute_reward_async
(deleted upstream in 3df21a68). Adds RayMAPPOTrainer._compute_reward as a
synchronous replacement. Drops the launch_reward_fn_async async path entirely
at all three call sites.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- ✅ Remove broken import → Task 2
- ✅ Add `_compute_reward` to `RayMAPPOTrainer` → Task 3
- ✅ Call site 1 (`_compute_round`) → Task 4
- ✅ Call site 2 (`_run_single_agent`) → Task 5
- ✅ Call site 3 (`RayRiskAverseTrainer`) → Task 6
- ✅ Drop async path (`launch_reward_fn_async`) → Tasks 4, 5, 6
- ✅ `RayRiskAverseTrainer` inherits — no duplicate method → covered in Task 6 note

**Placeholder scan:** No TBDs, no "implement later", all code blocks complete.

**Type consistency:** `_compute_reward` signature is `(self, batch: DataProto, reward_fn) -> tuple[torch.Tensor, dict]` — used consistently across Tasks 3, 4, 5, 6.
