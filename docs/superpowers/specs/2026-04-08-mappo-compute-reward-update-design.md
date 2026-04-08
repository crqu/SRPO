# Design: Replace deprecated `compute_reward` in `mappo_trainer.py`

**Date:** 2026-04-08  
**Branch:** `new/mappo`  
**File affected:** `verl/trainer/ppo/mappo_trainer.py`

---

## Background

`compute_reward` and `compute_reward_async` were removed from `verl/trainer/ppo/reward.py` in commit `3df21a68` ("deprecate batch reward manager"). `mappo_trainer.py` still imports them via a try/except fallback that silently sets them to `None`, then calls them at 3 locations — meaning reward computation is currently broken at runtime.

The upstream `ray_trainer.py` has migrated to a `RewardLoopManager` + `extract_reward` pattern, but MAPPO uses per-agent Python callables (`reward_fns`) rather than a `RewardLoopManager`, so a full migration is deferred.

---

## Decision

**Option B — Private helper method.** Add a `_compute_reward(self, batch, reward_fn)` method on each trainer class. Drop the async path (`launch_reward_fn_async`) entirely — always compute synchronously, mirroring the direction of `ray_trainer.py`.

---

## Changes

### 1. Import block (`mappo_trainer.py` lines 52–56)

Remove:
```python
try:
    from verl.trainer.ppo.reward import compute_reward, compute_reward_async
except ImportError:
    compute_reward = None
    compute_reward_async = None
```

No replacement import needed — the broken import block is simply removed.

### 2. New `_compute_reward` method on `RayMAPPOTrainer`

```python
def _compute_reward(self, batch: DataProto, reward_fn) -> tuple[torch.Tensor, dict]:
    reward_result = reward_fn(batch, return_dict=True)
    reward_tensor = reward_result["reward_tensor"]
    reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    return reward_tensor, reward_extra_infos_dict
```

### 3. Call site 1 — `_compute_round` (~line 1451)

Replace:
```python
if self.config.reward_model.launch_reward_fn_async:
    future_reward = compute_reward_async.remote(data=batch, reward_fn=reward_fn)
else:
    reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
```
With:
```python
reward_tensor, reward_extra_infos_dict = self._compute_reward(batch, reward_fn)
```

Also remove the deferred `ray.get(future_reward)` at ~line 1499–1500 and replace with direct use of `reward_tensor`.

### 4. Call site 2 — `_run_single_agent` (~line 1617)

Same replacement as call site 1. Remove async branch and deferred `ray.get`.

### 5. `RayRiskAverseTrainer` (~line 3075)

Add an identical `_compute_reward` method on `RayRiskAverseTrainer` (it does not inherit from `RayMAPPOTrainer`). Replace:
```python
if self.config.reward_model.launch_reward_fn_async:
    future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
else:
    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
```
With:
```python
reward_tensor, reward_extra_infos_dict = self._compute_reward(batch, self.reward_fn)
```

---

## What does NOT change

- `self.rm_wg.compute_rm_score(batch)` calls (RM worker path)
- Advantage computation, value computation, all downstream training logic
- `config.reward_model.launch_reward_fn_async` config key (can be cleaned up separately)

---

## Out of scope

Full migration to `RewardLoopManager` + `extract_reward` (Option C) is the right long-term direction but is a separate, larger refactor.
