# Resource Allocation Fix Design

**Date:** 2026-04-19  
**Status:** Approved

## Problem

Two bugs in MAPPO multi-agent resource allocation:

### Bug 1 — Fractional GPU allocation (`max_colocate_count` hardcoded to 3)

`ResourcePoolManager.create_resource_pool()` always creates `RayResourcePool` with `max_colocate_count=3`. This propagates to `_create_worker` where:

```python
num_gpus = 1 / resource_pool.max_colocate_count  # always 0.333...
```

The Ray placement group bundle is sized for 3 co-located workers (`{"CPU": 3, "GPU": 1}`). In MAPPO without reference policy, each `agent_pool_i` has only 2 co-located workers (actor+critic). Each worker claims `1/3` GPU instead of `1/2`, leaving `1/3` GPU per bundle wasted in Ray's accounting. Similarly, `reward_pool` always had 1 worker but was sized for 3.

### Bug 2 — `nodes_per_agent` double-divides nnodes

In `init_resource_pool_mgr` (`main_mappo.py`):

```python
nnodes = int(a.get("nnodes", config.trainer.nnodes))  # per-agent field or global fallback
nodes_per_agent = nnodes // num_agents                 # divides again by num_agents
```

If a user set `agents[i].nnodes=2` meaning "2 nodes for this agent", the code would compute `2 // num_agents`, giving the wrong result. `nnodes` is always global; the per-agent override was semantically broken.

## Solution

### Fix A — `nodes_per_agent` always uses `config.trainer.nnodes`

Remove the per-agent `nnodes` read. Always compute:

```python
nodes_per_agent = config.trainer.nnodes // num_agents
```

Per-agent `n_gpus_per_node` override is preserved. This makes the semantics explicit: the cluster's total nodes are divided equally among agents.

**Validation:** raise `ValueError` if `nodes_per_agent <= 0` (existing check, unchanged).

### Fix B — Dynamic `max_colocate_count` via `colocate_count_dict`

Add an optional `colocate_count_dict: dict[str, int]` field (default `{}`) to `ResourcePoolManager`. `create_resource_pool` looks up each pool by name, falling back to `3` for callers that don't provide it (backward-compatible with single-agent PPO).

```python
max_colocate_count = self.colocate_count_dict.get(resource_pool_name, 3)
```

In `init_resource_pool_mgr`, compute before constructing `ResourcePoolManager`:

```python
colocate_per_agent = 1 + int(need_critic(config)) + int(need_reference_policy(config))
colocate_count_dict = {f"agent_pool_{i}": colocate_per_agent for i in range(num_agents)}
if config.reward_model.enable_resource_pool:
    colocate_count_dict["reward_pool"] = 1  # only RewardModelWorker
```

## Files Changed

| File | Change |
|------|--------|
| `verl/single_controller/ray/base.py` | Add `colocate_count_dict` field to `ResourcePoolManager`; use it in `create_resource_pool` |
| `verl/trainer/main_mappo.py` | Fix `nodes_per_agent` to use `config.trainer.nnodes`; compute and pass `colocate_count_dict` |

## Invariants

- `nodes_per_agent >= 1` — enforced by existing ValueError
- `colocate_per_agent >= 1` — always includes actor_rollout
- `reward_pool` colocate count = 1 when `enable_resource_pool` is true
- No change to single-agent `ray_trainer.py` path (falls back to `max_colocate_count=3`)
