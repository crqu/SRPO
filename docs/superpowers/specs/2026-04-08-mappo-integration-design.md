# MAPPO Integration Design

**Date:** 2026-04-08  
**Scope:** Integrate `RayMAPPOTrainer`, `RayRiskAverseTrainer`, and the `main_mappo.py` entrypoint from the research file `ray_trainer_self_version.py` into the SRPO repo.

---

## Goals

- Move the MAPPO training stack into the verl package structure so it is importable, testable, and launchable like `main_ppo.py`
- Break the circular import between the trainer and the entrypoint
- Avoid touching `verl/trainer/ppo/ray_trainer.py` (PPO trainer stays unchanged)
- Keep configs composable — MAPPO config extends PPO config via Hydra defaults

## Out of Scope

- `RayMultiAgentRollout`, `RayZOTrainer`, `RayMAPPOShareTrainer` — not integrated in this effort
- Changes to `RayPPOTrainer` or `main_ppo.py`
- Writing tests

---

## Final File Layout

```
verl/trainer/
├── config/
│   └── mappo_trainer.yaml          # NEW: Hydra config extending ppo_trainer.yaml
├── main_ppo.py                     # UNCHANGED
├── main_mappo.py                   # MOVED from repo root; imports updated
└── ppo/
    ├── ray_trainer.py              # UNCHANGED
    └── mappo_trainer.py            # NEW: dataset utils + RayMAPPOTrainer + RayRiskAverseTrainer
```

Root-level `main_mappo.py` and `ray_trainer_self_version.py` are deleted after migration.

---

## Component Details

### 1. `verl/trainer/ppo/mappo_trainer.py` (new)

Extracted from `ray_trainer_self_version.py`. Contains:

**Module-level functions** (moved from `main_mappo.py`):
- `create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train)` — builds an `RLHFDataset` or custom dataset class for one agent
- `create_rl_sampler(data_config, dataset)` — builds a `RandomSampler` or `SequentialSampler` (or custom curriculum sampler)

Moving these here breaks the circular import: the trainer no longer needs to import from the entrypoint.

**Classes:**
- `RayMAPPOTrainer` — extracted from `ray_trainer_self_version.py` lines 2630–4966
  - `_create_dataloader` updated: remove `from verl.trainer.main_mappo import ...`; call the module-level functions directly
- `RayRiskAverseTrainer(RayMAPPOTrainer)` — extracted from lines 5742–6090, no changes needed

**Key imports used by these classes:**
```python
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (...)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
# ... etc (mirror the top of ray_trainer_self_version.py minus ResourcePoolManager definition)
```

`ResourcePoolManager` is **not** redefined — imported from `verl.single_controller.ray` (same object re-exported through `verl.trainer.ppo.ray_trainer`).

---

### 2. `verl/trainer/main_mappo.py` (moved from repo root)

Changes from the root version:
1. `config_path="config"` stays `"config"` — now resolves to `verl/trainer/config/` when the file lives in `verl/trainer/`, matching how `main_ppo.py` works
2. Replace local definitions of `create_rl_dataset` / `create_rl_sampler` with imports:
   ```python
   from verl.trainer.ppo.mappo_trainer import create_rl_dataset, create_rl_sampler
   ```
3. Update trainer import:
   ```python
   # before (root-level, pointed at future location):
   from verl.trainer.ppo.ray_trainer import RayMAPPOTrainer
   # after:
   from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer
   ```
4. `TaskRunner.init_resource_pool_mgr` already imports `ResourcePoolManager` from `verl.trainer.ppo.ray_trainer` — this continues to work unchanged.

---

### 3. `verl/trainer/config/mappo_trainer.yaml` (new)

Uses Hydra defaults to inherit the full PPO base config, then overrides only what differs:

```yaml
defaults:
  - ppo_trainer        # inherit all PPO config
  - _self_             # allow overrides below

# MAPPO-specific section
multi_agent:
  num_agents: 2
  num_rounds: 3
  risk_coef: 1.0
  discussion_prompt: "The discussion history is as follows:"
  agents:
    - actor:
        model:
          path: ???
      n_gpus_per_node: 8
      nnodes: 1
    - actor:
        model:
          path: ???
      n_gpus_per_node: 8
      nnodes: 1
```

All PPO fields (`trainer`, `actor_rollout_ref`, `critic`, `reward_model`, etc.) are inherited from `ppo_trainer.yaml` and can be overridden per-run on the command line.

---

## Import Flow (no circularity)

```
main_mappo.py
  └─imports─► verl.trainer.ppo.mappo_trainer
                 ├─ create_rl_dataset, create_rl_sampler  (module-level)
                 ├─ RayMAPPOTrainer
                 └─ RayRiskAverseTrainer
                      └─imports─► verl.trainer.ppo.ray_trainer  (Role, WorkerType, ...)
                      └─imports─► verl.single_controller.ray    (ResourcePoolManager)
                      └─imports─► verl.trainer.ppo.*            (core_algos, metric_utils, reward, ...)
```

---

## Migration Steps (high level)

1. Create `verl/trainer/ppo/mappo_trainer.py` — extract classes and utility functions from `ray_trainer_self_version.py`
2. Create `verl/trainer/main_mappo.py` — move from repo root, update imports
3. Create `verl/trainer/config/mappo_trainer.yaml` — compose from `ppo_trainer.yaml`
4. Delete `main_mappo.py` and `ray_trainer_self_version.py` from repo root
5. Verify import chain is clean (no circular deps)

---

## What Is Not Changed

- `verl/trainer/ppo/ray_trainer.py` — untouched
- `verl/trainer/main_ppo.py` — untouched
- All existing PPO config files — untouched
- All existing tests — untouched
