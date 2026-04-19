# Resource Allocation Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two bugs in MAPPO multi-agent resource allocation: (1) hardcoded `max_colocate_count=3` causing fractional GPU misallocation, and (2) `nodes_per_agent` incorrectly reading a per-agent `nnodes` override instead of always using `config.trainer.nnodes`.

**Architecture:** Add an optional `colocate_count_dict: dict[str, int]` field to `ResourcePoolManager` (backward-compatible, falls back to `3`). In `init_resource_pool_mgr`, compute actual co-location counts from config flags (`need_critic`, `need_reference_policy`) and pass them in. Fix `nodes_per_agent` to always divide `config.trainer.nnodes` evenly.

**Tech Stack:** Python dataclasses, Ray, OmegaConf, pytest, `inspect.getsource` for structural tests (matches existing test style in this repo).

---

## File Map

| File | Change |
|------|--------|
| `verl/single_controller/ray/base.py` | Add `colocate_count_dict` field to `ResourcePoolManager`; use it in `create_resource_pool` |
| `verl/trainer/main_mappo.py` | Fix `nodes_per_agent`; compute and pass `colocate_count_dict` to `ResourcePoolManager` |
| `tests/trainer/ppo/test_mappo_trainer_bugfixes.py` | Add structural tests for both fixes |

---

### Task 1: Write failing tests

**Files:**
- Modify: `tests/trainer/ppo/test_mappo_trainer_bugfixes.py`

- [ ] **Step 1: Add tests to the existing test file**

Append to `tests/trainer/ppo/test_mappo_trainer_bugfixes.py`:

```python
# ---------------------------------------------------------------------------
# Resource allocation fixes: colocate_count_dict + global nnodes
# ---------------------------------------------------------------------------
import dataclasses
from verl.single_controller.ray import ResourcePoolManager
from verl.trainer.main_mappo import TaskRunner


def test_resource_pool_manager_has_colocate_count_dict_field():
    """ResourcePoolManager must declare colocate_count_dict as a dataclass field."""
    fields = {f.name for f in dataclasses.fields(ResourcePoolManager)}
    assert "colocate_count_dict" in fields, (
        "ResourcePoolManager must have a colocate_count_dict field "
        "so callers can set per-pool max_colocate_count"
    )


def test_create_resource_pool_reads_colocate_count_dict():
    """create_resource_pool must read from colocate_count_dict, not hardcode 3."""
    src = inspect.getsource(ResourcePoolManager.create_resource_pool)
    assert "colocate_count_dict" in src, (
        "create_resource_pool must look up max_colocate_count from self.colocate_count_dict"
    )
    assert "max_colocate_count=3" not in src, (
        "create_resource_pool must not hardcode max_colocate_count=3"
    )


def test_init_resource_pool_mgr_uses_global_nnodes():
    """init_resource_pool_mgr must use config.trainer.nnodes, not per-agent nnodes."""
    src = inspect.getsource(TaskRunner.init_resource_pool_mgr)
    assert "config.trainer.nnodes" in src, (
        "nodes_per_agent must be derived from config.trainer.nnodes (global)"
    )
    assert not re.search(r'a\.get\s*\(\s*["\']nnodes["\']', src), (
        "init_resource_pool_mgr must not read per-agent 'nnodes' — nnodes is always global"
    )


def test_init_resource_pool_mgr_passes_colocate_count_dict():
    """init_resource_pool_mgr must build and pass colocate_count_dict to ResourcePoolManager."""
    src = inspect.getsource(TaskRunner.init_resource_pool_mgr)
    assert "colocate_count_dict" in src, (
        "init_resource_pool_mgr must compute colocate_count_dict and pass it to ResourcePoolManager"
    )
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
module load anaconda3/2024.02-1 && conda activate srpo
pytest tests/trainer/ppo/test_mappo_trainer_bugfixes.py \
    -k "colocate_count_dict or global_nnodes" -v 2>&1 | tail -20
```

Expected: 4 FAILED (fields/source don't exist yet).

---

### Task 2: Add `colocate_count_dict` to `ResourcePoolManager`

**Files:**
- Modify: `verl/single_controller/ray/base.py:181-209`

- [ ] **Step 1: Add the field and update `create_resource_pool`**

Current block (lines 181–209):
```python
@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[int, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()
```

Replace with:
```python
@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[int, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)
    # Per-pool co-location counts: how many WorkerGroups share one GPU slot.
    # Falls back to 3 for pools not listed (backward-compatible with single-agent PPO).
    colocate_count_dict: dict[str, int] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        max_colocate_count is read from colocate_count_dict for each pool
        (fallback: 3, which matches the single-agent actor+critic+ref layout).
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count = number of WorkerGroups co-located per GPU slot.
            # Callers should pass colocate_count_dict to match actual worker count.
            max_colocate_count = self.colocate_count_dict.get(resource_pool_name, 3)
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=max_colocate_count,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()
```

- [ ] **Step 2: Run the two `ResourcePoolManager` tests**

```bash
pytest tests/trainer/ppo/test_mappo_trainer_bugfixes.py \
    -k "colocate_count_dict" -v 2>&1 | tail -15
```

Expected: `test_resource_pool_manager_has_colocate_count_dict_field` PASSED, `test_create_resource_pool_reads_colocate_count_dict` PASSED. The two `main_mappo` tests still FAIL.

---

### Task 3: Fix `init_resource_pool_mgr` in `main_mappo.py`

**Files:**
- Modify: `verl/trainer/main_mappo.py:165-222`

- [ ] **Step 1: Replace the resource pool init method**

Current block (lines 165–222) — the `init_resource_pool_mgr` method of `TaskRunner`:

```python
    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        from verl.trainer.ppo.ray_trainer import Role

        # read multi-agent config
        ma = OmegaConf.select(config, "multi_agent") or {}
        num_agents = int(ma.get("num_agents", 1))
        assert num_agents >= 2, "MAPPO expected multi_agent.num_agents >= 2"
        agents_cfg = ma.get("agents", [])
        if not agents_cfg or len(agents_cfg) != num_agents:
            raise ValueError(
                "Please provide multi_agent.agents list (length == num_agents) with per-agent resource/model entries."
            )

        # global_pool_id = "global_pool"
        resource_pool_spec ={}
        # resource_pool_spec = {
        #     global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        # }
        # TODO Here you can use the new registration method to support dynamic registration of roles
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        
        # one pool per agent
        for i in range(num_agents):
            a = agents_cfg[i]
            total_n_gpus = int(a.get("n_gpus_per_node", config.trainer.n_gpus_per_node))
            nnodes = int(a.get("nnodes", config.trainer.nnodes))
            nodes_per_agent = nnodes // num_agents
            n_gpus = total_n_gpus
            if nodes_per_agent <= 0:
                raise ValueError(
                    f"nodes_per_agent must be >=1; got nnodes={nnodes} with num_agents={num_agents}. "
                    f"Increase nnodes or reduce num_agents."
                )
            # add actor mapping
            resource_pool_spec[f"agent_pool_{i}"] = [n_gpus] * nodes_per_agent
            self.mapping[f"agent_pool_{i}"] = f"agent_pool_{i}"
            # self.mapping[f"agent_pool_{i}"] = global_pool_id
            # add critic mapping
            # resource_pool_spec[f"critic_pool_{i}"] = [n_gpus] * nnodes
            # self.mapping[f"critic_pool_{i}"] = f"critic_pool_{i}"
            self.mapping[f"critic_pool_{i}"] = f"agent_pool_{i}"
            # self.mapping[f"critic_pool_{i}"] = global_pool_id
        
        # self.mapping[Role.ActorRollout] = global_pool_id
        # self.mapping[Role.Critic] = global_pool_id
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager
```

Replace with:

```python
    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        from verl.trainer.ppo.ray_trainer import Role
        from verl.trainer.ppo.utils import need_critic, need_reference_policy

        # read multi-agent config
        ma = OmegaConf.select(config, "multi_agent") or {}
        num_agents = int(ma.get("num_agents", 1))
        assert num_agents >= 2, "MAPPO expected multi_agent.num_agents >= 2"
        agents_cfg = ma.get("agents", [])
        if not agents_cfg or len(agents_cfg) != num_agents:
            raise ValueError(
                "Please provide multi_agent.agents list (length == num_agents) with per-agent resource/model entries."
            )

        resource_pool_spec = {}
        colocate_count_dict = {}

        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")
            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            colocate_count_dict["reward_pool"] = 1  # only RewardModelWorker

        # Divide the cluster's total nodes equally among agents.
        # nnodes is always global (config.trainer.nnodes); per-agent n_gpus_per_node may differ.
        nodes_per_agent = config.trainer.nnodes // num_agents
        if nodes_per_agent <= 0:
            raise ValueError(
                f"nodes_per_agent must be >=1; got trainer.nnodes={config.trainer.nnodes} "
                f"with num_agents={num_agents}. Increase trainer.nnodes or reduce num_agents."
            )

        # Compute actual co-location count: actor_rollout always, plus critic and ref if enabled.
        colocate_per_agent = 1 + int(need_critic(config)) + int(need_reference_policy(config))

        for i in range(num_agents):
            a = agents_cfg[i]
            n_gpus = int(a.get("n_gpus_per_node", config.trainer.n_gpus_per_node))
            resource_pool_spec[f"agent_pool_{i}"] = [n_gpus] * nodes_per_agent
            colocate_count_dict[f"agent_pool_{i}"] = colocate_per_agent
            self.mapping[f"agent_pool_{i}"] = f"agent_pool_{i}"
            self.mapping[f"critic_pool_{i}"] = f"agent_pool_{i}"

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
            colocate_count_dict=colocate_count_dict,
        )
        return resource_pool_manager
```

- [ ] **Step 2: Run all four new tests**

```bash
pytest tests/trainer/ppo/test_mappo_trainer_bugfixes.py \
    -k "colocate_count_dict or global_nnodes" -v 2>&1 | tail -15
```

Expected: all 4 PASSED.

- [ ] **Step 3: Run the full test suite to check for regressions**

```bash
pytest tests/trainer/ppo/test_mappo_trainer_bugfixes.py -v 2>&1 | tail -30
```

Expected: all tests PASSED.

---

### Task 4: Commit

- [ ] **Step 1: Stage and commit**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
git add verl/single_controller/ray/base.py \
        verl/trainer/main_mappo.py \
        tests/trainer/ppo/test_mappo_trainer_bugfixes.py
git commit -m "fix: dynamic max_colocate_count and global nnodes in MAPPO resource allocation

- ResourcePoolManager now accepts colocate_count_dict; create_resource_pool
  uses it per pool (fallback=3 keeps single-agent PPO unchanged)
- init_resource_pool_mgr always uses config.trainer.nnodes for nodes_per_agent
  instead of a broken per-agent nnodes override
- reward_pool gets colocate_count=1 (only RewardModelWorker)
- each agent_pool gets colocate_count = 1 + need_critic + need_reference_policy"
```

---

## Self-Review

**Spec coverage:**
- Bug 1 (fractional GPU / hardcoded `max_colocate_count=3`) → Task 2 adds field + Task 3 passes it → covered
- Bug 2 (`nodes_per_agent` double-divide) → Task 3 Step 1 removes per-agent nnodes read → covered
- `reward_pool` dynamic colocate count → Task 3 Step 1 sets `colocate_count_dict["reward_pool"] = 1` → covered
- Backward compatibility (single-agent PPO uses `fallback=3`) → `colocate_count_dict.get(name, 3)` → covered

**Placeholder scan:** No TBDs or vague steps. All code blocks are complete.

**Type consistency:** `colocate_count_dict` is `dict[str, int]` in both the field definition and all call sites. `ResourcePoolManager` constructor keyword `colocate_count_dict=` matches the field name exactly.
