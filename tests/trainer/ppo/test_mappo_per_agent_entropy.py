"""Verify per-agent actor.entropy_coeff override merges correctly via Hydra.

Locks the contract that `+multi_agent.agents.{i}.actor.actor.entropy_coeff`
overrides `actor_rollout_ref.actor.entropy_coeff` for agent i only. The `+`
prefix is required because the agents schema only declares `actor.model.path`
and `n_gpus_per_node`; adding nested actor keys is a Hydra append, not a set.

Spec: docs/superpowers/specs/2026-04-25-srpo-revisions-design.md §5.
"""

from copy import deepcopy

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
import os


CONFIG_DIR = os.path.abspath("verl/trainer/config")


def _compose(overrides):
    """Compose mappo_trainer.yaml with the given overrides; return OmegaConf cfg."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="mappo_trainer", overrides=overrides)


def _merge_per_agent(cfg, i):
    """Mirror the merge done in RayMAPPOTrainer.init_workers (mappo_trainer.py:751-754)."""
    per_agent_override = OmegaConf.select(cfg, f"multi_agent.agents.{i}") or {}
    per_agent_actor_override = per_agent_override.get("actor", {})
    return OmegaConf.merge(deepcopy(cfg.actor_rollout_ref), per_agent_actor_override)


def test_no_override_uses_top_level_default():
    """Without any per-agent override, both agents inherit actor.entropy_coeff=0.01."""
    cfg = _compose([
        "multi_agent.agents.0.actor.model.path=stub-0",
        "multi_agent.agents.1.actor.model.path=stub-1",
    ])
    a0 = _merge_per_agent(cfg, 0)
    a1 = _merge_per_agent(cfg, 1)
    assert float(a0.actor.entropy_coeff) == 0.01
    assert float(a1.actor.entropy_coeff) == 0.01


def test_srpo_main_override_only_agent_0():
    """srpo_main runner injects 0.05 for agent 0; agent 1 stays at 0.01."""
    cfg = _compose([
        "multi_agent.agents.0.actor.model.path=stub-0",
        "multi_agent.agents.1.actor.model.path=stub-1",
        "+multi_agent.agents.0.actor.actor.entropy_coeff=0.05",
    ])
    a0 = _merge_per_agent(cfg, 0)
    a1 = _merge_per_agent(cfg, 1)
    assert float(a0.actor.entropy_coeff) == 0.05
    assert float(a1.actor.entropy_coeff) == 0.01


def test_override_does_not_leak_across_agents():
    """Setting agent-1 override must not affect agent-0's config."""
    cfg = _compose([
        "multi_agent.agents.0.actor.model.path=stub-0",
        "multi_agent.agents.1.actor.model.path=stub-1",
        "+multi_agent.agents.1.actor.actor.entropy_coeff=0.07",
    ])
    a0 = _merge_per_agent(cfg, 0)
    a1 = _merge_per_agent(cfg, 1)
    assert float(a0.actor.entropy_coeff) == 0.01
    assert float(a1.actor.entropy_coeff) == 0.07
