"""Regression tests for mappo_trainer.py bugfixes (2026-04-12)."""
import inspect
import re
import pytest
import torch
from unittest.mock import MagicMock
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer, RayRiskAverseTrainer


# ---------------------------------------------------------------------------
# B1 + B5: _single_agent_rollout must not accept timing_raw
# ---------------------------------------------------------------------------

def test_single_agent_rollout_has_no_timing_raw_param():
    """timing_raw must be removed from _single_agent_rollout signature (B1)."""
    sig = inspect.signature(RayMAPPOTrainer._single_agent_rollout)
    assert "timing_raw" not in sig.parameters, (
        "_single_agent_rollout must not accept timing_raw — use a local agent_timing instead"
    )


def test_single_agent_rollout_uses_local_agent_timing():
    """Source of _single_agent_rollout must declare a local agent_timing dict (B1)."""
    src = inspect.getsource(RayMAPPOTrainer._single_agent_rollout)
    assert "agent_timing" in src, (
        "_single_agent_rollout must define a local agent_timing dict"
    )


# ---------------------------------------------------------------------------
# B4: _balance_batch must not hardcode "model_0"
# ---------------------------------------------------------------------------

def test_balance_batch_signature_has_agent_key():
    """_balance_batch must have an agent_key parameter (B4)."""
    sig = inspect.signature(RayMAPPOTrainer._balance_batch)
    assert "agent_key" in sig.parameters, (
        "_balance_batch must accept agent_key to determine world_size"
    )


def test_balance_batch_uses_agent_key_not_hardcoded():
    """_balance_batch source must not hardcode 'model_0' for world_size lookup (B4)."""
    src = inspect.getsource(RayMAPPOTrainer._balance_batch)
    assert re.search(r'actor_rollout_wgs\[agent_key\]', src), (
        "_balance_batch must use actor_rollout_wgs[agent_key], not a hardcoded key"
    )
    assert not re.search(r'actor_rollout_wgs\["model_0"\]\.world_size', src), (
        "_balance_batch must not hardcode 'model_0' for world_size"
    )


# ---------------------------------------------------------------------------
# B7: back_propogate_reward must be symmetric in RayMAPPOTrainer
# ---------------------------------------------------------------------------

def _make_score_batch(scores_at_last_token, T=4, resp_pos=3):
    B = len(scores_at_last_token)
    scores = torch.zeros(B, T)
    rmask = torch.zeros(B, T, dtype=torch.bool)
    for b, v in enumerate(scores_at_last_token):
        scores[b, resp_pos] = v
        rmask[b, resp_pos] = True
    batch = MagicMock()
    batch.batch = {"token_level_scores": scores, "response_mask": rmask}
    return batch


def test_base_back_propagate_reward_symmetric_two_agents():
    """RayMAPPOTrainer.back_propogate_reward must give each agent its own discounted return (B7)."""
    trainer = object.__new__(RayMAPPOTrainer)
    r1a0 = _make_score_batch([10.0, 20.0])   # final round, agent 0
    r1a1 = _make_score_batch([30.0, 40.0])   # final round, agent 1
    r0a0 = _make_score_batch([ 1.0,  2.0])   # round 0, agent 0
    r0a1 = _make_score_batch([ 3.0,  4.0])   # round 0, agent 1

    trainer.back_propogate_reward(
        num_rounds=2, num_agents=2,
        round_agent_batches=[[r0a0, r0a1], [r1a0, r1a1]],
        gamma=1.0,
    )

    # Agent 0: 1 + 10 = 11, 2 + 20 = 22
    assert torch.allclose(r0a0.batch["token_level_scores"][:, 3], torch.tensor([11.0, 22.0]))
    # Agent 1: 3 + 30 = 33, 4 + 40 = 44  (NOT negated — cooperative)
    assert torch.allclose(r0a1.batch["token_level_scores"][:, 3], torch.tensor([33.0, 44.0]))
    # Final round unchanged
    assert torch.allclose(r1a0.batch["token_level_scores"][:, 3], torch.tensor([10.0, 20.0]))
    assert torch.allclose(r1a1.batch["token_level_scores"][:, 3], torch.tensor([30.0, 40.0]))


def test_base_back_propagate_reward_works_with_three_agents():
    """RayMAPPOTrainer.back_propogate_reward must not assert num_agents == 2 (B7)."""
    trainer = object.__new__(RayMAPPOTrainer)
    # 2 rounds, 3 agents
    r1 = [_make_score_batch([10.0]) for _ in range(3)]
    r0 = [_make_score_batch([ 1.0]) for _ in range(3)]

    # Must not raise
    trainer.back_propogate_reward(
        num_rounds=2, num_agents=3,
        round_agent_batches=[r0, r1],
        gamma=1.0,
    )

    for a in range(3):
        assert torch.allclose(
            r0[a].batch["token_level_scores"][:, 3], torch.tensor([11.0])
        ), f"Agent {a} should have accumulated discounted return"


def test_base_back_propagate_reward_gamma_scaling():
    """gamma < 1.0 scales each agent's future reward independently (B7)."""
    trainer = object.__new__(RayMAPPOTrainer)
    r2a0 = _make_score_batch([100.0])
    r2a1 = _make_score_batch([200.0])
    r1a0 = _make_score_batch([  0.0])
    r1a1 = _make_score_batch([  0.0])
    r0a0 = _make_score_batch([  0.0])
    r0a1 = _make_score_batch([  0.0])

    # r=1: a0 = 0 + 0.5*100 = 50;  a1 = 0 + 0.5*200 = 100
    # r=0: a0 = 0 + 0.5*50  = 25;  a1 = 0 + 0.5*100 = 50
    trainer.back_propogate_reward(
        num_rounds=3, num_agents=2,
        round_agent_batches=[[r0a0, r0a1], [r1a0, r1a1], [r2a0, r2a1]],
        gamma=0.5,
    )

    assert torch.allclose(r1a0.batch["token_level_scores"][:, 3], torch.tensor([50.0]))
    assert torch.allclose(r1a1.batch["token_level_scores"][:, 3], torch.tensor([100.0]))
    assert torch.allclose(r0a0.batch["token_level_scores"][:, 3], torch.tensor([25.0]))
    assert torch.allclose(r0a1.batch["token_level_scores"][:, 3], torch.tensor([50.0]))


# ---------------------------------------------------------------------------
# B7: RayRiskAverseTrainer must have its own adversarial back_propogate_reward
# ---------------------------------------------------------------------------

def test_risk_averse_has_own_back_propogate_reward():
    """RayRiskAverseTrainer must define its own back_propogate_reward (B7)."""
    assert "back_propogate_reward" in RayRiskAverseTrainer.__dict__, (
        "RayRiskAverseTrainer must override back_propogate_reward with adversarial logic"
    )


def test_risk_averse_back_propagate_reward_adversarial_values():
    """RayRiskAverseTrainer.back_propogate_reward: agent 0=adversary gets negated hero reward; agent 1=hero accumulates (B7)."""
    trainer = object.__new__(RayRiskAverseTrainer)
    r1a0 = _make_score_batch([10.0, 20.0])   # final round, adversary (agent 0) — unchanged
    r1a1 = _make_score_batch([30.0, 40.0])   # final round, hero (agent 1)
    r0a0 = _make_score_batch([ 1.0,  2.0])   # round 0, adversary
    r0a1 = _make_score_batch([ 3.0,  4.0])   # round 0, hero

    trainer.back_propogate_reward(
        num_rounds=2, num_agents=2,
        round_agent_batches=[[r0a0, r0a1], [r1a0, r1a1]],
        gamma=1.0,
    )

    # Hero (agent 1) at r=0: 3 + 30 = 33, 4 + 40 = 44
    assert torch.allclose(r0a1.batch["token_level_scores"][:, 3], torch.tensor([33.0, 44.0]))
    # Adversary (agent 0) at r=0: -hero's next round = [-30, -40]
    assert torch.allclose(r0a0.batch["token_level_scores"][:, 3], torch.tensor([-30.0, -40.0]))
    # Final round unchanged
    assert torch.allclose(r1a0.batch["token_level_scores"][:, 3], torch.tensor([10.0, 20.0]))
    assert torch.allclose(r1a1.batch["token_level_scores"][:, 3], torch.tensor([30.0, 40.0]))


def test_risk_averse_mappo_fit_calls_back_propogate_reward():
    """RayRiskAverseTrainer.mappo_fit must call self.back_propogate_reward (B7)."""
    src = inspect.getsource(RayRiskAverseTrainer.mappo_fit)
    assert re.search(r"self\.back_propogate_reward\(", src), (
        "RayRiskAverseTrainer.mappo_fit must call self.back_propogate_reward "
        "instead of inlining score negation"
    )


# ---------------------------------------------------------------------------
# B2: apply_kl_penalty must not be inside a ThreadPoolExecutor in mappo_fit
# ---------------------------------------------------------------------------

def test_kl_penalty_not_submitted_to_thread_pool_base():
    """apply_kl_penalty must not be submitted to a ThreadPoolExecutor in RayMAPPOTrainer.mappo_fit (B2)."""
    src = inspect.getsource(RayMAPPOTrainer.mappo_fit)
    assert not re.search(r"executor\.submit\([^)]*apply_kl_penalty", src, re.DOTALL), (
        "apply_kl_penalty must be called serially — kl_ctrl.update() is not thread-safe"
    )


# ---------------------------------------------------------------------------
# B5: step_durations populated from agent_timing in base mappo_fit
# ---------------------------------------------------------------------------

def test_step_durations_populated_from_agent_timing_base():
    """mappo_fit must unpack agent_timing from _single_agent_rollout and populate step_durations (B5)."""
    src = inspect.getsource(RayMAPPOTrainer.mappo_fit)
    assert "agent_timing" in src, (
        "mappo_fit must unpack agent_timing from _single_agent_rollout results"
    )
    assert re.search(r"step_durations\.append", src), (
        "mappo_fit must append to step_durations from agent_timing"
    )


# ---------------------------------------------------------------------------
# B3: _update_metrics must not be inside ThreadPoolExecutor in RayRiskAverseTrainer.mappo_fit
# ---------------------------------------------------------------------------

def test_update_metrics_not_in_thread_pool_risk_averse():
    """_update_metrics must not be submitted to a ThreadPoolExecutor in RayRiskAverseTrainer.mappo_fit (B3)."""
    src = inspect.getsource(RayRiskAverseTrainer.mappo_fit)
    assert not re.search(r"executor\.submit\([^)]*_update_metrics", src, re.DOTALL), (
        "_update_metrics must be called serially — metrics dict is not thread-safe"
    )


# ---------------------------------------------------------------------------
# B6: _run_single_agent dead code must be removed
# ---------------------------------------------------------------------------

def test_run_single_agent_removed():
    """_run_single_agent is dead code and must be removed (B6)."""
    assert not hasattr(RayMAPPOTrainer, "_run_single_agent"), (
        "_run_single_agent is dead code — it must be deleted"
    )


# ---------------------------------------------------------------------------
# B6 (timing keys): _update_critic/_update_actor must use per-agent timing keys
# ---------------------------------------------------------------------------

def test_update_critic_uses_per_agent_timing_key():
    """_update_critic must write per-agent timing key, not shared 'update_critic' (B6-timing)."""
    src = inspect.getsource(RayMAPPOTrainer._update_critic)
    assert not re.search(r'timing_raw\["update_critic"\]', src), (
        "_update_critic must not write to a shared 'update_critic' key"
    )


def test_update_actor_uses_per_agent_timing_key():
    """_update_actor must write per-agent timing key, not shared 'update_actor' (B6-timing)."""
    src = inspect.getsource(RayMAPPOTrainer._update_actor)
    assert not re.search(r'timing_raw\["update_actor"\]', src), (
        "_update_actor must not write to a shared 'update_actor' key"
    )


# ---------------------------------------------------------------------------
# C1: _apply_kl_penalty docstring must explain adversarial KL semantics
# ---------------------------------------------------------------------------

def test_apply_kl_penalty_docstring_mentions_adversary_constraint():
    """_apply_kl_penalty docstring must state that data is the adversary constrained to data_ref (hero)."""
    src = inspect.getsource(RayRiskAverseTrainer._apply_kl_penalty)
    assert re.search(r"adversar", src, re.IGNORECASE), (
        "_apply_kl_penalty docstring must explain the adversarial KL semantics: "
        "agent 0 (adversary) is constrained to stay close to agent 1 (hero)'s policy"
    )


def test_risk_averse_mappo_fit_comment_identifies_agent0_as_adversary():
    """RayRiskAverseTrainer.mappo_fit must have a comment naming agent 0 as the adversary."""
    src = inspect.getsource(RayRiskAverseTrainer.mappo_fit)
    assert re.search(r"agent 0.*adversar|adversar.*agent 0", src, re.IGNORECASE), (
        "RayRiskAverseTrainer.mappo_fit must clarify that agent 0 is the adversary "
        "(the comment added in Task 2 should satisfy this)"
    )


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
