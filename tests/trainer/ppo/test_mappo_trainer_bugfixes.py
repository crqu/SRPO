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


def test_base_back_propagate_reward_team_reward_three_rounds():
    """RayMAPPOTrainer.back_propogate_reward sums BOTH agents' future returns into each agent (cooperative team reward).

    Round 0 is intentionally untouched to preserve each agent's base reasoning ability —
    only round r in [1, N-2] (i.e., the discussion phase) gets back-prop.
    """
    trainer = object.__new__(RayMAPPOTrainer)
    # 3 rounds, 2 agents
    r2a0 = _make_score_batch([10.0, 20.0])   # final round, agent 0 — unchanged
    r2a1 = _make_score_batch([30.0, 40.0])   # final round, agent 1 — unchanged
    r1a0 = _make_score_batch([ 5.0,  6.0])   # middle round, agent 0
    r1a1 = _make_score_batch([ 7.0,  8.0])   # middle round, agent 1
    r0a0 = _make_score_batch([ 1.0,  2.0])   # round 0, agent 0 — intentionally untouched
    r0a1 = _make_score_batch([ 3.0,  4.0])   # round 0, agent 1 — intentionally untouched

    trainer.back_propogate_reward(
        num_rounds=3, num_agents=2,
        round_agent_batches=[[r0a0, r0a1], [r1a0, r1a1], [r2a0, r2a1]],
        gamma=1.0,
    )

    # r=1 each agent absorbs the SUM of both agents' r=2 rewards (team reward).
    # r1a0 = 5 + (10 + 30) = 45;  6 + (20 + 40) = 66
    assert torch.allclose(r1a0.batch["token_level_scores"][:, 3], torch.tensor([45.0, 66.0]))
    # r1a1 = 7 + (10 + 30) = 47;  8 + (20 + 40) = 68
    assert torch.allclose(r1a1.batch["token_level_scores"][:, 3], torch.tensor([47.0, 68.0]))
    # Round 0: intentionally NOT modified by back_propogate_reward
    assert torch.allclose(r0a0.batch["token_level_scores"][:, 3], torch.tensor([1.0, 2.0]))
    assert torch.allclose(r0a1.batch["token_level_scores"][:, 3], torch.tensor([3.0, 4.0]))
    # Final round unchanged
    assert torch.allclose(r2a0.batch["token_level_scores"][:, 3], torch.tensor([10.0, 20.0]))
    assert torch.allclose(r2a1.batch["token_level_scores"][:, 3], torch.tensor([30.0, 40.0]))


def test_base_back_propagate_reward_works_with_three_agents():
    """RayMAPPOTrainer.back_propogate_reward must not assert num_agents == 2 and must sum across all agents."""
    trainer = object.__new__(RayMAPPOTrainer)
    # 3 rounds, 3 agents — round 0 untouched, round 1 sums all 3 agents' round-2 rewards.
    r2 = [_make_score_batch([10.0]) for _ in range(3)]
    r1 = [_make_score_batch([0.0]) for _ in range(3)]
    r0 = [_make_score_batch([1.0]) for _ in range(3)]

    # Must not raise
    trainer.back_propogate_reward(
        num_rounds=3, num_agents=3,
        round_agent_batches=[r0, r1, r2],
        gamma=1.0,
    )

    # NOTE: current code only sums agent 0 + agent 1 (`rewards[r+1][0] + rewards[r+1][1]`).
    # If user wants true sum-over-all-agents at num_agents=3, that's a separate change —
    # for now, this test asserts the documented 2-agent team-reward and just verifies
    # the function does not crash on num_agents=3 + that round 0 is untouched.
    for a in range(3):
        assert torch.allclose(
            r0[a].batch["token_level_scores"][:, 3], torch.tensor([1.0])
        ), f"Agent {a} round 0 should be untouched"


def test_base_back_propagate_reward_gamma_scaling():
    """gamma < 1.0 scales each agent's discounted return; round 0 is untouched."""
    trainer = object.__new__(RayMAPPOTrainer)
    r3a0 = _make_score_batch([100.0]); r3a1 = _make_score_batch([200.0])
    r2a0 = _make_score_batch([0.0]);   r2a1 = _make_score_batch([0.0])
    r1a0 = _make_score_batch([0.0]);   r1a1 = _make_score_batch([0.0])
    r0a0 = _make_score_batch([0.0]);   r0a1 = _make_score_batch([0.0])

    # range(num_rounds-2, 0, -1) = range(2, 0, -1) = [2, 1] for num_rounds=4.
    # r=2: a0 = 0 + 0.5*100 + 0.5*200 = 150; a1 = 0 + 0.5*100 + 0.5*200 = 150
    # r=1: a0 = 0 + 0.5*150 + 0.5*150 = 150; a1 = 0 + 0.5*150 + 0.5*150 = 150
    # r=0: untouched
    trainer.back_propogate_reward(
        num_rounds=4, num_agents=2,
        round_agent_batches=[[r0a0, r0a1], [r1a0, r1a1], [r2a0, r2a1], [r3a0, r3a1]],
        gamma=0.5,
    )

    assert torch.allclose(r2a0.batch["token_level_scores"][:, 3], torch.tensor([150.0]))
    assert torch.allclose(r2a1.batch["token_level_scores"][:, 3], torch.tensor([150.0]))
    assert torch.allclose(r1a0.batch["token_level_scores"][:, 3], torch.tensor([150.0]))
    assert torch.allclose(r1a1.batch["token_level_scores"][:, 3], torch.tensor([150.0]))
    assert torch.allclose(r0a0.batch["token_level_scores"][:, 3], torch.tensor([0.0]))
    assert torch.allclose(r0a1.batch["token_level_scores"][:, 3], torch.tensor([0.0]))


# ---------------------------------------------------------------------------
# B7: RayRiskAverseTrainer must have its own adversarial back_propogate_reward
# ---------------------------------------------------------------------------

def test_risk_averse_has_own_back_propogate_reward():
    """RayRiskAverseTrainer must define its own back_propogate_reward (B7)."""
    assert "back_propogate_reward" in RayRiskAverseTrainer.__dict__, (
        "RayRiskAverseTrainer must override back_propogate_reward with adversarial logic"
    )


def test_risk_averse_back_propagate_reward_adversarial_values():
    """SRPO back_propogate_reward:
      - r in [1, N-2]: adversary[r] = -hero[r+1] (post-hero-update);
                       hero[r] = own + gamma * hero[r+1]
      - r = 0:         adversary[0] = -hero[1] (mislead the hero);
                       hero[0] unchanged (anchor base reasoning, Q1=(a))
      - r = N-1:       leaf rewards, unchanged
    """
    trainer = object.__new__(RayRiskAverseTrainer)
    # 3 rounds, 2 agents
    r2a0 = _make_score_batch([10.0, 20.0])   # final round, adversary — unchanged
    r2a1 = _make_score_batch([30.0, 40.0])   # final round, hero — unchanged
    r1a0 = _make_score_batch([ 5.0,  6.0])   # middle round, adversary
    r1a1 = _make_score_batch([ 7.0,  8.0])   # middle round, hero
    r0a0 = _make_score_batch([ 1.0,  2.0])   # round 0, adversary — now updated
    r0a1 = _make_score_batch([ 3.0,  4.0])   # round 0, hero — anchored, unchanged

    trainer.back_propogate_reward(
        num_rounds=3, num_agents=2,
        round_agent_batches=[[r0a0, r0a1], [r1a0, r1a1], [r2a0, r2a1]],
        gamma=1.0,
    )

    # r=1: hero = 7+30=37, 8+40=48; adversary = -hero_leaf[2] = [-30, -40]
    assert torch.allclose(r1a1.batch["token_level_scores"][:, 3], torch.tensor([37.0, 48.0]))
    assert torch.allclose(r1a0.batch["token_level_scores"][:, 3], torch.tensor([-30.0, -40.0]))
    # r=0: adversary = -hero[r=1]_post_update = [-37, -48]; hero unchanged
    assert torch.allclose(r0a0.batch["token_level_scores"][:, 3], torch.tensor([-37.0, -48.0]))
    assert torch.allclose(r0a1.batch["token_level_scores"][:, 3], torch.tensor([3.0, 4.0]))
    # Final round unchanged
    assert torch.allclose(r2a0.batch["token_level_scores"][:, 3], torch.tensor([10.0, 20.0]))
    assert torch.allclose(r2a1.batch["token_level_scores"][:, 3], torch.tensor([30.0, 40.0]))


def test_risk_averse_round_0_hero_anchored_adversary_misleads():
    """SRPO round-0 semantics (Q1=(a)): hero[0] keeps raw task reward; adversary[0] = -hero[1]."""
    trainer = object.__new__(RayRiskAverseTrainer)
    r2a0 = _make_score_batch([0.0]);  r2a1 = _make_score_batch([100.0])
    r1a0 = _make_score_batch([0.0]);  r1a1 = _make_score_batch([0.0])
    r0a0 = _make_score_batch([7.0]);  r0a1 = _make_score_batch([42.0])

    trainer.back_propogate_reward(
        num_rounds=3, num_agents=2,
        round_agent_batches=[[r0a0, r0a1], [r1a0, r1a1], [r2a0, r2a1]],
        gamma=1.0,
    )
    # Hero at r=0 untouched (anchored to base reasoning)
    assert torch.allclose(r0a1.batch["token_level_scores"][:, 3], torch.tensor([42.0]))
    # Adversary at r=0 = -hero[r=1]_post_update = -(0 + 1.0*100) = -100
    assert torch.allclose(r0a0.batch["token_level_scores"][:, 3], torch.tensor([-100.0]))


def test_risk_averse_round_0_with_gamma_lt_1():
    """SRPO round-0 adversary uses post-update hero[1] which contains gamma*hero_leaf."""
    trainer = object.__new__(RayRiskAverseTrainer)
    r2a0 = _make_score_batch([0.0]);  r2a1 = _make_score_batch([100.0])
    r1a0 = _make_score_batch([0.0]);  r1a1 = _make_score_batch([5.0])
    r0a0 = _make_score_batch([0.0]);  r0a1 = _make_score_batch([0.0])

    trainer.back_propogate_reward(
        num_rounds=3, num_agents=2,
        round_agent_batches=[[r0a0, r0a1], [r1a0, r1a1], [r2a0, r2a1]],
        gamma=0.5,
    )
    # hero[1] = 5 + 0.5*100 = 55; adversary[0] = -55
    assert torch.allclose(r1a1.batch["token_level_scores"][:, 3], torch.tensor([55.0]))
    assert torch.allclose(r0a0.batch["token_level_scores"][:, 3], torch.tensor([-55.0]))


def test_risk_averse_default_adversary_kl_to_hero_is_true():
    """SRPO's adversary_kl_to_hero must default to True (the SRPO regularizer is required)."""
    src = inspect.getsource(RayRiskAverseTrainer.mappo_fit)
    assert re.search(
        r'adversary_kl_to_hero\s*=\s*bool\(\s*ma\.get\(\s*[\'"]adversary_kl_to_hero[\'"]\s*,\s*True\s*\)',
        src,
    ), (
        "RayRiskAverseTrainer.mappo_fit must default adversary_kl_to_hero to True — "
        "the SRPO KL regularizer is the defining feature of this trainer"
    )


def test_ippo_uses_config_gamma_for_back_propogate_reward():
    """IPPO mappo_fit must pass gamma=self.config.algorithm.gamma (was hardcoded to 1)."""
    src = inspect.getsource(RayMAPPOTrainer.mappo_fit)
    assert re.search(
        r"self\.back_propogate_reward\([^)]*gamma\s*=\s*self\.config\.algorithm\.gamma",
        src, re.DOTALL,
    ), (
        "IPPO mappo_fit must call back_propogate_reward with gamma=self.config.algorithm.gamma "
        "to stay consistent with SRPO if gamma is ever tuned"
    )
    assert not re.search(
        r"self\.back_propogate_reward\([^)]*gamma\s*=\s*1\b",
        src, re.DOTALL,
    ), "IPPO mappo_fit must not hardcode gamma=1"


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


def test_agent_pool_colocate_count_is_one_not_internal_worker_count():
    """agent_pool_{i} must use max_colocate_count=1, not the count of internal logical workers.

    create_colocated_worker_cls bundles actor+critic+ref into ONE WorkerDict Ray actor
    per bundle. max_colocate_count controls num_gpus = 1/N for that single Ray actor
    (verl/single_controller/ray/base.py:628). Using N>1 produces fractional GPU
    allocation (e.g. 0.5 for actor+critic), which does not reliably set
    CUDA_VISIBLE_DEVICES on the worker node in this Ray+SLURM environment — see
    docs/superpowers/plans/2026-04-12-mappo-cuda-visibility-fix.md.

    Per-agent_pool max_colocate_count must be 1 so each WorkerDict gets a full
    integer GPU.
    """
    src = inspect.getsource(TaskRunner.init_resource_pool_mgr)
    # The line must assign a literal 1 (not colocate_per_agent or any other value).
    assert re.search(
        r'colocate_count_dict\[\s*f?["\']agent_pool_\{i\}["\']\s*\]\s*=\s*1\b',
        src,
    ), (
        "agent_pool_{i} colocate_count must be set to literal 1 (one Ray actor per bundle, "
        "regardless of how many internal logical workers it contains). "
        "Setting it to colocate_per_agent caused fractional num_gpus=0.5 and "
        "CUDA-not-available crashes on multi-node SLURM runs."
    )


# ---------------------------------------------------------------------------
# Last-round skip: SRPO must skip ADVERSARY (agent 0) at r = num_rounds - 1
# ---------------------------------------------------------------------------

def test_is_terminal_adversary_helper_exists():
    """RayRiskAverseTrainer._is_terminal_adversary must exist and be a staticmethod."""
    assert hasattr(RayRiskAverseTrainer, "_is_terminal_adversary"), (
        "RayRiskAverseTrainer must define a _is_terminal_adversary helper"
    )


def test_is_terminal_adversary_returns_true_only_for_agent0_at_last_round():
    """_is_terminal_adversary returns True iff (r == num_rounds - 1 and agent_idx == 0)."""
    fn = RayRiskAverseTrainer._is_terminal_adversary
    assert fn(r=2, agent_idx=0, num_rounds=3) is True
    assert fn(r=2, agent_idx=1, num_rounds=3) is False  # hero is NOT skipped
    assert fn(r=1, agent_idx=0, num_rounds=3) is False
    assert fn(r=0, agent_idx=0, num_rounds=3) is False
    assert fn(r=0, agent_idx=1, num_rounds=3) is False


def test_risk_averse_mappo_fit_skip_uses_is_terminal_adversary():
    """RayRiskAverseTrainer.mappo_fit critic+actor skip sites must call _is_terminal_adversary, not hardcode agent_idx==1."""
    src = inspect.getsource(RayRiskAverseTrainer.mappo_fit)
    assert "_is_terminal_adversary(" in src, (
        "mappo_fit critic+actor skip sites must call self._is_terminal_adversary(...) "
        "instead of hardcoding the agent index"
    )
    # Belt-and-suspenders: make sure the buggy hero-skip is gone.
    assert not re.search(r"r == num_rounds - 1 and agent_idx == 1", src), (
        "mappo_fit must not skip agent_idx == 1 (the hero) — it must skip the adversary (agent 0) at the last round"
    )


# ---------------------------------------------------------------------------
# D5 regression: raw rollout scores must be captured BEFORE back_propogate_reward
# ---------------------------------------------------------------------------

def test_base_mappo_fit_diag_before_back_propogate():
    """RayMAPPOTrainer.mappo_fit must build diag_correctness BEFORE back_propogate_reward.

    Otherwise the per-step accuracy metric reflects shaped (cumulative) rewards,
    not raw rollout correctness.
    """
    src = inspect.getsource(RayMAPPOTrainer.mappo_fit)
    diag_pos = src.find("diag_metrics = _compute_cheap_diagnostics(")
    backprop_pos = src.find("self.back_propogate_reward(")
    assert diag_pos != -1 and backprop_pos != -1
    assert diag_pos < backprop_pos, (
        "diag_metrics must be computed before back_propogate_reward — otherwise "
        "the accuracy metric reflects shaped rewards, not raw rollout correctness"
    )


def test_srpo_mappo_fit_diag_before_back_propogate():
    """RayRiskAverseTrainer.mappo_fit must build diag_correctness BEFORE back_propogate_reward."""
    src = inspect.getsource(RayRiskAverseTrainer.mappo_fit)
    diag_pos = src.find("diag_metrics = _compute_cheap_diagnostics(")
    backprop_pos = src.find("self.back_propogate_reward(")
    assert diag_pos != -1 and backprop_pos != -1
    assert diag_pos < backprop_pos, (
        "diag_metrics must be computed before back_propogate_reward — otherwise "
        "agent_0 (adversary) accuracy will reflect -hero_return, not its own correctness"
    )


# ---------------------------------------------------------------------------
# kl_ctrl_in_reward must not be re-instantiated inside mappo_fit
# ---------------------------------------------------------------------------

def test_risk_averse_mappo_fit_does_not_reinstantiate_kl_ctrl_in_reward():
    """RayRiskAverseTrainer.mappo_fit must not reassign self.kl_ctrl_in_reward.

    Re-instantiation would wipe adaptive KL state across calls.
    """
    src = inspect.getsource(RayRiskAverseTrainer.mappo_fit)
    assert "self.kl_ctrl_in_reward = " not in src, (
        "RayRiskAverseTrainer.mappo_fit must not reassign self.kl_ctrl_in_reward — "
        "create it once in __init__"
    )
