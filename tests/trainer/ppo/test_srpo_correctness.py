"""SRPO-specific algorithmic correctness tests for RayRiskAverseTrainer."""
import copy
import inspect
import re
from unittest.mock import MagicMock

import torch

from verl.trainer.ppo.mappo_trainer import RayRiskAverseTrainer


def _make_adv_data(adv_log_probs, scores_at_last_token, T=4, resp_pos=3):
    """Build a minimal DataProto-like object for _apply_kl_penalty."""
    B = adv_log_probs.shape[0]
    response_mask = torch.ones(B, T)
    scores = torch.zeros(B, T)
    for b, v in enumerate(scores_at_last_token):
        scores[b, resp_pos] = v
    obj = MagicMock()
    obj.batch = {
        "old_log_probs": adv_log_probs,
        "response_mask": response_mask,
        "token_level_scores": scores,
    }
    obj.meta_info = {
        "micro_batch_size": 99,
        "max_token_len": 1234,
        "use_dynamic_bsz": False,
        "temperature": 0.7,
    }
    return obj


def _make_mock_hero_wg(hero_log_probs):
    """Mock hero actor worker group whose compute_log_prob mutates meta_info (like the real fsdp_workers)
    and returns the given hero log probs.
    """

    def compute_log_prob(adv_data):
        adv_data.meta_info["micro_batch_size"] = -1
        adv_data.meta_info["max_token_len"] = -1
        adv_data.meta_info["use_dynamic_bsz"] = True
        adv_data.meta_info["temperature"] = -1.0
        out = MagicMock()
        out.batch = {"old_log_probs": hero_log_probs}
        return out

    wg = MagicMock()
    wg.compute_log_prob = compute_log_prob
    return wg


def test_apply_kl_penalty_takes_risk_coef_directly():
    """_apply_kl_penalty signature must accept risk_coef (no kl_ctrl)."""
    sig = inspect.signature(RayRiskAverseTrainer._apply_kl_penalty)
    params = sig.parameters
    assert "risk_coef" in params, (
        "_apply_kl_penalty must take risk_coef as a fixed scalar, replacing the kl_ctrl argument"
    )
    assert "kl_ctrl" not in params, (
        "_apply_kl_penalty must not take kl_ctrl — risk_coef is the fixed coefficient"
    )


def test_apply_kl_penalty_kl_direction_and_sign():
    """KL math: token_level_rewards[i, t] = scores[i, t] - beta * (log_adv[i, t] - log_hero[i, t]) on response tokens."""
    trainer = object.__new__(RayRiskAverseTrainer)

    adv_lp = torch.tensor(
        [[-1.0, -2.0, -3.0, -4.0],
         [-0.5, -1.5, -2.5, -3.5]]
    )
    hero_lp = torch.tensor(
        [[-1.5, -2.5, -3.5, -4.5],
         [-1.0, -2.0, -3.0, -4.0]]
    )
    expected_kld = torch.full((2, 4), 0.5)  # k1: log_adv - log_hero = 0.5 everywhere

    adv_data = _make_adv_data(adv_log_probs=adv_lp, scores_at_last_token=[10.0, 20.0])
    hero_wg = _make_mock_hero_wg(hero_log_probs=hero_lp)

    out, metrics = trainer._apply_kl_penalty(
        adv_data=adv_data,
        hero_actor_wg=hero_wg,
        risk_coef=2.0,
        kl_penalty="kl",
    )

    expected_rewards = adv_data.batch["token_level_scores"] - 2.0 * expected_kld
    assert torch.allclose(out.batch["token_level_rewards"], expected_rewards), (
        "_apply_kl_penalty must compute scores - risk_coef * (log_adv - log_hero) on response tokens"
    )
    assert "actor/reward_kl_penalty" in metrics
    assert metrics["actor/reward_kl_penalty_coeff"] == 2.0


def test_apply_kl_penalty_does_not_mutate_adv_data_meta_info():
    """_apply_kl_penalty must restore adv_data.meta_info to its full pre-call state."""
    trainer = object.__new__(RayRiskAverseTrainer)
    adv_lp = torch.zeros(1, 4)
    hero_lp = torch.zeros(1, 4)
    adv_data = _make_adv_data(adv_log_probs=adv_lp, scores_at_last_token=[0.0])
    # Add a key compute_log_prob does NOT mutate, to confirm the restore is
    # full-dict (self-synchronizing) rather than a curated subset.
    adv_data.meta_info["user_custom_key"] = "preserved"

    saved = copy.deepcopy(adv_data.meta_info)
    hero_wg = _make_mock_hero_wg(hero_log_probs=hero_lp)
    trainer._apply_kl_penalty(
        adv_data=adv_data, hero_actor_wg=hero_wg, risk_coef=1.0, kl_penalty="kl"
    )

    assert adv_data.meta_info == saved, (
        "_apply_kl_penalty must restore the entire adv_data.meta_info to its pre-call state"
    )


def test_srpo_mappo_fit_passes_risk_coef_to_apply_kl_penalty():
    """mappo_fit must call self._apply_kl_penalty with risk_coef= (not kl_ctrl=)."""
    src = inspect.getsource(RayRiskAverseTrainer.mappo_fit)
    assert re.search(r"self\._apply_kl_penalty\([^)]*risk_coef\s*=", src, re.DOTALL), (
        "mappo_fit must pass risk_coef= to self._apply_kl_penalty"
    )
    assert not re.search(r"self\._apply_kl_penalty\([^)]*kl_ctrl\s*=", src, re.DOTALL), (
        "mappo_fit must not pass kl_ctrl= to self._apply_kl_penalty (risk_coef replaces it)"
    )


def test_srpo_mappo_fit_uses_algorithm_gamma_for_back_propogate_reward():
    """back_propogate_reward must receive gamma=self.config.algorithm.gamma, not risk_coef."""
    src = inspect.getsource(RayRiskAverseTrainer.mappo_fit)
    assert re.search(
        r"self\.back_propogate_reward\([^)]*gamma\s*=\s*self\.config\.algorithm\.gamma",
        src, re.DOTALL,
    ), (
        "back_propogate_reward must be called with gamma=self.config.algorithm.gamma; "
        "risk_coef is a KL coefficient, not a discount factor"
    )
    assert not re.search(
        r"self\.back_propogate_reward\([^)]*gamma\s*=\s*risk_coef",
        src, re.DOTALL,
    ), (
        "back_propogate_reward must not be called with gamma=risk_coef — that conflated "
        "two distinct hyperparameters"
    )
