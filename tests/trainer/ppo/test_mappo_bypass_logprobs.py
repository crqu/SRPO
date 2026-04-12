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
