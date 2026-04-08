"""Unit tests for RayMAPPOTrainer._compute_reward."""
import torch
from unittest.mock import MagicMock
from verl import DataProto


def _make_trainer():
    """Construct a minimal RayMAPPOTrainer without Ray/GPU."""
    from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer
    trainer = object.__new__(RayMAPPOTrainer)
    return trainer


def _make_batch():
    return DataProto.from_dict(tensors={"input_ids": torch.zeros(2, 4, dtype=torch.long)})


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
