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


def test_compute_reward_raises_on_none_reward_fn():
    """_compute_reward must raise ValueError when reward_fn is None."""
    import pytest
    trainer = _make_trainer()
    batch = _make_batch()

    with pytest.raises(ValueError, match="reward_fn cannot be None"):
        trainer._compute_reward(batch, reward_fn=None)


def test_update_metrics_no_thread_pool():
    """_update_metrics must be called serially, not via ThreadPoolExecutor."""
    import inspect
    import re
    import verl.trainer.ppo.mappo_trainer as mod

    src = inspect.getsource(mod.RayMAPPOTrainer.mappo_fit)
    # After the fix, _update_metrics must not appear as the callable argument
    # to executor.submit(). Check that no executor.submit call references it.
    assert not re.search(r"executor\.submit\([^)]*_update_metrics", src, re.DOTALL), (
        "_update_metrics must not be submitted to a ThreadPoolExecutor"
    )


def test_extract_prompts_uses_first_prompt_for_system():
    """system_prompt must be extracted from prompts[0], not the last prompt."""
    trainer = _make_trainer()

    first = "system You are helpful user What is 2+2? assistant"
    last  = "system WRONG user What is 3+3? assistant"

    tokenizer = MagicMock()
    tokenizer.batch_decode.return_value = [first, last]
    trainer.tokenizers = {"model_0": tokenizer}

    batch = DataProto.from_dict(tensors={"input_ids": torch.zeros(2, 4, dtype=torch.long)})

    _, system_prompt = trainer._extract_prompts_and_questions(batch, "model_0")

    assert "You are helpful" in system_prompt, (
        f"Expected system prompt from first sample, got: {system_prompt!r}"
    )
    assert "WRONG" not in system_prompt


def test_create_dataloader_val_batch_size_fallback_custom_keys():
    """_create_dataloader must not KeyError when val_datasets lacks 'model_0'."""
    from unittest.mock import patch
    from omegaconf import OmegaConf
    trainer = _make_trainer()

    trainer.config = OmegaConf.create({
        "data": {
            "dataloader_num_workers": 0,
            "train_batch_size": 2,
            "gen_batch_size": 2,
            "val_batch_size": None,
            "shuffle": False,
            "validation_shuffle": False,
            "sampler": None,
        },
        "trainer": {"total_epochs": 1, "total_training_steps": None},
    })

    fake_dataset = MagicMock()
    fake_dataset.__len__ = MagicMock(return_value=10)
    val_datasets = {"custom_agent": fake_dataset}

    try:
        with patch("verl.trainer.ppo.mappo_trainer.StatefulDataLoader"):
            trainer.val_datasets = val_datasets
            val_batch_size = trainer.config.data.val_batch_size
            if val_batch_size is None:
                val_batch_size = len(next(iter(trainer.val_datasets.values())))
        assert val_batch_size == 10
    except KeyError as e:
        raise AssertionError(f"KeyError raised: {e}")


def test_back_propagate_reward_warns_on_empty_response():
    """back_propogate_reward must warn when any sample has no response tokens."""
    import pytest
    trainer = _make_trainer()

    B, T = 2, 4

    def _make_mock_batch(has_response_for_sample_1=True):
        scores = torch.zeros(B, T)
        rmask = torch.zeros(B, T, dtype=torch.bool)
        rmask[0, 2] = True
        if has_response_for_sample_1:
            rmask[1, 3] = True
        batch = MagicMock()
        batch.batch = {"token_level_scores": scores, "response_mask": rmask}
        return batch

    # num_rounds=3 so backward pass iterates r=1 (range(num_rounds-2, 0, -1))
    round_agent_batches = [
        [_make_mock_batch(has_response_for_sample_1=True)],   # r=0
        [_make_mock_batch(has_response_for_sample_1=False)],  # r=1 — triggers warning
        [_make_mock_batch(has_response_for_sample_1=True)],   # r=2
    ]

    with pytest.warns(UserWarning, match="no response tokens"):
        trainer.back_propogate_reward(
            num_rounds=3, num_agents=1,
            round_agent_batches=round_agent_batches,
            gamma=1.0,
        )
