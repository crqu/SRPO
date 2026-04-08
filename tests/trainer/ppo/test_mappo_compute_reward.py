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


# ---------------------------------------------------------------------------
# _build_validation_rollout_id
# ---------------------------------------------------------------------------

def test_build_validation_rollout_id_format():
    """ID must follow the pattern gs{step}_b{batch}_i{idx}_{uid}."""
    trainer = _make_trainer()
    rollout_id = trainer._build_validation_rollout_id(
        global_step=5, batch_idx=3, sample_uid="abc-123", sample_idx=7
    )
    assert rollout_id == "gs5_b3_i7_abc-123"


def test_build_validation_rollout_id_zero_indices():
    """ID works correctly when all numeric fields are 0."""
    trainer = _make_trainer()
    rollout_id = trainer._build_validation_rollout_id(
        global_step=0, batch_idx=0, sample_uid="uid-0", sample_idx=0
    )
    assert rollout_id == "gs0_b0_i0_uid-0"


# ---------------------------------------------------------------------------
# _extract_validation_correctness
# ---------------------------------------------------------------------------

def test_extract_correctness_uses_acc_key():
    """'acc' key takes priority; returned as bool with key name."""
    trainer = _make_trainer()
    is_correct, source = trainer._extract_validation_correctness(
        reward_extra_info={"acc": [1, 0], "correct": [0, 1]},
        sample_idx=0,
        score=0.0,
    )
    assert is_correct is True
    assert source == "acc"


def test_extract_correctness_falls_through_to_correct():
    """Falls through to 'correct' when 'acc' is absent."""
    trainer = _make_trainer()
    is_correct, source = trainer._extract_validation_correctness(
        reward_extra_info={"correct": [0, 1]},
        sample_idx=1,
        score=0.0,
    )
    assert is_correct is True
    assert source == "correct"


def test_extract_correctness_falls_through_to_is_correct():
    """Falls through to 'is_correct' when 'acc' and 'correct' are absent."""
    trainer = _make_trainer()
    is_correct, source = trainer._extract_validation_correctness(
        reward_extra_info={"is_correct": [False, True]},
        sample_idx=0,
        score=0.0,
    )
    assert is_correct is False
    assert source == "is_correct"


def test_extract_correctness_score_fallback():
    """Falls back to score > 0 when reward_extra_info is None."""
    trainer = _make_trainer()

    is_correct, source = trainer._extract_validation_correctness(
        reward_extra_info=None, sample_idx=0, score=0.5
    )
    assert is_correct is True
    assert source == "score>0"

    is_correct, source = trainer._extract_validation_correctness(
        reward_extra_info=None, sample_idx=0, score=-1.0
    )
    assert is_correct is False
    assert source == "score>0"


def test_extract_correctness_score_fallback_empty_dict():
    """Falls back to score > 0 when reward_extra_info has no matching key."""
    trainer = _make_trainer()
    is_correct, source = trainer._extract_validation_correctness(
        reward_extra_info={"other_key": [1]}, sample_idx=0, score=0.0
    )
    assert is_correct is False
    assert source == "score>0"


def test_extract_correctness_sample_idx_out_of_range():
    """If sample_idx >= len(values) for a key, skip that key and try next."""
    trainer = _make_trainer()
    # 'acc' has only 1 element; sample_idx=1 is out of range → fall to 'correct'
    is_correct, source = trainer._extract_validation_correctness(
        reward_extra_info={"acc": [1], "correct": [0, 0]},
        sample_idx=1,
        score=0.0,
    )
    assert source == "correct"
    assert is_correct is False


def test_extract_correctness_tensor_item():
    """Values with .item() (tensor-like) are unwrapped before bool conversion."""
    trainer = _make_trainer()

    class FakeTensor:
        def item(self):
            return 1

    is_correct, source = trainer._extract_validation_correctness(
        reward_extra_info={"acc": [FakeTensor()]},
        sample_idx=0,
        score=0.0,
    )
    assert is_correct is True
    assert source == "acc"


# ---------------------------------------------------------------------------
# _append_validation_rollout_record
# ---------------------------------------------------------------------------

def test_append_validation_rollout_record_creates_record():
    """First append creates the record with the correct top-level structure."""
    trainer = _make_trainer()
    rollout_records = {}

    trainer._append_validation_rollout_record(
        rollout_records=rollout_records,
        rollout_id="gs1_b0_i0_uid1",
        global_step=1,
        batch_idx=0,
        sample_idx=0,
        round_idx=0,
        agent_idx=0,
        agent_key="model_0",
        question="What is 2+2?",
        ground_truth="4",
        output_text="4",
        score=1.0,
        reward_extra_info={"acc": [1]},
        sample_uid="uid1",
        data_source="math",
    )

    assert "gs1_b0_i0_uid1" in rollout_records
    record = rollout_records["gs1_b0_i0_uid1"]
    assert record["global_step"] == 1
    assert record["question"] == "What is 2+2?"
    assert record["ground_truth"] == "4"
    assert record["uid"] == "uid1"
    assert record["data_source"] == "math"
    assert len(record["rows"]) == 1


def test_append_validation_rollout_record_row_values():
    """Row inserted by append has the correct field values."""
    trainer = _make_trainer()
    rollout_records = {}

    trainer._append_validation_rollout_record(
        rollout_records=rollout_records,
        rollout_id="gs2_b1_i3_uid9",
        global_step=2,
        batch_idx=1,
        sample_idx=3,
        round_idx=1,
        agent_idx=1,
        agent_key="model_1",
        question="Q?",
        ground_truth="A",
        output_text="B",
        score=0.5,
        reward_extra_info=None,
        sample_uid="uid9",
        data_source="logic",
    )

    row = rollout_records["gs2_b1_i3_uid9"]["rows"][0]
    assert row["round_idx"] == 1
    assert row["agent_idx"] == 1
    assert row["agent_key"] == "model_1"
    assert row["output_text"] == "B"
    assert row["score"] == 0.5
    # score=0.5 > 0, so is_correct must be True
    assert row["is_correct"] is True
    assert row["correctness_source"] == "score>0"


def test_append_validation_rollout_record_multiple_rounds():
    """Subsequent appends to the same rollout_id accumulate rows."""
    trainer = _make_trainer()
    rollout_records = {}

    for round_idx in range(3):
        trainer._append_validation_rollout_record(
            rollout_records=rollout_records,
            rollout_id="gs0_b0_i0_uid0",
            global_step=0,
            batch_idx=0,
            sample_idx=0,
            round_idx=round_idx,
            agent_idx=0,
            agent_key="model_0",
            question="Q",
            ground_truth="A",
            output_text=f"out_{round_idx}",
            score=float(round_idx),
            reward_extra_info=None,
            sample_uid="uid0",
            data_source="src",
        )

    record = rollout_records["gs0_b0_i0_uid0"]
    assert len(record["rows"]) == 3
    assert record["rows"][2]["output_text"] == "out_2"
    assert record["rows"][2]["round_idx"] == 2


# ---------------------------------------------------------------------------
# back_propogate_reward — value correctness
# ---------------------------------------------------------------------------

def _make_score_batch(scores_at_last_token: list[float], T: int = 4, resp_pos: int = 3):
    """Helper: batch with token_level_scores nonzero only at resp_pos."""
    B = len(scores_at_last_token)
    scores = torch.zeros(B, T)
    rmask = torch.zeros(B, T, dtype=torch.bool)
    for b, v in enumerate(scores_at_last_token):
        scores[b, resp_pos] = v
        rmask[b, resp_pos] = True
    batch = MagicMock()
    batch.batch = {"token_level_scores": scores, "response_mask": rmask}
    return batch


def test_back_propagate_reward_two_rounds_no_update():
    """With num_rounds=2, range(0,0,-1) is empty — no scores modified."""
    trainer = _make_trainer()

    r0a0 = _make_score_batch([1.0, 2.0])
    r1a0 = _make_score_batch([5.0, 6.0])

    original_r0 = r0a0.batch["token_level_scores"].clone()
    original_r1 = r1a0.batch["token_level_scores"].clone()

    trainer.back_propogate_reward(
        num_rounds=2, num_agents=1,
        round_agent_batches=[[r0a0], [r1a0]],
        gamma=1.0,
    )

    # Neither round should be modified (loop doesn't execute for num_rounds=2)
    assert torch.allclose(r0a0.batch["token_level_scores"], original_r0)
    assert torch.allclose(r1a0.batch["token_level_scores"], original_r1)


def test_back_propagate_reward_three_rounds_two_agents_values():
    """
    num_rounds=3, num_agents=2, gamma=1.0 — verify exact reward math at r=1.

    Initial sum-rewards (only last token is nonzero):
      r=2, a=0: [10, 20]
      r=2, a=1: [30, 40]
      r=1, a=0: [ 1,  2]
      r=1, a=1: [ 3,  4]

    coef = gamma / num_agents = 1.0 / 2 = 0.5
    future_sum at r=2 = [10+30, 20+40] = [40, 60]

    After back-prop (r=1):
      rewards[1][0] = [1 + 0.5*40, 2 + 0.5*60] = [21, 32]
      rewards[1][1] = [3 + 0.5*40, 4 + 0.5*60] = [23, 34]

    r=0 must NOT be modified.
    """
    trainer = _make_trainer()

    r2a0 = _make_score_batch([10.0, 20.0])
    r2a1 = _make_score_batch([30.0, 40.0])
    r1a0 = _make_score_batch([ 1.0,  2.0])
    r1a1 = _make_score_batch([ 3.0,  4.0])
    r0a0 = _make_score_batch([ 7.0,  8.0])
    r0a1 = _make_score_batch([ 9.0, 11.0])

    original_r0a0 = r0a0.batch["token_level_scores"].clone()
    original_r0a1 = r0a1.batch["token_level_scores"].clone()

    round_agent_batches = [
        [r0a0, r0a1],
        [r1a0, r1a1],
        [r2a0, r2a1],
    ]

    trainer.back_propogate_reward(
        num_rounds=3, num_agents=2,
        round_agent_batches=round_agent_batches,
        gamma=1.0,
    )

    # r=1 last-token scores after write-back
    assert torch.allclose(r1a0.batch["token_level_scores"][:, 3], torch.tensor([21.0, 32.0]))
    assert torch.allclose(r1a1.batch["token_level_scores"][:, 3], torch.tensor([23.0, 34.0]))

    # r=2 must be unchanged (no future round)
    assert torch.allclose(r2a0.batch["token_level_scores"][:, 3], torch.tensor([10.0, 20.0]))
    assert torch.allclose(r2a1.batch["token_level_scores"][:, 3], torch.tensor([30.0, 40.0]))

    # r=0 must be unchanged (loop starts at num_rounds-2=1, so r=0 is skipped)
    assert torch.allclose(r0a0.batch["token_level_scores"], original_r0a0)
    assert torch.allclose(r0a1.batch["token_level_scores"], original_r0a1)


def test_back_propagate_reward_gamma_scaling():
    """gamma < 1.0 reduces the propagated credit proportionally."""
    trainer = _make_trainer()

    r2a0 = _make_score_batch([100.0, 100.0])
    r1a0 = _make_score_batch([  0.0,   0.0])
    r0a0 = _make_score_batch([  0.0,   0.0])

    round_agent_batches = [[r0a0], [r1a0], [r2a0]]

    gamma = 0.5
    # coef = 0.5 / 1 = 0.5
    # future_sum = [100, 100]
    # new rewards[1][0] = 0 + 0.5 * 100 = 50 for each sample

    trainer.back_propogate_reward(
        num_rounds=3, num_agents=1,
        round_agent_batches=round_agent_batches,
        gamma=gamma,
    )

    expected = torch.tensor([50.0, 50.0])
    assert torch.allclose(r1a0.batch["token_level_scores"][:, 3], expected)


def test_back_propagate_reward_only_updates_last_response_token():
    """Reward is written only to the last response token, not to other positions."""
    trainer = _make_trainer()
    B, T = 2, 6

    def _batch_with_custom_mask(resp_last_positions: list[int], init_scores: list[float]):
        scores = torch.zeros(B, T)
        rmask = torch.zeros(B, T, dtype=torch.bool)
        for b, (pos, val) in enumerate(zip(resp_last_positions, init_scores)):
            scores[b, pos] = val
            rmask[b, pos] = True
        batch = MagicMock()
        batch.batch = {"token_level_scores": scores, "response_mask": rmask}
        return batch

    # Response tokens at different positions per sample
    r2a0 = _batch_with_custom_mask([2, 4], [10.0, 20.0])
    r1a0 = _batch_with_custom_mask([1, 3], [1.0, 2.0])
    r0a0 = _batch_with_custom_mask([0, 5], [0.0, 0.0])

    trainer.back_propogate_reward(
        num_rounds=3, num_agents=1,
        round_agent_batches=[[r0a0], [r1a0], [r2a0]],
        gamma=1.0,
    )

    s = r1a0.batch["token_level_scores"]
    # Sample 0: last resp token at pos 1, expected value = 1 + 1.0/1 * 10 = 11
    assert s[0, 1].item() == pytest.approx(11.0)
    # Sample 1: last resp token at pos 3, expected value = 2 + 1.0/1 * 20 = 22
    assert s[1, 3].item() == pytest.approx(22.0)
    # All other positions in r=1 must remain zero
    for b in range(B):
        for t in range(T):
            if not r1a0.batch["response_mask"][b, t]:
                assert s[b, t].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _extract_prompts_and_questions — content values
# ---------------------------------------------------------------------------

def test_extract_prompts_question_content():
    """Text between 'user' and 'assistant' markers is extracted as the question."""
    trainer = _make_trainer()

    prompt = "system You are a helpful assistant. user What is 2+2? assistant"
    tokenizer = MagicMock()
    tokenizer.batch_decode.return_value = [prompt]
    trainer.tokenizers = {"model_0": tokenizer}

    batch = DataProto.from_dict(tensors={"input_ids": torch.zeros(1, 4, dtype=torch.long)})
    questions, _ = trainer._extract_prompts_and_questions(batch, "model_0")

    assert len(questions) == 1
    assert "What is 2+2?" in questions[0]
    # Should not contain 'system' or 'assistant' markers
    assert "assistant" not in questions[0].lower()


def test_extract_prompts_no_role_markers_returns_full_prompt():
    """When no user/assistant markers are found, the full prompt is returned."""
    trainer = _make_trainer()

    prompt = "This is a plain prompt with no markers."
    tokenizer = MagicMock()
    tokenizer.batch_decode.return_value = [prompt]
    trainer.tokenizers = {"model_0": tokenizer}

    batch = DataProto.from_dict(tensors={"input_ids": torch.zeros(1, 4, dtype=torch.long)})
    questions, _ = trainer._extract_prompts_and_questions(batch, "model_0")

    assert questions[0] == prompt


def test_extract_prompts_multiple_samples():
    """Each sample gets its own extracted question."""
    trainer = _make_trainer()

    prompts = [
        "system S user Q1 assistant",
        "system S user Q2 assistant",
    ]
    tokenizer = MagicMock()
    tokenizer.batch_decode.return_value = prompts
    trainer.tokenizers = {"model_0": tokenizer}

    batch = DataProto.from_dict(tensors={"input_ids": torch.zeros(2, 4, dtype=torch.long)})
    questions, _ = trainer._extract_prompts_and_questions(batch, "model_0")

    assert len(questions) == 2
    assert "Q1" in questions[0]
    assert "Q2" in questions[1]


def test_extract_prompts_whitespace_is_normalized():
    """Newlines and multiple spaces in questions are collapsed to single spaces."""
    trainer = _make_trainer()

    prompt = "system S user  What\nis\n\n2+2?  assistant"
    tokenizer = MagicMock()
    tokenizer.batch_decode.return_value = [prompt]
    trainer.tokenizers = {"model_0": tokenizer}

    batch = DataProto.from_dict(tensors={"input_ids": torch.zeros(1, 4, dtype=torch.long)})
    questions, _ = trainer._extract_prompts_and_questions(batch, "model_0")

    assert "\n" not in questions[0]
    assert "  " not in questions[0]  # no double spaces


import pytest
