# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for NaiveRewardManager.__call__ (synchronous batch interface).

This tests the __call__ method added to support mappo_trainer's synchronous
reward computation: val_reward_fns[agent_key](test_batch, return_dict=True).
"""

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


class MockTokenizer:
    """Minimal tokenizer for CPU tests — encodes/decodes via token-ID strings."""

    eos_token_id = 0
    pad_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i.item() if hasattr(i, "item") else i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [int(x) for x in text.split() if x.lstrip("-").isdigit()]


def simple_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Returns 1.0 if solution matches ground truth, else 0.0."""
    return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0


def dict_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Returns a dict with 'score' and 'acc'."""
    correct = solution_str.strip() == ground_truth.strip()
    return {"score": 1.0 if correct else 0.0, "acc": 1.0 if correct else 0.0}


def make_batch(prompt_ids, response_ids, ground_truth, data_source="test"):
    """Build a single-item DataProto with prompts, responses, and attention_mask."""
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long)
    response_t = torch.tensor([response_ids], dtype=torch.long)
    # attention_mask covers both prompt and response tokens
    total_len = len(prompt_ids) + len(response_ids)
    attn_mask = torch.ones(1, total_len, dtype=torch.long)

    data = DataProto.from_dict(
        {"prompts": prompt_t, "responses": response_t, "attention_mask": attn_mask}
    )
    data.non_tensor_batch = {
        "data_source": np.array([data_source], dtype=object),
        "reward_model": np.array([{"ground_truth": ground_truth}], dtype=object),
    }
    return data


def make_manager(compute_score=simple_score):
    config = DictConfig({})
    NaiveRewardManager._class_initialized = False
    return NaiveRewardManager(config=config, tokenizer=MockTokenizer(), compute_score=compute_score)


class TestNaiveRewardManagerCall:
    """Tests for the synchronous __call__ interface."""

    def test_manager_is_callable(self):
        """NaiveRewardManager instances must be callable after the fix."""
        manager = make_manager()
        assert callable(manager)

    def test_call_returns_dict_when_requested(self):
        """__call__ with return_dict=True returns a dict with 'reward_tensor'."""
        manager = make_manager()
        batch = make_batch(prompt_ids=[10, 11], response_ids=[20, 21, 22], ground_truth="20 21 22")

        result = manager(batch, return_dict=True)

        assert isinstance(result, dict)
        assert "reward_tensor" in result
        assert "reward_extra_info" in result

    def test_call_reward_tensor_has_correct_shape(self):
        """reward_tensor shape matches [batch_size, response_length]."""
        manager = make_manager()
        prompt_ids = [10, 11, 12]
        response_ids = [20, 21, 22, 23]
        batch = make_batch(prompt_ids, response_ids, ground_truth="x")

        result = manager(batch, return_dict=True)
        reward_tensor = result["reward_tensor"]

        assert reward_tensor.shape == (1, len(response_ids))

    def test_call_score_placed_at_last_valid_token(self):
        """Score is placed at position [i, valid_response_length - 1], zeros elsewhere."""
        manager = make_manager(compute_score=simple_score)
        # The decoded response IDs will be "20 21 22"; ground_truth matches → score=1.0
        prompt_ids = [10, 11]
        response_ids = [20, 21, 22]
        tokenizer = MockTokenizer()
        ground_truth = tokenizer.decode(response_ids)
        batch = make_batch(prompt_ids, response_ids, ground_truth=ground_truth)

        result = manager(batch, return_dict=True)
        reward_tensor = result["reward_tensor"]

        # Only the last response token position should be non-zero
        assert reward_tensor[0, -1].item() == pytest.approx(1.0)
        assert reward_tensor[0, :-1].sum().item() == pytest.approx(0.0)

    def test_call_correct_answer_scores_one(self):
        """A response matching the ground truth receives score 1.0."""
        manager = make_manager()
        tokenizer = MockTokenizer()
        response_ids = [42, 43, 44]
        ground_truth = tokenizer.decode(response_ids)
        batch = make_batch(prompt_ids=[1, 2], response_ids=response_ids, ground_truth=ground_truth)

        result = manager(batch, return_dict=True)

        assert result["reward_tensor"][0, -1].item() == pytest.approx(1.0)

    def test_call_incorrect_answer_scores_zero(self):
        """A response not matching the ground truth receives score 0.0."""
        manager = make_manager()
        batch = make_batch(prompt_ids=[1, 2], response_ids=[42, 43, 44], ground_truth="wrong answer")

        result = manager(batch, return_dict=True)

        assert result["reward_tensor"].sum().item() == pytest.approx(0.0)

    def test_call_return_dict_false_returns_tensor(self):
        """Without return_dict, __call__ returns a plain Tensor."""
        manager = make_manager()
        batch = make_batch(prompt_ids=[1, 2], response_ids=[20, 21], ground_truth="x")

        result = manager(batch, return_dict=False)

        assert isinstance(result, torch.Tensor)

    def test_call_batch_of_two_items(self):
        """Batch with 2 items: each item scored independently."""
        config = DictConfig({})
        NaiveRewardManager._class_initialized = False
        manager = NaiveRewardManager(config=config, tokenizer=MockTokenizer(), compute_score=simple_score)
        tokenizer = MockTokenizer()

        prompt_ids = [10, 11]
        correct_response = [20, 21, 22]
        wrong_response = [99, 98, 97]
        correct_gt = tokenizer.decode(correct_response)

        # Build a 2-item batch manually
        prompt_t = torch.tensor([prompt_ids, prompt_ids], dtype=torch.long)
        response_t = torch.tensor([correct_response, wrong_response], dtype=torch.long)
        total_len = len(prompt_ids) + len(correct_response)
        attn_mask = torch.ones(2, total_len, dtype=torch.long)

        data = DataProto.from_dict(
            {"prompts": prompt_t, "responses": response_t, "attention_mask": attn_mask}
        )
        data.non_tensor_batch = {
            "data_source": np.array(["test", "test"], dtype=object),
            "reward_model": np.array(
                [{"ground_truth": correct_gt}, {"ground_truth": correct_gt}], dtype=object
            ),
        }

        result = manager(data, return_dict=True)
        reward_tensor = result["reward_tensor"]

        assert reward_tensor.shape == (2, len(correct_response))
        assert reward_tensor[0, -1].item() == pytest.approx(1.0)  # correct → 1.0
        assert reward_tensor[1, -1].item() == pytest.approx(0.0)  # wrong → 0.0

    def test_call_collects_extra_info_from_dict_score(self):
        """When compute_score returns a dict, extra fields land in reward_extra_info."""
        manager = make_manager(compute_score=dict_score)
        tokenizer = MockTokenizer()
        response_ids = [5, 6, 7]
        ground_truth = tokenizer.decode(response_ids)
        batch = make_batch(prompt_ids=[1], response_ids=response_ids, ground_truth=ground_truth)

        result = manager(batch, return_dict=True)

        extra = result["reward_extra_info"]
        assert "score" in extra
        assert extra["score"] == [pytest.approx(1.0)]
        assert "acc" in extra
        assert extra["acc"] == [pytest.approx(1.0)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
