# tests/trainer/test_mappo_metrics.py
"""Tests for the §6.1 cheap diagnostics, especially answer_flip_rate.

The bug being fixed: _extract_final_answer used to return the entire raw
substring after the last '####', so 'same number with different trailing
text' counted as a flip. We delegate to extract_solution (strict) to align
with how the reward function defines 'the answer'.
"""

import numpy as np
import pytest

from verl.trainer.ppo.mappo_trainer import (
    _compute_cheap_diagnostics,
    _extract_final_answer,
)


class TestExtractFinalAnswer:
    def test_returns_none_when_marker_missing(self):
        assert _extract_final_answer("just some reasoning, no answer") is None

    def test_keeps_trailing_period(self):
        # extract_solution's regex `[0-9\.\,]+` captures trailing periods as part
        # of the number. We accept this to keep the metric coupled to the reward
        # function (which sees the same string). Documenting the wart explicitly.
        assert _extract_final_answer("reasoning #### 42.") == "42."

    def test_strips_trailing_units_and_text(self):
        # Trailing prose after the number must not count as part of the answer.
        assert _extract_final_answer("reasoning #### 42 dollars") == "42"

    def test_strips_commas(self):
        assert _extract_final_answer("blah #### 1,234") == "1234"

    def test_dollar_sign_blocks_match(self):
        # gsm8k strict regex requires `#### ` followed by a digit; a leading
        # `$` blocks the match entirely. Match the reward function's behavior.
        assert _extract_final_answer("blah #### $42") is None

    def test_takes_last_marker_when_multiple(self):
        assert _extract_final_answer("#### 1\nmore #### 2") == "2"

    def test_handles_negative_numbers(self):
        assert _extract_final_answer("the answer is #### -7") == "-7"

    def test_handles_decimals(self):
        assert _extract_final_answer("step #### 3.14") == "3.14"

    def test_returns_none_when_marker_present_but_no_number(self):
        # 'strict' mode of extract_solution requires a number after ####.
        assert _extract_final_answer("here it is: #### unknown") is None


class TestAnswerFlipRate:
    @staticmethod
    def _make_inputs(round0_texts, round1_texts):
        # Two-agent shape; we only use agent 0 for these tests.
        B = len(round0_texts)
        responses = {
            0: {0: round0_texts, 1: round0_texts},
            1: {0: round1_texts, 1: round1_texts},
        }
        correctness = {
            0: {0: np.zeros(B), 1: np.zeros(B)},
            1: {0: np.zeros(B), 1: np.zeros(B)},
        }
        response_lens = {
            0: {0: np.array([len(t) for t in round0_texts]), 1: np.array([len(t) for t in round0_texts])},
            1: {0: np.array([len(t) for t in round1_texts]), 1: np.array([len(t) for t in round1_texts])},
        }
        return correctness, responses, response_lens

    def test_no_flip_when_only_trailing_text_differs(self):
        # All four round-1 responses extract to "42" despite different trailing prose.
        # Pre-fix this returned 1.0 because raw post-#### text differed.
        round0 = ["reasoning #### 42"] * 4
        round1 = [
            "reasoning #### 42 dollars",
            "reasoning #### 42 cents",
            "x #### 42 ",
            "y #### 42",
        ]
        correctness, responses, response_lens = self._make_inputs(round0, round1)
        m = _compute_cheap_diagnostics(2, 2, correctness, responses, response_lens)
        assert m["answer_flip_rate/round_1/agent_0"] == 0.0

    def test_flips_when_numeric_answer_changes(self):
        round0 = ["#### 42", "#### 42", "#### 42", "#### 42"]
        round1 = ["#### 7", "#### 42", "#### -3", "#### 42.0"]
        correctness, responses, response_lens = self._make_inputs(round0, round1)
        m = _compute_cheap_diagnostics(2, 2, correctness, responses, response_lens)
        # samples 0 and 2 flip; samples 1 and 3 do not (42 == 42; 42 == 42.0 normalized).
        # Note: "42" vs "42.0" — extract_solution strict returns "42" and "42.0" as
        # captured groups; they differ as strings. We accept that as a flip
        # (numeric equality without unit-normalization is out of scope).
        assert m["answer_flip_rate/round_1/agent_0"] == pytest.approx(0.75)

    def test_none_to_number_counts_as_flip(self):
        # Format-compliance change is a real semantic event, keep it as a flip.
        round0 = ["no marker here"] * 2
        round1 = ["#### 1", "still no marker"]
        correctness, responses, response_lens = self._make_inputs(round0, round1)
        m = _compute_cheap_diagnostics(2, 2, correctness, responses, response_lens)
        assert m["answer_flip_rate/round_1/agent_0"] == 0.5

    def test_none_to_none_does_not_flip(self):
        round0 = ["no marker"] * 3
        round1 = ["still no marker"] * 3
        correctness, responses, response_lens = self._make_inputs(round0, round1)
        m = _compute_cheap_diagnostics(2, 2, correctness, responses, response_lens)
        assert m["answer_flip_rate/round_1/agent_0"] == 0.0
