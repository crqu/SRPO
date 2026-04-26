"""Unit tests for the cheap diagnostics helper.

Spec: docs/superpowers/specs/2026-04-25-srpo-revisions-design.md §6.1.
"""

import numpy as np

from verl.trainer.ppo.mappo_trainer import (
    _extract_final_answer,
    _compute_cheap_diagnostics,
)


def test_extract_final_answer_basic():
    assert _extract_final_answer("Reasoning... #### 42") == "42"
    assert _extract_final_answer("no marker here") is None
    assert _extract_final_answer("first #### 1 then #### 2") == "2"  # last one wins


def test_diagnostics_two_round_two_agent_handcrafted():
    """Construct two rounds, two agents, four samples with known correctness:

    sample 0: a0 wrong -> wrong;  a1 wrong -> right         (a1 recovers)
    sample 1: a0 right -> right;  a1 right -> wrong         (a1 corrupted)
    sample 2: a0 wrong -> right;  a1 right -> right         (a0 recovers)
    sample 3: a0 right -> right;  a1 right -> right         (no change)
    """
    correctness = {
        0: {0: np.array([0, 1, 0, 1]), 1: np.array([0, 1, 1, 1])},
        1: {0: np.array([0, 1, 1, 1]), 1: np.array([1, 0, 1, 1])},
    }
    responses = {
        0: {
            0: ["x #### 1", "x #### 2", "x #### 3", "x #### 4"],
            1: ["x #### 1", "x #### 2", "x #### 5", "x #### 4"],
        },
        1: {
            0: ["x #### 1", "x #### 2", "x #### 7", "x #### 4"],
            1: ["x #### 9", "x #### 8", "x #### 7", "x #### 4"],
        },
    }
    response_lens = {
        0: {0: np.array([5, 5, 5, 5]), 1: np.array([5, 5, 5, 5])},
        1: {0: np.array([10, 10, 10, 10]), 1: np.array([10, 10, 10, 10])},
    }

    metrics = _compute_cheap_diagnostics(
        num_rounds=2,
        num_agents=2,
        correctness=correctness,
        responses=responses,
        response_lens=response_lens,
    )

    assert metrics["accuracy/round_0/agent_0"] == 0.5
    assert metrics["accuracy/round_0/agent_1"] == 0.75
    assert metrics["accuracy/round_1/agent_0"] == 0.75
    assert metrics["accuracy/round_1/agent_1"] == 0.75

    assert metrics["agreement_rate/round_0"] == 0.75
    assert metrics["agreement_rate/round_1"] == 0.5

    assert metrics["hero_recovery_rate/round_1"] == 0.25

    assert metrics["corrupted_by_debate/round_1/agent_0"] == 0.0
    assert metrics["corrupted_by_debate/round_1/agent_1"] == 0.25

    assert metrics["answer_flip_rate/round_1/agent_0"] == 0.25
    assert metrics["answer_flip_rate/round_1/agent_1"] == 0.75

    assert metrics["response_len/agent_0/round_0/mean"] == 5.0
    assert metrics["response_len/agent_1/round_1/mean"] == 10.0


def test_diagnostics_skips_undefined_round_zero_metrics():
    """hero_recovery, corrupted_by_debate, answer_flip are only defined for r>=1."""
    correctness = {0: {0: np.array([1, 0]), 1: np.array([0, 1])}}
    responses = {0: {0: ["#### 1", "#### 2"], 1: ["#### 3", "#### 4"]}}
    response_lens = {0: {0: np.array([3, 3]), 1: np.array([3, 3])}}
    metrics = _compute_cheap_diagnostics(
        num_rounds=1,
        num_agents=2,
        correctness=correctness,
        responses=responses,
        response_lens=response_lens,
    )
    assert "hero_recovery_rate/round_0" not in metrics
    assert "corrupted_by_debate/round_0/agent_0" not in metrics
    assert "answer_flip_rate/round_0/agent_0" not in metrics
    assert metrics["accuracy/round_0/agent_0"] == 0.5


def test_diagnostics_4gram_repetition():
    """Repetition rate = fraction of 4-grams that repeat within a single response,
    averaged over batch."""
    correctness = {0: {0: np.array([1, 1]), 1: np.array([1, 1])}}
    responses = {
        0: {
            0: ["a b c d a b c d", "a b c d e f g h"],
            1: ["a b c d a b c d", "a b c d e f g h"],
        }
    }
    response_lens = {0: {0: np.array([8, 8]), 1: np.array([8, 8])}}
    metrics = _compute_cheap_diagnostics(
        num_rounds=1, num_agents=2,
        correctness=correctness, responses=responses, response_lens=response_lens,
    )
    assert abs(metrics["repetition_4gram/agent_0/round_0"] - 0.1) < 1e-9
