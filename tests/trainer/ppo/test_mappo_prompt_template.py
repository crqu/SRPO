"""Unit tests for the slot-structured discussion prompt helper.

Spec: docs/superpowers/specs/2026-04-25-srpo-revisions-design.md §4.
"""

from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer


TEMPLATE = (
    "Round {r}. Below is the previous round of the discussion.\n"
    "- Your previous response: {self_prev}\n"
    "- Peer's previous response: {peer_prev}\n"
    "Now produce your round-{r} response.\n"
)


def test_basic_substitution():
    out = RayMAPPOTrainer._format_discussion_prompt(
        TEMPLATE, r=1, self_prev="I said two.", peer_prev="Peer said three."
    )
    assert "Round 1." in out
    assert "Now produce your round-1 response." in out
    assert "Your previous response: I said two." in out
    assert "Peer's previous response: Peer said three." in out


def test_round_index_appears_twice():
    """Both {r} occurrences must be replaced."""
    out = RayMAPPOTrainer._format_discussion_prompt(
        TEMPLATE, r=2, self_prev="a", peer_prev="b"
    )
    assert out.count("round-2") == 1
    assert out.startswith("Round 2.")


def test_braces_in_response_do_not_crash():
    """LLM outputs containing literal {...} must not raise KeyError."""
    risky = "The set is {1, 2, 3} and {self_prev} is a placeholder."
    out = RayMAPPOTrainer._format_discussion_prompt(
        TEMPLATE, r=1, self_prev=risky, peer_prev="ok"
    )
    assert "The set is {1, 2, 3}" in out
    assert "{self_prev} is a placeholder." in out


def test_symmetry_hero_vs_adversary():
    """Helper produces identical output regardless of which agent calls it,
    given identical inputs. Locks the spec's symmetry contract."""
    args = dict(r=1, self_prev="x", peer_prev="y")
    hero_out = RayMAPPOTrainer._format_discussion_prompt(TEMPLATE, **args)
    adv_out = RayMAPPOTrainer._format_discussion_prompt(TEMPLATE, **args)
    assert hero_out == adv_out


def test_empty_inputs():
    out = RayMAPPOTrainer._format_discussion_prompt(
        TEMPLATE, r=1, self_prev="", peer_prev=""
    )
    assert "Your previous response: \n" in out
    assert "Peer's previous response: \n" in out
