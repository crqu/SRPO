# MAPPO Adversarial Reward Back-Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the symmetric reward back-propagation in `back_propogate_reward` with an asymmetric adversarial signal (agent 0: own discounted future; agent 1: negated agent 0's future), include round 0 in propagation, and skip adversary actor/critic updates at the final round.

**Architecture:** Two targeted edits to `verl/trainer/ppo/mappo_trainer.py` — the `back_propogate_reward` method and the actor/critic update loops in `mappo_fit` — plus corresponding test updates in the existing test file. TDD: tests written first, implementation second.

**Tech Stack:** PyTorch, pytest, verl DataProto

---

### Task 1: Update tests for the new `back_propogate_reward` behavior

**Files:**
- Modify: `tests/trainer/ppo/test_mappo_compute_reward.py`

Five existing tests test the old symmetric behaviour or use `num_agents=1`. Replace/update them all before touching the implementation so they drive the new code.

- [ ] **Step 1: Replace `test_back_propagate_reward_two_rounds_no_update`**

The old test expected round 0 to be *un*touched with `num_rounds=2`. Under the new design, round 0 is always back-propagated. Replace the entire function:

```python
def test_back_propagate_reward_two_rounds_round_zero_updated():
    """With num_rounds=2, round 0 IS updated (loop covers r=0)."""
    trainer = _make_trainer()

    r1a0 = _make_score_batch([10.0, 20.0])   # final round, agent 0
    r1a1 = _make_score_batch([30.0, 40.0])   # final round, agent 1
    r0a0 = _make_score_batch([ 1.0,  2.0])   # round 0, agent 0
    r0a1 = _make_score_batch([ 3.0,  4.0])   # round 0, agent 1

    trainer.back_propogate_reward(
        num_rounds=2, num_agents=2,
        round_agent_batches=[[r0a0, r0a1], [r1a0, r1a1]],
        gamma=1.0,
    )

    # Agent 0 at r=0: base + γ * rewards[1][0] = [1+10, 2+20]
    assert torch.allclose(r0a0.batch["token_level_scores"][:, 3], torch.tensor([11.0, 22.0]))
    # Agent 1 at r=0: -rewards[1][0] = [-10, -20]
    assert torch.allclose(r0a1.batch["token_level_scores"][:, 3], torch.tensor([-10.0, -20.0]))
    # Final round must be unchanged
    assert torch.allclose(r1a0.batch["token_level_scores"][:, 3], torch.tensor([10.0, 20.0]))
    assert torch.allclose(r1a1.batch["token_level_scores"][:, 3], torch.tensor([30.0, 40.0]))
```

- [ ] **Step 2: Replace `test_back_propagate_reward_three_rounds_two_agents_values`**

The old test verified the symmetric `coef = gamma / num_agents` formula. Replace with the asymmetric adversarial expectations:

```python
def test_back_propagate_reward_three_rounds_two_agents_values():
    """
    num_rounds=3, num_agents=2, gamma=1.0 — asymmetric adversarial back-prop.

    Initial last-token scores:
      r=2, a=0: [10, 20]   r=1, a=0: [1, 2]   r=0, a=0: [7,  8]
      r=2, a=1: [30, 40]   r=1, a=1: [3, 4]   r=0, a=1: [9, 11]

    After r=1 pass:
      rewards[1][0] = [1 + 1.0*10, 2 + 1.0*20] = [11, 22]
      rewards[1][1] = [-10, -20]

    After r=0 pass:
      rewards[0][0] = [7 + 1.0*11, 8 + 1.0*22] = [18, 30]
      rewards[0][1] = [-11, -22]

    r=2 must be unchanged.
    """
    trainer = _make_trainer()

    r2a0 = _make_score_batch([10.0, 20.0])
    r2a1 = _make_score_batch([30.0, 40.0])
    r1a0 = _make_score_batch([ 1.0,  2.0])
    r1a1 = _make_score_batch([ 3.0,  4.0])
    r0a0 = _make_score_batch([ 7.0,  8.0])
    r0a1 = _make_score_batch([ 9.0, 11.0])

    trainer.back_propogate_reward(
        num_rounds=3, num_agents=2,
        round_agent_batches=[[r0a0, r0a1], [r1a0, r1a1], [r2a0, r2a1]],
        gamma=1.0,
    )

    assert torch.allclose(r1a0.batch["token_level_scores"][:, 3], torch.tensor([11.0, 22.0]))
    assert torch.allclose(r1a1.batch["token_level_scores"][:, 3], torch.tensor([-10.0, -20.0]))
    assert torch.allclose(r0a0.batch["token_level_scores"][:, 3], torch.tensor([18.0, 30.0]))
    assert torch.allclose(r0a1.batch["token_level_scores"][:, 3], torch.tensor([-11.0, -22.0]))
    # final round unchanged
    assert torch.allclose(r2a0.batch["token_level_scores"][:, 3], torch.tensor([10.0, 20.0]))
    assert torch.allclose(r2a1.batch["token_level_scores"][:, 3], torch.tensor([30.0, 40.0]))
```

- [ ] **Step 3: Replace `test_back_propagate_reward_gamma_scaling`**

Old test used `num_agents=1`. Update to 2 agents and adversarial expectations:

```python
def test_back_propagate_reward_gamma_scaling():
    """gamma < 1.0 scales agent 0's credit; agent 1 negation is unaffected by gamma."""
    trainer = _make_trainer()

    r2a0 = _make_score_batch([100.0, 100.0])
    r2a1 = _make_score_batch([  0.0,   0.0])
    r1a0 = _make_score_batch([  0.0,   0.0])
    r1a1 = _make_score_batch([  0.0,   0.0])
    r0a0 = _make_score_batch([  0.0,   0.0])
    r0a1 = _make_score_batch([  0.0,   0.0])

    # r=1: rewards[1][0] = 0 + 0.5*100 = 50;  rewards[1][1] = -100
    # r=0: rewards[0][0] = 0 + 0.5*50  = 25;  rewards[0][1] = -50
    trainer.back_propogate_reward(
        num_rounds=3, num_agents=2,
        round_agent_batches=[[r0a0, r0a1], [r1a0, r1a1], [r2a0, r2a1]],
        gamma=0.5,
    )

    assert torch.allclose(r1a0.batch["token_level_scores"][:, 3], torch.tensor([ 50.0,  50.0]))
    assert torch.allclose(r1a1.batch["token_level_scores"][:, 3], torch.tensor([-100.0, -100.0]))
    assert torch.allclose(r0a0.batch["token_level_scores"][:, 3], torch.tensor([ 25.0,  25.0]))
    assert torch.allclose(r0a1.batch["token_level_scores"][:, 3], torch.tensor([-50.0, -50.0]))
```

- [ ] **Step 4: Replace `test_back_propagate_reward_only_updates_last_response_token`**

Old test used `num_agents=1`. Update to 2 agents and verify both agents' write-back positions:

```python
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

    r2a0 = _batch_with_custom_mask([2, 4], [10.0, 20.0])
    r2a1 = _batch_with_custom_mask([2, 4], [ 0.0,  0.0])
    r1a0 = _batch_with_custom_mask([1, 3], [ 1.0,  2.0])
    r1a1 = _batch_with_custom_mask([1, 3], [ 0.0,  0.0])
    r0a0 = _batch_with_custom_mask([0, 5], [ 0.0,  0.0])
    r0a1 = _batch_with_custom_mask([0, 5], [ 0.0,  0.0])

    trainer.back_propogate_reward(
        num_rounds=3, num_agents=2,
        round_agent_batches=[[r0a0, r0a1], [r1a0, r1a1], [r2a0, r2a1]],
        gamma=1.0,
    )

    s0 = r1a0.batch["token_level_scores"]
    # Sample 0: last resp at pos 1; agent 0 = 1 + 10 = 11
    assert s0[0, 1].item() == pytest.approx(11.0)
    # Sample 1: last resp at pos 3; agent 0 = 2 + 20 = 22
    assert s0[1, 3].item() == pytest.approx(22.0)
    for b in range(B):
        for t in range(T):
            if not r1a0.batch["response_mask"][b, t]:
                assert s0[b, t].item() == pytest.approx(0.0)

    s1 = r1a1.batch["token_level_scores"]
    # Agent 1 at r=1: -rewards[2][0] = [-10, -20]
    assert s1[0, 1].item() == pytest.approx(-10.0)
    assert s1[1, 3].item() == pytest.approx(-20.0)
    for b in range(B):
        for t in range(T):
            if not r1a1.batch["response_mask"][b, t]:
                assert s1[b, t].item() == pytest.approx(0.0)
```

- [ ] **Step 5: Update `test_back_propagate_reward_warns_on_empty_response` to use 2 agents**

```python
def test_back_propagate_reward_warns_on_empty_response():
    """back_propogate_reward must warn when any sample has no response tokens."""
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

    round_agent_batches = [
        [_make_mock_batch(True),  _make_mock_batch(True)],   # r=0
        [_make_mock_batch(False), _make_mock_batch(True)],   # r=1: sample 1 missing on agent 0
        [_make_mock_batch(True),  _make_mock_batch(True)],   # r=2
    ]

    with pytest.warns(UserWarning, match="no response tokens"):
        trainer.back_propogate_reward(
            num_rounds=3, num_agents=2,
            round_agent_batches=round_agent_batches,
            gamma=1.0,
        )
```

- [ ] **Step 6: Run the five updated tests to confirm they FAIL**

```bash
module load anaconda3/2024.02-1 && conda run -n srpo python -m pytest \
  tests/trainer/ppo/test_mappo_compute_reward.py::test_back_propagate_reward_two_rounds_round_zero_updated \
  tests/trainer/ppo/test_mappo_compute_reward.py::test_back_propagate_reward_three_rounds_two_agents_values \
  tests/trainer/ppo/test_mappo_compute_reward.py::test_back_propagate_reward_gamma_scaling \
  tests/trainer/ppo/test_mappo_compute_reward.py::test_back_propagate_reward_only_updates_last_response_token \
  tests/trainer/ppo/test_mappo_compute_reward.py::test_back_propagate_reward_warns_on_empty_response \
  -v 2>&1 | tail -25
```

Expected: 5 FAILs (AssertionError or similar — implementation still has old logic).

- [ ] **Step 7: Commit the updated tests**

```bash
git add tests/trainer/ppo/test_mappo_compute_reward.py
git commit -m "test(mappo): update back_propogate_reward tests for adversarial asymmetric design"
```

---

### Task 2: Implement the new `back_propogate_reward`

**Files:**
- Modify: `verl/trainer/ppo/mappo_trainer.py:1741-1787`

- [ ] **Step 1: Replace the method body**

In `verl/trainer/ppo/mappo_trainer.py`, replace everything from line 1741 (`def back_propogate_reward`) through line 1787 (the closing line of the write-back block) with:

```python
    def back_propogate_reward(self, num_rounds, num_agents, round_agent_batches, gamma):
        assert num_agents == 2, (
            "back_propogate_reward requires exactly 2 agents "
            "(risk-averse at index 0, adversary at index 1)"
        )
        # Cache scalar rewards: rewards[r][a] shape [B]
        rewards = [
            [
                round_agent_batches[r][a].batch["token_level_scores"].sum(-1)
                for a in range(num_agents)
            ]
            for r in range(num_rounds)
        ]

        for r in range(num_rounds - 2, -1, -1):
            # Agent 0 (risk-averse): base reward + discounted own future reward
            rewards[r][0] = rewards[r][0] + gamma * rewards[r + 1][0]
            # Agent 1 (adversary): negated agent 0's next-round reward
            rewards[r][1] = -rewards[r + 1][0]

            # Write updated reward back to the last response token of each agent
            for a in range(num_agents):
                batch = round_agent_batches[r][a]
                scores = batch.batch["token_level_scores"]    # [B, T]
                rmask  = batch.batch["response_mask"].bool()  # [B, T]

                B, T = scores.shape
                rows = torch.arange(B, device=scores.device)
                idx = torch.arange(T, device=scores.device).unsqueeze(0).expand(B, T)
                last_resp_idx = (idx * rmask.long()).max(dim=1).values  # [B]

                has_resp = rmask.any(dim=1)
                n_missing = int((~has_resp).sum().item())
                if n_missing > 0:
                    import warnings
                    warnings.warn(
                        f"back_propogate_reward: {n_missing} sample(s) have no response tokens "
                        f"in round {r}, agent {a}. Their rewards will not be updated.",
                        UserWarning,
                        stacklevel=2,
                    )
                scores[rows[has_resp], last_resp_idx[has_resp]] = (
                    rewards[r][a][has_resp].to(scores.dtype)
                )
```

- [ ] **Step 2: Run the five back-prop tests**

```bash
module load anaconda3/2024.02-1 && conda run -n srpo python -m pytest \
  tests/trainer/ppo/test_mappo_compute_reward.py -k "back_propagate" -v 2>&1 | tail -25
```

Expected: all 5 PASS.

- [ ] **Step 3: Run the full test file to check for regressions**

```bash
module load anaconda3/2024.02-1 && conda run -n srpo python -m pytest \
  tests/trainer/ppo/test_mappo_compute_reward.py -v 2>&1 | tail -40
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add verl/trainer/ppo/mappo_trainer.py
git commit -m "feat(mappo): implement asymmetric adversarial reward back-propagation

- loop now covers round 0 (range stop changed from 0 to -1)
- agent 0: base + gamma * own future reward
- agent 1: negated agent 0's next-round reward
- assert num_agents == 2"
```

---

### Task 3: Write a failing test for the adversary-skip-at-final-round guard

**Files:**
- Modify: `tests/trainer/ppo/test_mappo_compute_reward.py`

- [ ] **Step 1: Append the source-inspection test**

Add to the end of `tests/trainer/ppo/test_mappo_compute_reward.py`:

```python
def test_adversary_skipped_at_final_round():
    """mappo_fit must contain a guard that skips agent 1 updates at the final round."""
    import inspect
    import re
    import verl.trainer.ppo.mappo_trainer as mod

    src = inspect.getsource(mod.RayMAPPOTrainer.mappo_fit)

    assert re.search(
        r"num_rounds\s*-\s*1\s*and\s*agent_idx\s*==\s*1",
        src,
    ), (
        "mappo_fit must skip agent_idx==1 updates at the final round. "
        "Expected pattern: 'num_rounds - 1 and agent_idx == 1'"
    )
```

- [ ] **Step 2: Run it to confirm FAIL**

```bash
module load anaconda3/2024.02-1 && conda run -n srpo python -m pytest \
  tests/trainer/ppo/test_mappo_compute_reward.py::test_adversary_skipped_at_final_round \
  -v 2>&1 | tail -15
```

Expected: FAIL — `AssertionError: mappo_fit must skip...`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/trainer/ppo/test_mappo_compute_reward.py
git commit -m "test(mappo): add guard test for adversary skip at final round"
```

---

### Task 4: Skip adversary actor/critic updates at the final round

**Files:**
- Modify: `verl/trainer/ppo/mappo_trainer.py:1974-2012`

- [ ] **Step 1: Add the skip guard to the critic update loop**

Locate the critic update block in `mappo_fit` (around line 1976). It currently reads:

```python
                if self.use_critic:
                    for r in range(num_rounds):
                        futures=[]
                        with ThreadPoolExecutor(max_workers=num_agents) as executor:
                            for agent_idx,agent_key in enumerate(agent_keys):
                                futures.append(
                                    executor.submit(
                                        self._update_critic,
```

Add the guard immediately before `futures.append`:

```python
                if self.use_critic:
                    for r in range(num_rounds):
                        futures=[]
                        with ThreadPoolExecutor(max_workers=num_agents) as executor:
                            for agent_idx,agent_key in enumerate(agent_keys):
                                if r == num_rounds - 1 and agent_idx == 1:
                                    continue  # adversary has no future signal at final round
                                futures.append(
                                    executor.submit(
                                        self._update_critic,
                                        r,
                                        agent_idx,
                                        agent_key,
                                        round_agent_batches,
                                        timing_raw,
                                        round_agent_metrics
                                    )
                                )
                        for f in futures:
                            f.result()  # propagate exceptions
```

- [ ] **Step 2: Add the same guard to the actor update loop**

Locate the actor update block (around line 1996). It currently reads:

```python
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    for r in range(num_rounds):
                        futures=[]
                        with ThreadPoolExecutor(max_workers=num_agents) as executor:
                            for agent_idx,agent_key in enumerate(agent_keys):
                                futures.append(
                                    executor.submit(
                                        self._update_actor,
```

Add the guard immediately before `futures.append`:

```python
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    for r in range(num_rounds):
                        futures=[]
                        with ThreadPoolExecutor(max_workers=num_agents) as executor:
                            for agent_idx,agent_key in enumerate(agent_keys):
                                if r == num_rounds - 1 and agent_idx == 1:
                                    continue  # adversary has no future signal at final round
                                futures.append(
                                    executor.submit(
                                        self._update_actor,
                                        r,
                                        agent_idx,
                                        agent_key,
                                        round_agent_batches,
                                        timing_raw,
                                        round_agent_metrics
                                    )
                                )
                        for f in futures:
                            f.result()  # propagate exceptions
```

- [ ] **Step 3: Run the skip-guard test**

```bash
module load anaconda3/2024.02-1 && conda run -n srpo python -m pytest \
  tests/trainer/ppo/test_mappo_compute_reward.py::test_adversary_skipped_at_final_round \
  -v 2>&1 | tail -15
```

Expected: PASS.

- [ ] **Step 4: Run the full test suite**

```bash
module load anaconda3/2024.02-1 && conda run -n srpo python -m pytest \
  tests/trainer/ppo/test_mappo_compute_reward.py -v 2>&1 | tail -40
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add verl/trainer/ppo/mappo_trainer.py
git commit -m "feat(mappo): skip adversary actor/critic updates at final round"
```
