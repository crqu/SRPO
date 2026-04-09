# MAPPO Adversarial Reward Back-Propagation Design

**Date:** 2026-04-09  
**File:** `verl/trainer/ppo/mappo_trainer.py`

---

## Context

The MAPPO trainer runs a multi-round discussion game between two agents:

- **Agent 0 (risk-averse):** Tries to arrive at the correct answer. Incentivized to persuade agent 1.
- **Agent 1 (adversary):** Tries to mislead agent 0 into giving wrong answers. KL-constrained to stay close to agent 0's policy for stability.

Each episode has `num_rounds` rounds (indexed 0 to `num_rounds-1`). At the final round, rewards come from the reward function. At intermediate rounds, rewards are back-propagated.

---

## Reward Signal

For rounds `r ∈ [0, num_rounds-2]` (all rounds except the last):

| Agent | Back-propagated reward |
|-------|------------------------|
| Agent 0 (risk-averse) | `reward[r][0] = base_reward[r][0] + γ * reward[r+1][0]` |
| Agent 1 (adversary)   | `reward[r][1] = −reward[r+1][0]` |

At the **last round** (`r = num_rounds-1`):

| Agent | Action |
|-------|--------|
| Agent 0 | Raw reward from reward function; no modification |
| Agent 1 | **No training update** — actor and critic updates skipped |

### Rationale

- Agent 0's discounted-own-future signal incentivizes it to produce arguments that improve its own correctness over time (persuasion emerges as a strategy).
- Agent 1's negated-agent-0-future signal incentivizes it to produce outputs that reduce agent 0's next-round score (misleading emerges as a strategy).
- Skipping agent 1's update at the last round avoids rewarding or penalizing it for a terminal state where no future signal exists.
- The adversary's intermediate-round base reward (from the reward function) is 0 in practice, so `reward[r][1] = −reward[r+1][0]` is a clean replacement with no information loss.
- KL constraint on agent 1 (existing) prevents degenerate adversarial outputs.

---

## Implementation Changes

### 1. `back_propogate_reward` in `mappo_trainer.py`

**Loop range fix:** Change `range(num_rounds-2, 0, -1)` to `range(num_rounds-2, -1, -1)` so round 0 is included.

**Per-agent branching:** Replace the uniform update with:

```python
# Agent 0: discounted own future reward
rewards[r][0] = base_reward[r][0] + gamma * rewards[r+1][0]

# Agent 1: negated agent 0's next-round reward
rewards[r][1] = -rewards[r+1][0]
```

Write-back to last response token is unchanged (same mechanics as current implementation).

### 2. Training loop in `mappo_fit` (update phase)

When calling `_update_actor` and `_update_critic`, skip agent 1 at the last round:

```python
if r == num_rounds - 1 and agent_idx == 1:
    continue  # adversary has no future signal at final round
```

---

## What Does Not Change

- Write-back mechanism (last response token assignment) — unchanged
- KL penalty application — unchanged
- Agent 0's update at the last round — unchanged (uses raw reward)
- Validation logic — unchanged
- Number of agents assumed to be exactly 2 for adversarial logic

---

## Assumptions

- `agent_keys[0]` is always the risk-averse agent; `agent_keys[1]` is always the adversary.
- `num_rounds >= 2` (otherwise there is nothing to back-propagate).
- Reward function fires only at the final round; intermediate `token_level_scores` are 0 for agent 1 before back-propagation.
