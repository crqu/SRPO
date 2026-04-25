# SRPO Revisions Design — monitoring-led prompt and entropy revisions

Date: 2026-04-25
Status: spec
Goal: produce the smallest credible revision to the IPPO/SRPO training setup that (a) gives a training-time signal for whether the paper's claim is emerging, (b) fixes a visible prompt inconsistency, and (c) operationalizes "bounded rationality" through one entropy ablation point — without changing the SRPO algorithm.

## 1. Context and scope

This repo backs the empirical claim in arxiv 2602.21515: *SRPO yields better partner-generalization than IPPO in multi-agent LLM training.* The codebase already contains:

- `RayMAPPOTrainer` (IPPO) and `RayRiskAverseTrainer` (SRPO) in `verl/trainer/ppo/mappo_trainer.py`. They share rollouts/critic/actor; only `back_propogate_reward` and the optional `_apply_kl_penalty` (gated by `multi_agent.adversary_kl_to_hero`) differ. τ=10 enforced via `algorithm.kl_ctrl.kl_coef=0.1`.
- A 3-round 2-agent debate environment over GSM8K, with per-round per-agent metric scaffolding already wired in validation.
- Hardcoded system prompt and discussion prompt strings at `mappo_trainer.py:478-480` and `:1208-1210`.
- Per-agent actor config override path at `mappo_trainer.py:751-754` (already supports per-agent `entropy_coeff`).

The repo is in **state A** of the empirical loop: no full IPPO-vs-SRPO run has been attempted yet. The bottleneck is observability — there is currently no training-time signal for whether SRPO is moving the partner-generalization needle.

### Knobs locked vs in design space

Locked (paper-faithful):
- ε = 0.01 hero entropy coefficient.
- τ = 10 (kl_coef = 0.1).
- 3 debate rounds, GSM8K, validation every 10 steps.
- Algorithm code (`back_propogate_reward`, `_apply_kl_penalty`).

In design space:
- Adversary entropy coefficient (one ablation point at 0.05).
- System prompt content for both agents.
- Discussion prompt structure.
- Training-time monitoring metrics.
- Cross-pair probe at validation cadence.

Out of scope for this spec:
- Reward shaping beyond terminal correctness.
- Curriculum / KL annealing.
- Self-play vs frozen-hero variants.
- Distinct adversary system prompt (no role differentiation).
- Untuned-Llama probes during training.
- Frozen-self-from-N-steps probes.
- Q0.6B / Q3B / Q4B model arms (this spec covers Q0.5B; later specs extend).

## 2. Visible bug in the current code

The current system prompt at `mappo_trainer.py:478` reads:

> *"You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Please collaborate with the other agent to help the user. Provide a well−reasoned response …"*

Both agents see this prompt verbatim. For the SRPO adversary, "collaborate with the other agent" directly contradicts the reward signal (`reward_adv = -hero_return`), and confounds any gap between IPPO and SRPO with prompt-vs-reward dissonance. This spec fixes the contradiction by adopting a neutral symmetric prompt that does not invoke a "collaborate" or "adversarial" frame.

## 3. Training arms

Three arms; all on Qwen2.5-0.5B-Instruct for the iteration loop. Larger models are deferred.

| Arm | `trainer_type` | Adversary ε | Hero ε | `adversary_kl_to_hero` | Notes |
|---|---|---|---|---|---|
| `ippo` | `mappo` | 0.01 | 0.01 | n/a | Baseline; also serves as cross-pair partner for SRPO arms |
| `srpo_main` | `risk_averse` | **0.05** | 0.01 | true (1/τ = 0.1) | Bounded-rational adversary via entropy asymmetry |
| `srpo_reward_only` | `risk_averse` | 0.01 | 0.01 | true (1/τ = 0.1) | Reward-only ablation; isolates entropy contribution |

All three arms use the same revised system prompt and discussion prompt template (Section 4). The prompt revision is therefore **not** a confound between IPPO and SRPO.

### Bounded-rationality interpretation

The adversary's role in SRPO is to model the worst-case partner the hero could face at deployment. "Bounded rationality" maps cleanly to a noisier policy — i.e., a higher entropy regularizer. The spec encodes this with one point (ε=0.05) rather than a sweep. If the entropy lever turns out to bite, a follow-up spec sweeps {0.05, 0.10}. If it doesn't bite, the `srpo_reward_only` arm is identical to the paper's reward-only construction and serves as the credible baseline.

## 4. Prompt revisions

### System prompt — neutral, identical across both agents and all three arms

> *"You are solving a math word problem with one peer. In each round, you will see the question, your previous response, and your peer's previous response. Reason step by step. Where your peer's reasoning is sound, incorporate it; where it is flawed, identify the flaw and revise. End your response with a single line `Final answer: <number>`."*

Properties:
- No "collaborate" framing (fixes Section 2 bug).
- No "argue against" or "skeptical" framing (avoids injecting adversarial role; SRPO's adversariality must come from reward, not prompt).
- Explicit final-answer format aids extraction.

### Discussion prompt template — slot-structured

> *"Round {r}. Below is the previous round of the discussion."*
> *"— Your previous response: {self_prev}"*
> *"— Peer's previous response: {peer_prev}"*
> *"Now produce your round-{r} response."*

Where `{r}` is the current round index (0-indexed), `{self_prev}` is the agent's own round-(r-1) response, and `{peer_prev}` is the other agent's round-(r-1) response.

**Round 0 special case.** At r=0 there is no prior round, so the prompt skips the template entirely: agent sees only the system prompt and the question, matching the existing first-round path at `mappo_trainer.py:1159-1199` (`_build_input_ids_adversary` style — no history). The templated builder is invoked only for r≥1.

Today the code keeps a single shared `histories` list that concatenates both agents' round-(r-1) responses as `[Last round]: \nAgent 0: ...\nAgent 1: ...`. The new template requires that the trainer track `self_history[agent_idx][batch_idx]` and `peer_history[agent_idx][batch_idx]` separately so each agent's prompt fills its own `{self_prev}` and `{peer_prev}` slot.

### Config plumbing

Both prompt strings move out of inline literals into the multi-agent config:

```yaml
multi_agent:
  system_prompt: |
    You are solving a math word problem with one peer. ...
  discussion_prompt_template: |
    Round {r}. Below is the previous round of the discussion.
    - Your previous response: {self_prev}
    - Peer's previous response: {peer_prev}
    Now produce your round-{r} response.
```

CLAUDE.md already flags the discussion prompt as a tunable research lever; this change makes it real.

## 5. Per-agent entropy

The merge at `mappo_trainer.py:751-754` already supports per-agent actor overrides. Spec-level work is config-only:

For `srpo_main`, the runner injects:

```
multi_agent.agents.0.actor.actor.entropy_coeff=0.05
```

For `srpo_reward_only` and `ippo`, no override is needed; both agents inherit `actor.entropy_coeff=0.01` from `verl/trainer/config/actor/actor.yaml:89`.

No code changes for entropy wiring.

## 6. Monitoring

### 6.1 Cheap training-step diagnostics

All metrics emitted from rollouts the trainer already produces. Logged under `train/<metric>`.

| Metric key | Definition | Source |
|---|---|---|
| `accuracy/round_{r}/agent_{a}` | Mean reward (binary correctness) per round per agent | Existing `token_level_scores` |
| `agreement_rate/round_{r}` | Fraction of samples where both agents extract the same final answer | Equality check on extracted answers |
| `hero_recovery_rate/round_{r}` | Fraction where agent_1 was incorrect at round r-1 and correct at round r (r≥1) | Cached round-(r-1) correctness diff |
| `corrupted_by_debate/round_{r}/agent_{a}` | Fraction where agent_a was correct at round r-1 and incorrect at round r (r≥1) | Same |
| `answer_flip_rate/round_{r}/agent_{a}` | Fraction where extracted answer changed across rounds (r≥1) | Same |
| `kl_adv_to_hero/mean`, `kl_adv_to_hero/p95` | Already computed in `_apply_kl_penalty` (mappo_trainer.py:1812-1817); add p95 emission | Free |
| `entropy/agent_{a}` | Per-step actor entropy, already logged by base trainer | Free |
| `entropy/adv_minus_hero` | Difference of the two | Free |
| `response_len/agent_{a}/round_{r}/{mean,p50,p95}` | Length distribution per agent per round | Token counts already available |
| `repetition_4gram/agent_{a}/round_{r}` | Fraction of 4-grams that repeat within a single response, averaged over batch | One pass per response; cheap |

Implementation point: a single helper `_compute_cheap_diagnostics(round_agent_batches)` is called per training step after rollouts complete and emits the full block.

### 6.2 Cross-pair probe at validation cadence

At trainer init, if `multi_agent.cross_pair_probe.partner_ckpt_dir` is non-empty, instantiate a third actor worker group `probe_partner` from that checkpoint. The partner is loaded once and never updated; reuses the existing `actor_rollout_wgs` machinery so no new worker class is required.

For each `srpo_*` arm, the probe partner is the frozen `ippo` checkpoint at the same `global_step`. For the `ippo` arm, the probe partner is the frozen `srpo_main` checkpoint. If the sibling arm has not yet produced a checkpoint at the matching step, all `cross_pair/*` metrics are emitted as `nan` for that step — no error.

Probe rollout: identical to normal validation, except:
- **Direction A:** trained agent_0 paired with `probe_partner` as agent_1.
- **Direction B:** `probe_partner` as agent_0 paired with trained agent_1.

Both directions run on the same val set used for normal validation.

| Metric key | Definition |
|---|---|
| `cross_pair/joint_acc/{direction}` | Joint correctness after final round (training-time analogue of H1) |
| `cross_pair/trained_agent_acc/{direction}` | Trained agent's final-round accuracy (analogue of H2 at moderate partner shift) |
| `cross_pair/<metric>/{direction}/round_{r}` | All Section 6.1 metrics replayed under cross-pair |

`{direction}` is either `srpo_first` (trained agent at index 0, partner at 1) or `srpo_second` (partner at 0, trained agent at 1). Logging both directions hedges against position bias in the debate format.

### 6.3 What's deliberately deferred

- Frozen-self-from-N-steps probe (used in some SRPO papers as a self-stability check).
- Untuned-Llama partner probe (full partner-shift table is end-of-training only for now).
- Semantic disagreement metric (vs string-equality).
- Token-level entropy histograms.

These extensions are all single-config-flag additions on top of the cross-pair probe machinery once state A is past.

## 7. File-level scope

| File | Change |
|---|---|
| `verl/trainer/config/mappo_trainer.yaml` | Add `multi_agent.system_prompt`, `multi_agent.discussion_prompt_template`, `multi_agent.cross_pair_probe.partner_ckpt_dir` (default empty), `multi_agent.cross_pair_probe.every_n_val_steps` (default 1). |
| `verl/trainer/ppo/mappo_trainer.py` | (1) Replace inline `system_prompt=` and `discussion_prompt=` strings at `:478-480` and `:1208-1210` with config reads. (2) `_build_input_ids_from_histories` and `_build_input_ids_adversary` accept self/peer histories and apply the new template. (3) History bookkeeping at `:522-525`, `:580-581`, `:1589`, `:2014` switches from a single shared `histories` list to per-agent `self_history` and `peer_history`. (4) Add `_compute_cheap_diagnostics(round_agent_batches)` invoked per training step. (5) Add `_run_cross_pair_probe()` invoked at validation cadence; lazy-init of `probe_partner` actor wg from `partner_ckpt_dir` at trainer init. |
| `debug_q05b_local.sh`, `train_q05b.slurm` | Extend `METHOD` switch to `{ippo, srpo_main, srpo_reward_only}`. For `srpo_main` only, append `multi_agent.agents.0.actor.actor.entropy_coeff=0.05`. For all three arms, append `multi_agent.cross_pair_probe.partner_ckpt_dir=<sibling-arm-ckpt-dir>` (resolved per arm). |

Files **not** modified: `verl/trainer/main_mappo.py`, `verl/trainer/ppo/core_algos.py`, `verl/trainer/ppo/reward.py`, `back_propogate_reward`, `_apply_kl_penalty`.

## 8. Tests

Three new unit tests under `tests/trainer/ppo/`:

1. `test_mappo_prompt_template.py` — feed deterministic histories through the templated builder; assert (a) `{r}`, `{self_prev}`, `{peer_prev}` slots fill correctly for r ∈ {0, 1, 2}, (b) hero and adversary prompts are byte-identical when given the same self/peer histories (symmetry contract — guards against accidental role-prompt drift).
2. `test_mappo_per_agent_entropy.py` — load the merged Hydra config for the `srpo_main` arm; assert `actor_cfg_0.actor.entropy_coeff == 0.05` and `actor_cfg_1.actor.entropy_coeff == 0.01`. Repeat for `srpo_reward_only` and `ippo`, asserting both agents see `0.01`. CPU-only, no GPU dependency.
3. `test_mappo_cheap_diagnostics.py` — construct a `round_agent_batches` fixture with known per-sample correctness across rounds and known answer flips; assert each Section 6.1 metric matches the hand-computed expectation.

## 9. Verification ladder

Before committing compute to a full run:

1. All three unit tests pass on CPU.
2. `METHOD=ippo bash debug_q05b_local.sh` (2 steps, `trainer.save_freq=1` for the smoke run so a checkpoint lands at step 2): checkpoint exists; Section 6.1 metric keys appear in console output.
3. `METHOD=srpo_main bash debug_q05b_local.sh` (2 steps) with `partner_ckpt_dir` pointed at the IPPO checkpoint from step 2: `cross_pair/*` keys present in console output; `kl_adv_to_hero/mean > 0`; `entropy/adv_minus_hero > 0`.
4. `METHOD=srpo_reward_only bash debug_q05b_local.sh` (2 steps): `entropy/adv_minus_hero ≈ 0` (within sampling noise).

Only after all four pass do we launch the full Q0.5B 200-step × 2-seed × 3-arm run.

## 10. Success criteria for this spec

This spec is successful if, after the verification ladder and one full Q0.5B run, the operator can answer:

- Is the SRPO algorithm doing something coherent? (KL adv→hero finite, entropy gap matches the configured asymmetry, no length collapse.)
- Is the partner-generalization signal moving? (`cross_pair/joint_acc` for `srpo_main` trends above the same-step `cross_pair/joint_acc` for `ippo`.)
- Is the entropy lever load-bearing? (`srpo_main` cross-pair joint accuracy differs from `srpo_reward_only` by more than seed noise.)

If the answers are unclear after Q0.5B, the next spec extends to entropy sweep (Option 2 from brainstorm), or curriculum (Option 3), or to additional probe partners (Section 6.3 deferred items) — whichever the metrics indicate.
