# SRPO Revisions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the monitoring-led SRPO revisions described in `docs/superpowers/specs/2026-04-25-srpo-revisions-design.md` — fix the adversary prompt-vs-reward contradiction, add per-agent entropy plumbing, add cheap training-step diagnostics, and add a cross-pair probe at validation cadence. No SRPO algorithm changes.

**Architecture:** Surgical edits to `verl/trainer/ppo/mappo_trainer.py` plus config and runner changes. Three new unit tests under `tests/trainer/ppo/`. No new modules or restructuring. The implementation order is: foundation (config + pure helpers under TDD) → integration (refactor the three rollout call sites + lazy partner load) → instrumentation (diagnostics + probe loop) → wiring (runner scripts) → smoke ladder.

**Tech Stack:** Python 3.10, OmegaConf/Hydra, Ray, FSDP, vLLM, pytest.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `verl/trainer/config/mappo_trainer.yaml` | Modify | Add `system_prompt`, `discussion_prompt_template`, `cross_pair_probe.{partner_ckpt_dir,every_n_val_steps}` config keys |
| `verl/trainer/ppo/mappo_trainer.py` | Modify | All trainer-side logic: pure helpers, history refactor, diagnostics helper, cross-pair probe |
| `debug_q05b_local.sh` | Modify | Extend `METHOD` switch to `{ippo,srpo_main,srpo_reward_only}`; inject per-arm flags |
| `train_q05b.slurm` | Modify | Same `METHOD` extension for the slurm runner |
| `tests/trainer/ppo/test_mappo_prompt_template.py` | Create | Unit tests for `_format_discussion_prompt` pure helper |
| `tests/trainer/ppo/test_mappo_per_agent_entropy.py` | Create | Verify per-agent `entropy_coeff` config merge via Hydra compose |
| `tests/trainer/ppo/test_mappo_cheap_diagnostics.py` | Create | Verify cheap-diagnostic metric values against hand-computed expectations |

---

## Conventions for every task

- Repo root: `/weka/scratch/lshi40_llm/mallm/SRPO`. Run all commands from this dir.
- Activate env first in any shell session: `module load anaconda3/2024.02-1 && conda activate srpo`. (Skip if already active.)
- Use `pytest -xvs` for fast-fail verbose output during TDD.
- After each task's final step, the working tree must be clean (committed) before moving on.
- Commit messages follow the existing repo convention (e.g., `feat:`, `fix:`, `test:`, `refactor:`). See `git log --oneline -5` for examples.

---

## Task 1: Add config fields

**Files:**
- Modify: `verl/trainer/config/mappo_trainer.yaml`

- [ ] **Step 1.1: Read the current yaml**

Run: `cat verl/trainer/config/mappo_trainer.yaml`

Expected: file ends at line 45, `agents:` block is the last entry. Confirm `multi_agent.discussion_prompt: "The discussion history is as follows:"` is present on line 26.

- [ ] **Step 1.2: Replace the discussion-prompt line and append new fields**

Replace the single existing field with the templated version and add the new fields. Final state of `multi_agent` block:

```yaml
multi_agent:

  # Number of agents participating in the multi-agent rollout.
  num_agents: 2

  # Number of discussion rounds per training step.
  num_rounds: 3

  # Trainer class: "mappo" for IPPO (RayMAPPOTrainer), "risk_averse" for SRPO (RayRiskAverseTrainer).
  trainer_type: mappo

  # System prompt prepended to every agent turn. Identical for all agents and arms.
  # See docs/superpowers/specs/2026-04-25-srpo-revisions-design.md §4.
  system_prompt: |
    You are solving a math word problem with one peer. In each round, you will see the question, your previous response, and your peer's previous response. Reason step by step. Where your peer's reasoning is sound, incorporate it; where it is flawed, identify the flaw and revise. Output the final answer after "####".

  # Slot-structured discussion-prompt template used for round r >= 1.
  # Placeholders: {r}, {self_prev}, {peer_prev}. Substitution uses str.replace,
  # so literal '{...}' inside agent responses will not crash.
  discussion_prompt_template: |
    Round {r}. Below is the previous round of the discussion.
    - Your previous response: {self_prev}
    - Peer's previous response: {peer_prev}
    Now produce your round-{r} response.

  # SRPO-only: apply (1/tau) * KL(pi_adv || pi_hero) regularizer to the
  # adversary's token-level reward, evaluated on adversary trajectories with
  # the hero policy's logprobs. Only consulted by RayRiskAverseTrainer; tau is
  # the reciprocal of algorithm.kl_ctrl.kl_coef.
  adversary_kl_to_hero: false

  # Cross-pair probe: at validation cadence, swap one trained agent for a frozen
  # partner loaded from this checkpoint dir. Empty disables the probe.
  cross_pair_probe:
    partner_ckpt_dir: ""
    every_n_val_steps: 1

  # Per-agent resource and model overrides.
  # Length must equal num_agents. Each entry can override actor.model.path,
  # n_gpus_per_node, nnodes, and any actor/critic sub-config (including
  # actor.entropy_coeff for the bounded-rationality ablation).
  agents:
    - actor:
        model:
          path: ???
      n_gpus_per_node: 1
    - actor:
        model:
          path: ???
      n_gpus_per_node: 1
```

Use Edit to replace the existing `discussion_prompt:` line and append the new blocks. The old line to remove:

```
  # Prompt prepended to discussion history for each agent turn.
  discussion_prompt: "The discussion history is as follows:"
```

- [ ] **Step 1.3: Verify Hydra parses the config**

Run: `python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('verl/trainer/config/mappo_trainer.yaml'); print(cfg.multi_agent.system_prompt[:30]); print(cfg.multi_agent.cross_pair_probe.partner_ckpt_dir)"`

Expected: prints `You are solving a math word p` and an empty line. No `MissingMandatoryValue` errors (the `???` on agent paths is expected — they're meant to be set on the CLI).

- [ ] **Step 1.4: Commit**

```bash
git add verl/trainer/config/mappo_trainer.yaml
git commit -m "feat(mappo): add system_prompt, discussion_prompt_template, cross_pair_probe config fields"
```

---

## Task 2: Pure helper for discussion prompt formatting (TDD)

**Files:**
- Create: `tests/trainer/ppo/test_mappo_prompt_template.py`
- Modify: `verl/trainer/ppo/mappo_trainer.py` (add `_format_discussion_prompt` static method on `RayMAPPOTrainer`)

Rationale: `str.format` would crash on LLM outputs that contain literal `{...}` (common in code/math). Using `str.replace` for the three slots avoids this and is easier to reason about.

- [ ] **Step 2.1: Read top of test file directory**

Run: `ls tests/trainer/ppo/ 2>/dev/null && ls tests/trainer 2>/dev/null`

Expected: directory exists or needs creation. If `tests/trainer/ppo/` is missing, mkdir it before writing the test.

```bash
mkdir -p tests/trainer/ppo
touch tests/trainer/ppo/__init__.py
```

(Skip the `__init__.py` if other test dirs in this repo don't use one — `find tests -name __init__.py | head` to check.)

- [ ] **Step 2.2: Write the failing test**

Create `tests/trainer/ppo/test_mappo_prompt_template.py`:

```python
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
    # Self_prev content is inserted verbatim; the literal "{self_prev}" inside
    # the response stays as-is (not re-substituted).
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
    # Empty strings must not break the template.
    assert "Your previous response: \n" in out
    assert "Peer's previous response: \n" in out
```

- [ ] **Step 2.3: Run test, expect FAIL**

Run: `pytest tests/trainer/ppo/test_mappo_prompt_template.py -xvs`

Expected: `AttributeError: type object 'RayMAPPOTrainer' has no attribute '_format_discussion_prompt'` on every test.

- [ ] **Step 2.4: Add the helper to RayMAPPOTrainer**

In `verl/trainer/ppo/mappo_trainer.py`, find the existing `_build_input_ids_from_histories` method (around line 1116 — `grep -n "def _build_input_ids_from_histories" verl/trainer/ppo/mappo_trainer.py` to confirm). Insert the static method **immediately above** it on the same indentation level:

```python
    @staticmethod
    def _format_discussion_prompt(template: str, r: int, self_prev: str, peer_prev: str) -> str:
        """Substitute {r}, {self_prev}, {peer_prev} into the template using str.replace.

        Uses .replace rather than str.format so literal '{...}' inside LLM responses
        does not raise KeyError. See spec §4 for the slot semantics.
        """
        return (
            template
            .replace("{r}", str(r))
            .replace("{self_prev}", self_prev)
            .replace("{peer_prev}", peer_prev)
        )
```

- [ ] **Step 2.5: Run test, expect PASS**

Run: `pytest tests/trainer/ppo/test_mappo_prompt_template.py -xvs`

Expected: 5 passed.

- [ ] **Step 2.6: Commit**

```bash
git add tests/trainer/ppo/test_mappo_prompt_template.py verl/trainer/ppo/mappo_trainer.py
git commit -m "feat(mappo): add _format_discussion_prompt helper with brace-safe substitution"
```

---

## Task 3: Refactor history bookkeeping to per-agent self/peer

This task changes three rollout call sites to track `self_history[agent_idx]` instead of a single shared `histories`, and switches the prompt builder to use the new template via the helper from Task 2.

**Files:**
- Modify: `verl/trainer/ppo/mappo_trainer.py`

The three call sites are validation (around line 460), `RayMAPPOTrainer.mappo_fit` (around line 1553), and `RayRiskAverseTrainer.mappo_fit` (around line 1981). All three currently maintain a single `histories` list.

- [ ] **Step 3.1: Update `_build_input_ids_from_histories` signature**

Find the existing method on `RayMAPPOTrainer` (around line 1116 — `grep -n "def _build_input_ids_from_histories" verl/trainer/ppo/mappo_trainer.py`). Replace its body with the templated version.

Old (line ~1116-1157):

```python
    def _build_input_ids_from_histories(self,system_prompt,discussion_prompt,questions,histories,batch:DataProto,agent_key,max_history_tokens):
        prompts = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "user","content": discussion_prompt+hist}
            ]
            for q,hist in zip(questions,histories)
        ]
        # ... (tokenization, postprocess) ...
```

New: keep the method name but accept `self_history` and `peer_history`, plus `r` and `discussion_prompt_template`. Drop `discussion_prompt` (no longer used). The system prompt becomes optional override; default to `self.config.multi_agent.system_prompt`.

```python
    def _build_input_ids_from_histories(
        self,
        questions,
        self_history,
        peer_history,
        r: int,
        batch: DataProto,
        agent_key,
        max_history_tokens,
        system_prompt: str | None = None,
        discussion_prompt_template: str | None = None,
    ):
        """Build per-sample chat prompts using the slot-structured template.

        For r >= 1, fills {r}, {self_prev}, {peer_prev} via _format_discussion_prompt.
        For r == 0, callers should not invoke this method (use the no-history path).
        """
        ma = OmegaConf.select(self.config, "multi_agent", default={}) or {}
        if system_prompt is None:
            system_prompt = ma.get("system_prompt", "")
        if discussion_prompt_template is None:
            discussion_prompt_template = ma.get("discussion_prompt_template", "")

        prompts = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {
                    "role": "user",
                    "content": self._format_discussion_prompt(
                        discussion_prompt_template, r, self_prev, peer_prev
                    ),
                },
            ]
            for q, self_prev, peer_prev in zip(questions, self_history, peer_history)
        ]
        tokenizer = self.tokenizers[agent_key]
        raw_prompts = [
            tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in prompts
        ]
        input_ids_list = []
        attention_mask_list = []
        for rp in raw_prompts:
            model_inputs = tokenizer(rp, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            truncation = self.config.data.get("truncation", "error")
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_history_tokens,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation=truncation,
            )
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)

        batch.batch["input_ids"] = input_ids
        batch.batch["attention_mask"] = attention_mask
        batch.non_tensor_batch.pop("raw_prompt_ids", None)
        return raw_prompts
```

- [ ] **Step 3.2: Update the SRPO override to match the new signature**

Find `RayRiskAverseTrainer._build_input_ids_from_histories` at line ~1834. Currently:

```python
    def _build_input_ids_from_histories(
        self,
        system_prompt,
        discussion_prompt,
        questions,
        histories,
        batch: DataProto,
        agent_key,
        max_history_tokens,
    ):
        shared_prompt = self._shared_discussion_prompt()
        return super()._build_input_ids_from_histories(
            system_prompt, shared_prompt, questions, histories, batch, agent_key, max_history_tokens
        )
```

Replace with: delete the override entirely. The new templated path lives only on the parent and reads from config; there is no remaining reason for SRPO to override it (the `_shared_discussion_prompt` indirection is now redundant). Also delete the now-unused `_shared_discussion_prompt` method (around line ~1762 — `grep -n "_shared_discussion_prompt" verl/trainer/ppo/mappo_trainer.py` to confirm both call sites are gone after this step).

- [ ] **Step 3.3: Update `_single_agent_rollout` to use new signature**

Find `_single_agent_rollout` at line ~1201. Currently it uses inline literals for system_prompt and discussion_prompt and takes a single `histories` parameter. Change the signature and body:

Old call signature: `def _single_agent_rollout(self, agent_idx, agent_key, batch_dict, histories, r, round_agent_metrics)`

New: `def _single_agent_rollout(self, agent_idx, agent_key, batch_dict, self_history, peer_history, r, round_agent_metrics)`

Old body fragment (line ~1206-1210):

```python
        if r>0:
            questions, _ =self._extract_prompts_and_questions(batch,agent_key)
            system_prompt="You are Qwen, created by Alibaba Cloud. ..."
            discussion_prompt=f"The discussion history is as follows:"
            chat_prompts=self._build_input_ids_from_histories(system_prompt,discussion_prompt,questions,histories,batch,agent_key,max_history_tokens=4096)
```

Replace with:

```python
        if r > 0:
            questions, _ = self._extract_prompts_and_questions(batch, agent_key)
            chat_prompts = self._build_input_ids_from_histories(
                questions=questions,
                self_history=self_history,
                peer_history=peer_history,
                r=r,
                batch=batch,
                agent_key=agent_key,
                max_history_tokens=4096,
            )
```

- [ ] **Step 3.4: Update validation loop call site (line ~460)**

Find the validation rollout loop around line 458 (`for batch_idx, batch_tuple in enumerate(...)`). The loop currently does:

```python
        histories = [""] * batch_size
        for r in range(num_rounds):
            this_round = [""] * batch_size
            for agent_idx, agent_key in enumerate(agent_keys):
                # ...
                if r > 0:
                    system_prompt = "You are Qwen, ..."
                    discussion_prompt = f"The discussion history is as follows: "
                    chat_prompts = self._build_input_ids_from_histories(
                        system_prompt, discussion_prompt, questions, histories, ...)
                # ... generate ...
                this_round = [old + f"\nAgent {agent_idx}: {new}" for old, new in zip(this_round, output_texts)]
            # ...
        histories[:] = [f"[Last round]: {new}" for new in this_round]
```

Replace with per-agent tracking:

```python
        # Per-agent prior responses. self_history[a][b] is agent a's response on sample b
        # from the previous round. For r=0 both agents see empty histories (no template applied).
        self_history = [[""] * batch_size for _ in range(num_agents)]
        for r in range(num_rounds):
            this_round_responses = [[""] * batch_size for _ in range(num_agents)]
            for agent_idx, agent_key in enumerate(agent_keys):
                # ... existing setup (sample_inputs, sample_uids, batch_dict, etc.) ...
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                questions, _ = self._extract_prompts_and_questions(test_batch, agent_key)
                if r == 0:
                    chat_prompts = questions
                else:
                    peer_idx = 1 - agent_idx  # 2-agent assumption
                    chat_prompts = self._build_input_ids_from_histories(
                        questions=questions,
                        self_history=self_history[agent_idx],
                        peer_history=self_history[peer_idx],
                        r=r,
                        batch=test_batch,
                        agent_key=agent_key,
                        max_history_tokens=4096,
                    )
                # ... rest of the per-agent body unchanged through generation/scoring ...
                # At the end of the per-agent block, capture this agent's responses:
                this_round_responses[agent_idx] = list(output_texts)
            # After all agents: rotate histories for next round.
            for a in range(num_agents):
                self_history[a] = this_round_responses[a]
```

Notes:
- The existing `this_round` accumulator (`old + f"\nAgent {agent_idx}: {new}"`) and the trailing `histories[:] = [f"[Last round]: {new}" for new in this_round]` both go away. Replace them with `this_round_responses[agent_idx] = list(output_texts)` and the rotation loop above.
- The existing call to `print(histories[0])` at line ~585 (debug print) can be replaced with `print(self_history[0][0] if self_history[0] else "")` or removed.

- [ ] **Step 3.5: Update IPPO training loop call site (line ~1553)**

Apply the same per-agent history pattern. The current code:

```python
                histories = [""] * batch_size
                for r in range(num_rounds):
                    if r > 0 and self.async_rollout_mode:
                        for agent_key in agent_keys:
                            self.checkpoint_managers[agent_key].wake_up_replicas()
                    this_round = [""] * batch_size
                    futures = []
                    with ThreadPoolExecutor(max_workers=num_agents) as executor:
                        for agent_idx, agent_key in enumerate(agent_keys):
                            batch_dict = batch_tuple[agent_idx]
                            futures.append(executor.submit(
                                self._single_agent_rollout,
                                agent_idx, agent_key, batch_dict, histories, r, round_agent_metrics,
                            ))
                    results = [f.result() for f in futures]
                    for agent_idx, (resp_texts, batch, agent_timing) in enumerate(results):
                        this_round = [old + f"\nAgent {agent_idx}: {new}" for old, new in zip(this_round, resp_texts)]
                        round_agent_batches[r][agent_idx] = batch
                        # ... timings ...
                    histories[:] = [f"[Last round]: {new}" for new in this_round]
```

New:

```python
                self_history = [[""] * batch_size for _ in range(num_agents)]
                for r in range(num_rounds):
                    if r > 0 and self.async_rollout_mode:
                        for agent_key in agent_keys:
                            self.checkpoint_managers[agent_key].wake_up_replicas()
                    futures = []
                    with ThreadPoolExecutor(max_workers=num_agents) as executor:
                        for agent_idx, agent_key in enumerate(agent_keys):
                            batch_dict = batch_tuple[agent_idx]
                            peer_idx = 1 - agent_idx
                            futures.append(executor.submit(
                                self._single_agent_rollout,
                                agent_idx,
                                agent_key,
                                batch_dict,
                                self_history[agent_idx],
                                self_history[peer_idx],
                                r,
                                round_agent_metrics,
                            ))
                    results = [f.result() for f in futures]
                    this_round_responses = [[""] * batch_size for _ in range(num_agents)]
                    for agent_idx, (resp_texts, batch, agent_timing) in enumerate(results):
                        this_round_responses[agent_idx] = list(resp_texts)
                        round_agent_batches[r][agent_idx] = batch
                        round_agent_timings[r][agent_idx] = agent_timing
                        if "step" in agent_timing:
                            step_durations.append(agent_timing["step"])
                    for a in range(num_agents):
                        self_history[a] = this_round_responses[a]
```

- [ ] **Step 3.6: Update SRPO training loop call site (line ~1981)**

Identical pattern — apply the same replacement to the `RayRiskAverseTrainer.mappo_fit` rollout block.

- [ ] **Step 3.7: Verify no other callers reference the old API**

Run: `grep -n "discussion_prompt\b\|histories\[\|histories=\|_shared_discussion_prompt\|histories\[:\]" verl/trainer/ppo/mappo_trainer.py`

Expected: no matches except inside the new code (any matches inside the new helper bodies are fine; old `[Last round]:` / `Agent {agent_idx}` accumulators must be gone).

- [ ] **Step 3.8: Smoke import**

Run: `python -c "from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer, RayRiskAverseTrainer; print('ok')"`

Expected: `ok`. Catches signature typos and dangling references.

- [ ] **Step 3.9: Re-run the prompt-template unit test**

Run: `pytest tests/trainer/ppo/test_mappo_prompt_template.py -xvs`

Expected: still 5 passed (this task didn't change the helper).

- [ ] **Step 3.10: Commit**

```bash
git add verl/trainer/ppo/mappo_trainer.py
git commit -m "refactor(mappo): per-agent self/peer histories + templated discussion prompt"
```

---

## Task 4: Per-agent entropy override verification (TDD)

The merge at `mappo_trainer.py:751-754` already supports per-agent overrides. This task adds a unit test that locks the contract.

**Files:**
- Create: `tests/trainer/ppo/test_mappo_per_agent_entropy.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/trainer/ppo/test_mappo_per_agent_entropy.py`:

```python
"""Verify per-agent actor.entropy_coeff override merges correctly via Hydra.

Locks the contract that `multi_agent.agents.{i}.actor.actor.entropy_coeff`
overrides `actor_rollout_ref.actor.entropy_coeff` for agent i only.

Spec: docs/superpowers/specs/2026-04-25-srpo-revisions-design.md §5.
"""

from copy import deepcopy

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
import os


CONFIG_DIR = os.path.abspath("verl/trainer/config")


def _compose(overrides):
    """Compose mappo_trainer.yaml with the given overrides; return OmegaConf cfg."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="mappo_trainer", overrides=overrides)


def _merge_per_agent(cfg, i):
    """Mirror the merge done in RayMAPPOTrainer.init_workers (mappo_trainer.py:751-754)."""
    per_agent_override = OmegaConf.select(cfg, f"multi_agent.agents.{i}") or {}
    per_agent_actor_override = per_agent_override.get("actor", {})
    return OmegaConf.merge(deepcopy(cfg.actor_rollout_ref), per_agent_actor_override)


def test_no_override_uses_top_level_default():
    """Without any per-agent override, both agents inherit actor.entropy_coeff=0.01."""
    cfg = _compose([
        "multi_agent.agents.0.actor.model.path=stub-0",
        "multi_agent.agents.1.actor.model.path=stub-1",
    ])
    a0 = _merge_per_agent(cfg, 0)
    a1 = _merge_per_agent(cfg, 1)
    assert float(a0.actor.entropy_coeff) == 0.01
    assert float(a1.actor.entropy_coeff) == 0.01


def test_srpo_main_override_only_agent_0():
    """srpo_main runner injects 0.05 for agent 0; agent 1 stays at 0.01."""
    cfg = _compose([
        "multi_agent.agents.0.actor.model.path=stub-0",
        "multi_agent.agents.1.actor.model.path=stub-1",
        "multi_agent.agents.0.actor.actor.entropy_coeff=0.05",
    ])
    a0 = _merge_per_agent(cfg, 0)
    a1 = _merge_per_agent(cfg, 1)
    assert float(a0.actor.entropy_coeff) == 0.05
    assert float(a1.actor.entropy_coeff) == 0.01


def test_override_does_not_leak_across_agents():
    """Setting agent-1 override must not affect agent-0's config."""
    cfg = _compose([
        "multi_agent.agents.0.actor.model.path=stub-0",
        "multi_agent.agents.1.actor.model.path=stub-1",
        "multi_agent.agents.1.actor.actor.entropy_coeff=0.07",
    ])
    a0 = _merge_per_agent(cfg, 0)
    a1 = _merge_per_agent(cfg, 1)
    assert float(a0.actor.entropy_coeff) == 0.01
    assert float(a1.actor.entropy_coeff) == 0.07
```

- [ ] **Step 4.2: Run test, expect PASS**

Run: `pytest tests/trainer/ppo/test_mappo_per_agent_entropy.py -xvs`

Expected: 3 passed. (No code change needed — the wiring already exists; this test only locks the contract. If a test fails, the actor.yaml default has changed away from 0.01 and the spec needs an update — stop and flag.)

- [ ] **Step 4.3: Commit**

```bash
git add tests/trainer/ppo/test_mappo_per_agent_entropy.py
git commit -m "test(mappo): lock per-agent actor.entropy_coeff override contract"
```

---

## Task 5: Cheap diagnostics helper (TDD)

**Files:**
- Create: `tests/trainer/ppo/test_mappo_cheap_diagnostics.py`
- Modify: `verl/trainer/ppo/mappo_trainer.py` (add `_compute_cheap_diagnostics` method + invocation site)

The helper takes `round_agent_batches[r][a]` (already populated by the rollout loop) plus the per-round per-agent decoded response texts (which we'll thread in alongside) and returns a flat metric dict suitable for `Tracking.log`.

For deterministic testing, the helper accepts pure Python inputs (numpy arrays + Python lists of strings), not the full DataProto stack. The trainer-side caller marshals the relevant fields out of DataProto before invoking the helper.

- [ ] **Step 5.1: Decide answer extraction policy**

Per the revised system prompt (Section 4 of the spec), the model emits the final answer after `####`. Define a small extractor used by both the helper and tests:

```python
def _extract_final_answer(text: str) -> str | None:
    """Return the substring after the last '####', stripped. None if absent."""
    if "####" not in text:
        return None
    return text.rsplit("####", 1)[1].strip()
```

This goes inside `mappo_trainer.py` as a module-level function (not a method).

- [ ] **Step 5.2: Write the failing test**

Create `tests/trainer/ppo/test_mappo_cheap_diagnostics.py`:

```python
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
        # round_idx -> agent_idx -> [B] correctness in {0,1}
        0: {0: np.array([0, 1, 0, 1]), 1: np.array([0, 1, 1, 1])},
        1: {0: np.array([0, 1, 1, 1]), 1: np.array([1, 0, 1, 1])},
    }
    # Decoded responses with explicit final-answer markers.
    # Round 0: a0 says "1"/"2"/"3"/"4"; a1 says "1"/"2"/"5"/"4".
    # Round 1: a0 says "1"/"2"/"7"/"4"; a1 says "9"/"8"/"7"/"4".
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
    # Response token lengths (already known by the rollout loop).
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

    # Round-wise per-agent accuracy.
    assert metrics["accuracy/round_0/agent_0"] == 0.5  # [0,1,0,1]
    assert metrics["accuracy/round_0/agent_1"] == 0.75  # [0,1,1,1]
    assert metrics["accuracy/round_1/agent_0"] == 0.75  # [0,1,1,1]
    assert metrics["accuracy/round_1/agent_1"] == 0.75  # [1,0,1,1]

    # Agreement rate (extracted answers equal across agents) by round.
    # Round 0: (1,1) (2,2) (3,5) (4,4) -> 3/4
    assert metrics["agreement_rate/round_0"] == 0.75
    # Round 1: (1,9) (2,8) (7,7) (4,4) -> 2/4
    assert metrics["agreement_rate/round_1"] == 0.5

    # Hero recovery (agent_1 wrong at r-1, correct at r) — only defined for r>=1.
    # Round 1: a1[r=0]=[0,1,1,1], a1[r=1]=[1,0,1,1] -> recovered: sample 0 only -> 1/4
    assert metrics["hero_recovery_rate/round_1"] == 0.25

    # Corrupted-by-debate (agent correct at r-1, wrong at r) — both agents.
    # a0: r=0 [0,1,0,1], r=1 [0,1,1,1] -> corrupted: none -> 0
    assert metrics["corrupted_by_debate/round_1/agent_0"] == 0.0
    # a1: r=0 [0,1,1,1], r=1 [1,0,1,1] -> corrupted: sample 1 -> 1/4
    assert metrics["corrupted_by_debate/round_1/agent_1"] == 0.25

    # Answer flip rate (extracted answer changed between rounds), per agent.
    # a0: r=0 [1,2,3,4], r=1 [1,2,7,4] -> flipped: sample 2 -> 1/4
    assert metrics["answer_flip_rate/round_1/agent_0"] == 0.25
    # a1: r=0 [1,2,5,4], r=1 [9,8,7,4] -> flipped: samples 0,1,2 -> 3/4
    assert metrics["answer_flip_rate/round_1/agent_1"] == 0.75

    # Response length stats (mean) — sanity check.
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
    # No round_1 keys at all — only round_0 defined ones.
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
            # Sample 0: 'a b c d a b c d' has 5 4-grams, 4 unique -> 1/5 repeat fraction.
            # Sample 1: 'a b c d e f g h' has 5 4-grams, 5 unique -> 0.
            0: ["a b c d a b c d", "a b c d e f g h"],
            1: ["a b c d a b c d", "a b c d e f g h"],
        }
    }
    response_lens = {0: {0: np.array([8, 8]), 1: np.array([8, 8])}}
    metrics = _compute_cheap_diagnostics(
        num_rounds=1, num_agents=2,
        correctness=correctness, responses=responses, response_lens=response_lens,
    )
    # Mean of [0.2, 0.0] = 0.1
    assert abs(metrics["repetition_4gram/agent_0/round_0"] - 0.1) < 1e-9
```

- [ ] **Step 5.3: Run test, expect FAIL**

Run: `pytest tests/trainer/ppo/test_mappo_cheap_diagnostics.py -xvs`

Expected: `ImportError: cannot import name '_extract_final_answer'` (or `_compute_cheap_diagnostics`).

- [ ] **Step 5.4: Implement the helpers in mappo_trainer.py**

Add these as module-level functions near the top of `verl/trainer/ppo/mappo_trainer.py` (under the existing imports, before any class definitions):

```python
def _extract_final_answer(text: str) -> str | None:
    """Return the substring after the last '####', stripped. None if marker absent."""
    if "####" not in text:
        return None
    return text.rsplit("####", 1)[1].strip()


def _compute_cheap_diagnostics(
    num_rounds: int,
    num_agents: int,
    correctness: dict,
    responses: dict,
    response_lens: dict,
) -> dict:
    """Compute the §6.1 metric block from already-rolled-out per-round per-agent data.

    Args:
        num_rounds: number of debate rounds
        num_agents: number of agents (2 for this spec)
        correctness: dict[r][a] -> np.ndarray[B] of {0,1}
        responses: dict[r][a] -> list[str] of length B (decoded text)
        response_lens: dict[r][a] -> np.ndarray[B] of int (response token counts)

    Returns:
        Flat dict of metric_key -> float value, using the keys defined in spec §6.1.
        Round-0 conditional metrics (hero_recovery, corrupted_by_debate, answer_flip)
        are emitted only for r >= 1.
    """
    import numpy as np

    metrics: dict = {}

    # Pre-extract answers once per (r, a, sample) for reuse below.
    answers = {
        r: {a: [_extract_final_answer(t) for t in responses[r][a]] for a in range(num_agents)}
        for r in range(num_rounds)
    }

    for r in range(num_rounds):
        # Per-agent accuracy.
        for a in range(num_agents):
            metrics[f"accuracy/round_{r}/agent_{a}"] = float(correctness[r][a].mean())

        # Agreement rate (both agents extract the same final answer; None != None counts as disagree).
        agree = [
            (ans0 is not None and ans1 is not None and ans0 == ans1)
            for ans0, ans1 in zip(answers[r][0], answers[r][1])
        ]
        metrics[f"agreement_rate/round_{r}"] = float(np.mean(agree)) if agree else 0.0

        # Per-agent response length stats.
        for a in range(num_agents):
            lens = response_lens[r][a]
            metrics[f"response_len/agent_{a}/round_{r}/mean"] = float(lens.mean())
            metrics[f"response_len/agent_{a}/round_{r}/p50"] = float(np.percentile(lens, 50))
            metrics[f"response_len/agent_{a}/round_{r}/p95"] = float(np.percentile(lens, 95))

        # 4-gram repetition rate per agent per round (within-response, word-level).
        for a in range(num_agents):
            rates = []
            for text in responses[r][a]:
                tokens = text.split()
                if len(tokens) < 4:
                    rates.append(0.0)
                    continue
                grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
                if not grams:
                    rates.append(0.0)
                    continue
                unique = len(set(grams))
                rates.append(1.0 - unique / len(grams))
            metrics[f"repetition_4gram/agent_{a}/round_{r}"] = float(np.mean(rates)) if rates else 0.0

        if r == 0:
            continue

        # Conditional cross-round metrics (r >= 1).
        # Hero recovery: agent_1 wrong at r-1, correct at r.
        prev_a1 = correctness[r - 1][1]
        curr_a1 = correctness[r][1]
        recovered = ((prev_a1 == 0) & (curr_a1 == 1)).astype(float)
        metrics[f"hero_recovery_rate/round_{r}"] = float(recovered.mean())

        for a in range(num_agents):
            prev = correctness[r - 1][a]
            curr = correctness[r][a]
            corrupted = ((prev == 1) & (curr == 0)).astype(float)
            metrics[f"corrupted_by_debate/round_{r}/agent_{a}"] = float(corrupted.mean())

            prev_ans = answers[r - 1][a]
            curr_ans = answers[r][a]
            flips = [pa != ca for pa, ca in zip(prev_ans, curr_ans)]
            metrics[f"answer_flip_rate/round_{r}/agent_{a}"] = float(np.mean(flips)) if flips else 0.0

    return metrics
```

- [ ] **Step 5.5: Run test, expect PASS**

Run: `pytest tests/trainer/ppo/test_mappo_cheap_diagnostics.py -xvs`

Expected: 3 passed.

- [ ] **Step 5.6: Wire the helper into the IPPO training loop**

Find the IPPO training loop block in `RayMAPPOTrainer.mappo_fit` (around line 1591, just after the rollout `for r in range(num_rounds)` block ends and before `back_propogate_reward`). Insert a call that builds the per-agent inputs from `round_agent_batches` and the responses already captured in Task 3.

This step requires a new local list `round_agent_responses` populated alongside `round_agent_batches` (currently only `this_round_responses` from Task 3 holds the texts, but that's per-round and overwritten). Update the rollout loop in mappo_fit to also stash responses into a `round_agent_responses[r][agent_idx]` list parallel to `round_agent_batches`.

After Task 3's rollout loop:

```python
                # ... existing rollout loop ends here ...

                # Cheap diagnostics — emit BEFORE back_propogate_reward (which mutates token_level_scores).
                diag_correctness = {
                    r: {
                        a: round_agent_batches[r][a].batch["token_level_scores"].sum(-1).cpu().numpy()
                        for a in range(num_agents)
                    }
                    for r in range(num_rounds)
                }
                # Binarize: GSM8K rewards are already 0/1; clip in case of fractional.
                for r in range(num_rounds):
                    for a in range(num_agents):
                        diag_correctness[r][a] = (diag_correctness[r][a] > 0.5).astype("int64")
                diag_response_lens = {
                    r: {
                        a: round_agent_batches[r][a].batch["response_mask"].sum(-1).cpu().numpy()
                        for a in range(num_agents)
                    }
                    for r in range(num_rounds)
                }
                diag_metrics = _compute_cheap_diagnostics(
                    num_rounds=num_rounds,
                    num_agents=num_agents,
                    correctness=diag_correctness,
                    responses=round_agent_responses,
                    response_lens=diag_response_lens,
                )
                metrics.update({f"train/{k}": v for k, v in diag_metrics.items()})
```

To populate `round_agent_responses`, modify the post-rollout collection block from Task 3 step 3.5:

```python
                round_agent_responses = [["" for _ in range(num_agents)] for _ in range(num_rounds)]
                # ... inside the for r in range(num_rounds) loop, after results = [f.result() for f in futures]:
                    for agent_idx, (resp_texts, batch, agent_timing) in enumerate(results):
                        # ... existing assignments ...
                        round_agent_responses[r][agent_idx] = list(resp_texts)
```

Note `round_agent_responses` is initialized once before the rounds loop (a `[num_rounds][num_agents]` matrix), populated as each round completes, then consumed once after all rounds.

- [ ] **Step 5.7: Mirror the wiring in the SRPO training loop**

Apply the same `round_agent_responses` collection and `_compute_cheap_diagnostics` call inside `RayRiskAverseTrainer.mappo_fit` (the second loop at line ~1981). The SRPO loop has the same shape, so the diff is identical — diagnostics are emitted before `back_propogate_reward` mutates scores. Identical metric prefix `train/`.

- [ ] **Step 5.8: Smoke import**

Run: `python -c "from verl.trainer.ppo.mappo_trainer import _extract_final_answer, _compute_cheap_diagnostics, RayMAPPOTrainer, RayRiskAverseTrainer; print('ok')"`

Expected: `ok`.

- [ ] **Step 5.9: Re-run all unit tests**

Run: `pytest tests/trainer/ppo/test_mappo_prompt_template.py tests/trainer/ppo/test_mappo_per_agent_entropy.py tests/trainer/ppo/test_mappo_cheap_diagnostics.py -xvs`

Expected: 11 passed total.

- [ ] **Step 5.10: Commit**

```bash
git add tests/trainer/ppo/test_mappo_cheap_diagnostics.py verl/trainer/ppo/mappo_trainer.py
git commit -m "feat(mappo): cheap training-step diagnostics (accuracy, agreement, recovery, flip, repetition)"
```

---

## Task 6: Cross-pair probe at validation cadence

This is the largest task. It splits into three subtasks: (6a) lazy partner instantiation, (6b) probe rollout loop with both directions, (6c) validation-cadence wiring.

**Files:**
- Modify: `verl/trainer/ppo/mappo_trainer.py`

### Task 6a: Lazy partner worker-group instantiation

The probe partner is a third actor worker group, instantiated only when `multi_agent.cross_pair_probe.partner_ckpt_dir` is non-empty. It reuses the same `RayClassWithInitArgs` machinery as training agents and is loaded once via the existing FSDP load path.

- [ ] **Step 6a.1: Locate the worker-group init block**

Run: `grep -n "self.actor_rollout_wgs\b\|self.actor_rollout_wgs =" verl/trainer/ppo/mappo_trainer.py | head`

Confirm the assignment at ~line 849 (`self.actor_rollout_wgs = {}`) and the population loop at ~line 853.

- [ ] **Step 6a.2: Register the probe partner class before the spawn loop**

In `init_workers`, find the line `# initialize WorkerGroup` (line 829, immediately before the spawn loop at line 836). Insert this block **right before** that comment line, after the per-agent registration loop ends (line 800):

```python
        # Cross-pair probe partner (spec §6.2). Loaded only if partner_ckpt_dir is set.
        probe_cfg = OmegaConf.select(self.config, "multi_agent.cross_pair_probe", default={}) or {}
        partner_ckpt_dir = str(probe_cfg.get("partner_ckpt_dir", "") or "")
        if partner_ckpt_dir:
            # Reuse agent_pool_0 since the probe runs at val cadence and isn't on the training critical path.
            partner_pool = self.resource_pool_manager.get_resource_pool("agent_pool_0") \
                or self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            # Use agent-0's actor config as the structural template; the partner's
            # weights come from partner_ckpt_dir, not from this config.
            partner_actor_cfg = OmegaConf.merge(
                deepcopy(self.config.actor_rollout_ref),
                OmegaConf.select(self.config, "multi_agent.agents.0.actor", default={}) or {},
            )
            OmegaConf.update(partner_actor_cfg, "rollout.agent_index", 99, force_add=True)
            partner_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=partner_actor_cfg,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[partner_pool]["actor_rollout_probe_partner"] = partner_cls
            self._partner_ckpt_dir = partner_ckpt_dir  # consumed in Step 6a.3
        else:
            self._partner_ckpt_dir = None
```

- [ ] **Step 6a.2b: Pull the partner out of `all_wg` after the existing per-agent loop**

After the existing per-agent loop at line 849-853:

```python
        self.actor_rollout_wgs = {}
        for i in range(num_agents):
            actor_wg = all_wg[f"actor_rollout_{i}"]
            actor_wg.init_model()
            self.actor_rollout_wgs[f"model_{i}"] = actor_wg
```

Append (still inside `init_workers`):

```python
        if self._partner_ckpt_dir:
            partner_wg = all_wg["actor_rollout_probe_partner"]
            partner_wg.init_model()
            self.actor_rollout_wgs["probe_partner"] = partner_wg
```

- [ ] **Step 6a.3: Load partner weights once at trainer init**

Find `_load_checkpoint` at line 979. The existing per-agent load at line 1023 calls:

```python
            self.actor_rollout_wgs[f"model_{i}"].load_checkpoint(
                agent_local_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
```

Right after the `# load actor` block (after line 1025), add the partner load. The partner's checkpoint dir layout matches the trained agents' layout (since the partner *is* a previously trained agent), so the actor weights live under `<partner_ckpt_dir>/actor/0/`:

```python
        if getattr(self, "_partner_ckpt_dir", None):
            partner_actor_path = os.path.join(self._partner_ckpt_dir, "actor", "0")
            # del_local_after_load=False — never delete the sibling arm's checkpoint.
            self.actor_rollout_wgs["probe_partner"].load_checkpoint(
                partner_actor_path, del_local_after_load=False
            )
```

Note: `partner_ckpt_dir` is expected to be a `global_step_N` directory (see Step 8.3 which passes `checkpoints/ippo_q05b_dbg/global_step_2`). If the user passes a base ckpt root by mistake, the load will fail with a clear FSDP error — preferred over silent miswiring.

- [ ] **Step 6a.4: Smoke import**

Run: `python -c "from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer; print('ok')"`

Expected: `ok`. (Functional probe-load behavior is verified later in the smoke ladder, not as a unit test — the FSDP/Ray plumbing has no easy CPU-only test.)

- [ ] **Step 6a.5: Commit**

```bash
git add verl/trainer/ppo/mappo_trainer.py
git commit -m "feat(mappo): lazy probe_partner actor worker group instantiation + checkpoint load"
```

### Task 6b: Probe rollout loop

Implement `_run_cross_pair_probe()` that runs the full validation rollout twice (one per direction) with the probe partner swapped in for one of the two agents.

- [ ] **Step 6b.1: Refactor validation to a parameterized inner**

Find the validation rollout (line ~458 from Task 3). Extract its inner per-(direction) body into a new method `_validate_with_agents(agent_keys_override)` so we can call it with `["model_0", "probe_partner"]`, `["probe_partner", "model_1"]`, and the default.

If extracting cleanly is invasive, an alternative is to copy the validation rollout body into `_run_cross_pair_probe` and parameterize it there. Pick whichever produces the smaller diff. Document the choice in the commit message.

(Recommendation: copy-paste into `_run_cross_pair_probe` to avoid touching the validation path. The validation method is large and has many call sites for metric aggregation; an extracted helper would have a long parameter list.)

- [ ] **Step 6b.2: Implement `_run_cross_pair_probe`**

Add as a method on `RayMAPPOTrainer`. Signature:

```python
    def _run_cross_pair_probe(self) -> dict:
        """Run validation-shaped rollouts with one trained agent swapped for probe_partner.

        Returns a flat dict of cross_pair/<...> metric keys ready to merge into the
        trainer's metric dict. Returns {} if probe_partner is not loaded.
        """
        if not getattr(self, "_partner_ckpt_dir", None):
            return {}

        ma = OmegaConf.select(self.config, "multi_agent", default={}) or {}
        num_agents = int(ma.get("num_agents", 2))
        num_rounds = int(ma.get("num_rounds", 1))
        assert num_agents == 2, "Cross-pair probe requires exactly 2 agents."

        agent_keys = list(self.train_dataloaders.keys())  # e.g., ["model_0", "model_1"]
        # Direction A: trained agent_0 + probe_partner-as-agent_1.
        # Direction B: probe_partner-as-agent_0 + trained agent_1.
        directions = {
            "srpo_first": [agent_keys[0], "probe_partner"],
            "srpo_second": ["probe_partner", agent_keys[1]],
        }

        out: dict = {}
        for direction, keys_for_round in directions.items():
            round_correctness = {}
            round_responses = {}
            round_response_lens = {}
            joint_acc_final = None

            for batch_idx, batch_tuple in enumerate(zip(*(self.val_dataloaders[k] for k in agent_keys))):
                batch_size = len(DataProto.from_single_dict(batch_tuple[0]))
                self_history = [[""] * batch_size for _ in range(num_agents)]
                for r in range(num_rounds):
                    this_round_responses = [[""] * batch_size for _ in range(num_agents)]
                    for agent_idx in range(num_agents):
                        # The dataset is the agent_idx-th val_dataloader; the worker is keys_for_round[agent_idx].
                        worker_key = keys_for_round[agent_idx]
                        batch_dict = batch_tuple[agent_idx]
                        test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                        questions, _ = self._extract_prompts_and_questions(test_batch, worker_key)
                        if r == 0:
                            chat_prompts = questions
                        else:
                            peer_idx = 1 - agent_idx
                            chat_prompts = self._build_input_ids_from_histories(
                                questions=questions,
                                self_history=self_history[agent_idx],
                                peer_history=self_history[peer_idx],
                                r=r,
                                batch=test_batch,
                                agent_key=worker_key,
                                max_history_tokens=4096,
                            )
                        if "uid" not in test_batch.non_tensor_batch:
                            test_batch.non_tensor_batch["uid"] = np.array(
                                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                            )
                        gen_batch = self._get_gen_batch(test_batch)
                        gen_batch.meta_info["global_steps"] = self.global_steps
                        gen_batch_output = self.actor_rollout_wgs[worker_key].generate_sequences(gen_batch)
                        test_batch = test_batch.union(gen_batch_output)
                        test_batch.meta_info["validate"] = True

                        result = self.val_reward_fns[agent_keys[agent_idx]](test_batch, return_dict=True)
                        scores = result["reward_tensor"].sum(-1).cpu().numpy()
                        output_ids = test_batch.batch["responses"]
                        output_texts = [
                            self.tokenizers[agent_keys[agent_idx]].decode(ids, skip_special_tokens=True)
                            for ids in output_ids
                        ]
                        response_lens = test_batch.batch["response_mask"].sum(-1).cpu().numpy()

                        round_correctness.setdefault(r, {})[agent_idx] = (scores > 0.5).astype("int64")
                        round_responses.setdefault(r, {})[agent_idx] = list(output_texts)
                        round_response_lens.setdefault(r, {})[agent_idx] = response_lens
                        this_round_responses[agent_idx] = list(output_texts)
                    for a in range(num_agents):
                        self_history[a] = this_round_responses[a]

                # NOTE: probe accumulates ONE batch only — short-circuit here. Spec §6.2 says
                # probe shares the val set; full enumeration matches normal validation cost.
                # Remove this break to enumerate the full val set (recommended for production).
                break

            # Joint accuracy at final round.
            r_last = num_rounds - 1
            both_right = (round_correctness[r_last][0] & round_correctness[r_last][1]).astype(float)
            out[f"cross_pair/joint_acc/{direction}"] = float(both_right.mean())

            # Trained-agent accuracy: the index that's NOT probe_partner.
            trained_idx = 0 if keys_for_round[0] != "probe_partner" else 1
            out[f"cross_pair/trained_agent_acc/{direction}"] = float(
                round_correctness[r_last][trained_idx].mean()
            )

            # Replay all §6.1 metrics under cross_pair/<metric>/<direction>/...
            sub = _compute_cheap_diagnostics(
                num_rounds=num_rounds,
                num_agents=num_agents,
                correctness=round_correctness,
                responses=round_responses,
                response_lens=round_response_lens,
            )
            for k, v in sub.items():
                out[f"cross_pair/{k}/{direction}"] = v

        return out
```

Note the deliberate `break` inside the val-batch loop. For state-A iteration, one validation batch suffices. The smoke ladder uses one batch; production runs should remove the `break` to enumerate the full val set. The `break` is documented inline.

- [ ] **Step 6b.3: Smoke import**

Run: `python -c "from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer; print('ok')"`

Expected: `ok`.

- [ ] **Step 6b.4: Commit**

```bash
git add verl/trainer/ppo/mappo_trainer.py
git commit -m "feat(mappo): _run_cross_pair_probe with both directions and §6.1 metric replay"
```

### Task 6c: Wire probe into validation cadence

- [ ] **Step 6c.1: Find validation invocation sites**

Run: `grep -n "self._validate\b\|_validate(" verl/trainer/ppo/mappo_trainer.py | head`

Confirm where the existing validation method is called from inside `mappo_fit` (both the IPPO and SRPO paths).

- [ ] **Step 6c.2: Insert probe call after validation**

In each of the two `mappo_fit` methods, immediately after the existing `_validate` invocation (which produces `val_metrics`), add:

```python
                    probe_metrics = self._run_cross_pair_probe()
                    val_metrics.update(probe_metrics)
```

If the existing validation call returns into a different variable name, match it. The probe metrics merge into the same dict that `Tracking.log` consumes at the end of the validation cadence block.

- [ ] **Step 6c.3: Smoke import**

Run: `python -c "from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer, RayRiskAverseTrainer; print('ok')"`

Expected: `ok`.

- [ ] **Step 6c.4: Commit**

```bash
git add verl/trainer/ppo/mappo_trainer.py
git commit -m "feat(mappo): wire cross-pair probe into validation cadence"
```

---

## Task 7: Update runner scripts

**Files:**
- Modify: `debug_q05b_local.sh`
- Modify: `train_q05b.slurm`

- [ ] **Step 7.1: Read the current debug runner**

Run: `cat debug_q05b_local.sh`

Confirm the current `METHOD` switch (lines 67-75 in the snapshot) and the python invocation (lines 80-126).

- [ ] **Step 7.2: Extend `METHOD` to three arms in `debug_q05b_local.sh`**

Replace the existing block:

```bash
if [ "$METHOD" = "srpo" ]; then
    TRAINER_TYPE="risk_averse"
    KL_COEF=0.1
    EXP_NAME="srpo_q05b_dbg"
else
    TRAINER_TYPE="mappo"
    KL_COEF=0.001
    EXP_NAME="ippo_q05b_dbg"
fi
```

With:

```bash
case "$METHOD" in
    srpo_main)
        TRAINER_TYPE="risk_averse"
        KL_COEF=0.1
        EXP_NAME="srpo_main_q05b_dbg"
        ENTROPY_OVERRIDE="multi_agent.agents.0.actor.actor.entropy_coeff=0.05"
        ;;
    srpo_reward_only)
        TRAINER_TYPE="risk_averse"
        KL_COEF=0.1
        EXP_NAME="srpo_reward_only_q05b_dbg"
        ENTROPY_OVERRIDE=""
        ;;
    ippo|*)
        TRAINER_TYPE="mappo"
        KL_COEF=0.001
        EXP_NAME="ippo_q05b_dbg"
        ENTROPY_OVERRIDE=""
        ;;
esac

# Cross-pair probe partner: optional. Default empty (probe disabled).
# To enable, export PARTNER_CKPT_DIR=/path/to/sibling/arm/ckpt before invocation.
PARTNER_CKPT_DIR="${PARTNER_CKPT_DIR:-}"
```

Then change the python invocation block (around line 80-126) to add three lines:

```bash
PYTHONUNBUFFERED=1 python -m verl.trainer.main_mappo \
    ... existing args ...
    multi_agent.cross_pair_probe.partner_ckpt_dir=${PARTNER_CKPT_DIR} \
    ${ENTROPY_OVERRIDE:+${ENTROPY_OVERRIDE} \\}
    2>&1 | tee "${LOG_FILE}"
```

(The `${ENTROPY_OVERRIDE:+...}` form expands only when non-empty, so `srpo_reward_only` and `ippo` arms get no entropy override.)

Also update the `multi_agent.adversary_kl_to_hero=true` line: keep it `true` for both srpo arms; for ippo it doesn't matter (the IPPO trainer ignores it).

- [ ] **Step 7.3: Mirror the same changes in `train_q05b.slurm`**

Run: `cat train_q05b.slurm`

Apply the equivalent `case "$METHOD" in ... esac` block and the same two CLI arg additions. Slurm-specific shell quoting may differ slightly; preserve the existing pattern.

- [ ] **Step 7.4: Verify shell syntax**

Run: `bash -n debug_q05b_local.sh && bash -n train_q05b.slurm`

Expected: no output (clean parse).

- [ ] **Step 7.5: Dry-run — verify Hydra args are constructed correctly**

Insert `echo` immediately before the `python -m verl.trainer.main_mappo` line in `debug_q05b_local.sh` (so the script prints the full command instead of executing it), then run for each arm:

```bash
METHOD=ippo bash debug_q05b_local.sh 2>&1 | grep -E "entropy_coeff|cross_pair_probe.partner"
METHOD=srpo_main bash debug_q05b_local.sh 2>&1 | grep -E "entropy_coeff|cross_pair_probe.partner"
METHOD=srpo_reward_only bash debug_q05b_local.sh 2>&1 | grep -E "entropy_coeff|cross_pair_probe.partner"
```

Expected:
- `ippo`: prints `multi_agent.cross_pair_probe.partner_ckpt_dir=` (empty value, no entropy line).
- `srpo_main`: prints both `multi_agent.cross_pair_probe.partner_ckpt_dir=...` and `multi_agent.agents.0.actor.actor.entropy_coeff=0.05`.
- `srpo_reward_only`: prints `partner_ckpt_dir=` only, no entropy line.

Remove the `echo` prefix once verified.

- [ ] **Step 7.6: Commit**

```bash
git add debug_q05b_local.sh train_q05b.slurm
git commit -m "feat(runners): METHOD switch supports {ippo,srpo_main,srpo_reward_only} + cross-pair partner"
```

---

## Task 8: Smoke ladder verification

This task is the spec's verification gate (§9). It does not produce code; it verifies the prior tasks integrate end-to-end. Each step runs an actual debug job. Allocate ~10 minutes per arm on a 2-GPU node.

**Files:** none modified.

- [ ] **Step 8.1: Unit tests pass**

Run: `pytest tests/trainer/ppo/test_mappo_prompt_template.py tests/trainer/ppo/test_mappo_per_agent_entropy.py tests/trainer/ppo/test_mappo_cheap_diagnostics.py -xvs`

Expected: 11 passed.

- [ ] **Step 8.2: IPPO smoke run produces a usable checkpoint**

Run: `METHOD=ippo bash debug_q05b_local.sh` after temporarily editing the script to set `trainer.save_freq=1` (or pass it via env). Expected log indicators:
- Console contains `train/accuracy/round_0/agent_0=...` and `train/agreement_rate/round_0=...` keys.
- `checkpoints/ippo_q05b_dbg/global_step_2/` directory exists.
- Exit code 0.

If save_freq=1 is awkward to inject, edit the script's `trainer.save_freq=50` to `trainer.save_freq=1` for this verification run; revert before commit.

- [ ] **Step 8.3: SRPO main smoke run produces cross_pair metrics and entropy gap**

Run:

```bash
PARTNER_CKPT_DIR=$(realpath checkpoints/ippo_q05b_dbg/global_step_2) \
    METHOD=srpo_main \
    bash debug_q05b_local.sh
```

Expected:
- Console contains `cross_pair/joint_acc/srpo_first` and `cross_pair/joint_acc/srpo_second` keys.
- `train/kl_adv_to_hero/mean` strictly positive (not 0, not NaN).
- `train/entropy/agent_0` > `train/entropy/agent_1` by a margin consistent with the 5x ε ratio.
- Exit code 0.

- [ ] **Step 8.4: SRPO reward-only smoke run shows ≈ symmetric entropy**

Run:

```bash
PARTNER_CKPT_DIR=$(realpath checkpoints/ippo_q05b_dbg/global_step_2) \
    METHOD=srpo_reward_only \
    bash debug_q05b_local.sh
```

Expected:
- `train/entropy/agent_0` ≈ `train/entropy/agent_1` (within sampling noise, since both ε=0.01).
- `cross_pair/*` keys still present.
- Exit code 0.

- [ ] **Step 8.5: Spec-success record**

If all four steps pass, the spec is implementation-complete. Add a one-line entry to the bottom of `docs/superpowers/specs/2026-04-25-srpo-revisions-design.md` under a new heading:

```markdown
## Implementation status

- 2026-MM-DD: smoke ladder passed on 2-GPU debug node. See `logs/{ippo,srpo_main,srpo_reward_only}_q05b_dbg_*.log`.
```

(Use the actual completion date.)

```bash
git add docs/superpowers/specs/2026-04-25-srpo-revisions-design.md
git commit -m "docs(srpo): record smoke ladder pass on 2-GPU debug node"
```

---

## Done criteria

After all tasks:
1. `pytest tests/trainer/ppo/test_mappo_*.py` → 11 passed.
2. `git log --oneline | head -10` shows roughly 11 commits along the lines of:
   - `feat(mappo): add system_prompt, discussion_prompt_template, cross_pair_probe config fields`
   - `feat(mappo): add _format_discussion_prompt helper with brace-safe substitution`
   - `refactor(mappo): per-agent self/peer histories + templated discussion prompt`
   - `test(mappo): lock per-agent actor.entropy_coeff override contract`
   - `feat(mappo): cheap training-step diagnostics ...`
   - `feat(mappo): lazy probe_partner actor worker group instantiation + checkpoint load`
   - `feat(mappo): _run_cross_pair_probe ...`
   - `feat(mappo): wire cross-pair probe into validation cadence`
   - `feat(runners): METHOD switch ...`
   - `docs(srpo): record smoke ladder pass ...`
3. Smoke ladder Steps 8.2-8.4 all completed with the expected metric keys present.

If any step fails: stop, do not paper over with try/except, identify root cause, fix, then re-run from the failing step. Do not skip ahead.
