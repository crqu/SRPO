## Project Context

This repo exists to support the empirical claim in **https://arxiv.org/html/2602.21515**:
*SRPO yields better partner-generalization than IPPO in multi-agent LLM training.*
Every change should be evaluated against that goal (does it sharpen the comparison, or just add features?).

### Where the algorithms live

- Entry point: `verl/trainer/main_mappo.py` (Hydra app; `multi_agent.trainer_type` selects which class).
- Core algorithm: `verl/trainer/ppo/mappo_trainer.py`
  - `RayMAPPOTrainer` — **IPPO** baseline. Each agent runs independent PPO; rewards from the env without cross-agent regularization.
  - `RayRiskAverseTrainer(RayMAPPOTrainer)` — **SRPO**. Agent 0 = adversary, Agent 1 = hero. `back_propogate_reward` flips the adversary's reward to `-hero_return`; `_apply_kl_penalty` (gated by `multi_agent.adversary_kl_to_hero`) regularizes the adversary toward the hero via `(1/τ) · KL(π_adv ‖ π_hero)` evaluated on adversary trajectories with the hero's logprobs.

The IPPO vs SRPO comparison must use the **same** rollout/critic/actor stacks; only the trainer class and adversary regularizer differ. Don't introduce features that touch only one side without an honest matched ablation.

### Active research directions (do not regress these)

- Better training setup and discussion-prompt design to make the SRPO > IPPO claim hold robustly across seeds and partner distributions. Treat the discussion prompt as a tunable component, not a constant.
- Extension to GRPO-style algorithms (group-relative advantages instead of value-fn baselines). Future trainers will likely subclass `RayMAPPOTrainer` similarly to `RayRiskAverseTrainer`.

### Operational notes for this repo

- Conda env: `module load anaconda3/2024.02-1 && conda activate srpo` (env path: `/home/cqu9/scratchlshi40_llm/conda_envs/srpo`).
- 2-GPU local debug: `bash debug_q05b_local.sh` (no slurm). Multi-node prod: `sbatch train_q05b.slurm`.
- Smoke runs reuse `checkpoints/srpo_q05b_dbg/` — pass `trainer.resume_mode=disable` or wipe the dir or you'll silently auto-resume into a no-op.
- vLLM HYBRID + LoRA + free_cache_engine=true exposes a chain of cumem sleep/wake bugs (see memory). Default debug uses `actor_rollout_ref.rollout.free_cache_engine=false` to sidestep.

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### 7. Project Instructions
- load anaconda first using "module load anaconda3/2024.02-1"
- use conda environment for managing dependencies and running all the tests: "conda activate srpo"

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items  
2. **Verify Plan**: Check in before starting implementation  
3. **Track Progress**: Mark items complete as you go  
4. **Explain Changes**: High-level summary at each step  
5. **Document Results**: Add review section to `tasks/todo.md`  
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections 

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
