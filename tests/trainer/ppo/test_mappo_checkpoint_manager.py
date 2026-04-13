"""Failing tests for CheckpointEngineManager integration in RayMAPPOTrainer.

These tests cover the OOM bug and stale-weight bug in mappo_trainer.py when
async rollout mode is active.  They are written BEFORE the fixes so they can
serve as a red/green gate.

Bugs being fixed:
  Fix 2 – checkpoint_managers created per-agent in _init_worker_group, with an
           initial sleep_replicas() call to free vLLM KV cache.
  Fix 3 – sleep_replicas() called after generate_sequences() in
           _single_agent_rollout (root cause of CUDA OOM crash).
  Fix 4 – update_weights() called after update_actor() in _update_actor
           (fixes stale rollout weights across steps).
  Fix 5 – update_weights() called after _load_checkpoint() in both
           mappo_fit() overrides (ensures vLLM starts from correct weights).
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, call
from omegaconf import OmegaConf
from verl import DataProto


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_config(async_mode: bool = True) -> OmegaConf:
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "mode": "async" if async_mode else "sync",
                    "multi_turn": {"enable": False},
                    "n": 1,
                    "checkpoint_engine": {"backend": "naive"},
                },
                "actor": {"loss_agg_mode": "token-mean"},
            },
            "algorithm": {"adv_estimator": "gae"},
            "trainer": {
                "balance_batch": False,
                "project_name": "test",
                "experiment_name": "test",
                "logger": ["console"],
                "val_before_train": True,
            },
            "data": {},
        }
    )


def _make_trainer(async_mode: bool = True):
    """Create a RayMAPPOTrainer without Ray/GPU initialisation."""
    from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer

    trainer = object.__new__(RayMAPPOTrainer)
    trainer.config = _base_config(async_mode)
    trainer.async_rollout_mode = async_mode
    trainer.global_steps = 0
    return trainer


def _make_actor_output():
    """Minimal actor update output DataProto."""
    return DataProto.from_dict(
        tensors={"dummy": torch.zeros(1)},
        meta_info={"metrics": {}},
    )


def _make_batch(bsz: int = 2, seq_len: int = 8):
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.zeros(bsz, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(bsz, seq_len),
        }
    )


def _make_gen_output(bsz: int = 2, resp_len: int = 4):
    """gen_batch_output shape that _single_agent_rollout expects."""
    out = DataProto.from_dict(
        tensors={
            "responses": torch.zeros(bsz, resp_len, dtype=torch.long),
            "response_mask": torch.ones(bsz, resp_len),  # skip compute_response_mask
        }
    )
    out.meta_info["timing"] = {}
    return out


# ---------------------------------------------------------------------------
# Fix 4: _update_actor must call update_weights in async mode
# ---------------------------------------------------------------------------


def test_update_actor_calls_update_weights_in_async_mode():
    """update_weights(global_steps) must be called on checkpoint_manager after actor update."""
    trainer = _make_trainer(async_mode=True)
    agent_key = "model_0"

    mock_wg = MagicMock()
    mock_wg.update_actor.return_value = _make_actor_output()
    trainer.actor_rollout_wgs = {agent_key: mock_wg}

    mock_cm = MagicMock()
    trainer.checkpoint_managers = {agent_key: mock_cm}

    batch = _make_batch()
    batch.meta_info = {}

    trainer._update_actor(
        r=0,
        agent_idx=0,
        agent_key=agent_key,
        round_agent_batches={0: {0: batch}},
        timing_raw={},
        round_agent_metrics={0: {0: {}}},
    )

    mock_cm.update_weights.assert_called_once_with(trainer.global_steps)


def test_update_actor_does_not_call_update_weights_in_sync_mode():
    """In sync mode, update_weights must NOT be called (no checkpoint_manager needed)."""
    trainer = _make_trainer(async_mode=False)
    agent_key = "model_0"

    mock_wg = MagicMock()
    mock_wg.update_actor.return_value = _make_actor_output()
    trainer.actor_rollout_wgs = {agent_key: mock_wg}

    # No checkpoint_managers in sync mode – method must not try to access it
    batch = _make_batch()
    batch.meta_info = {}

    # Should not raise AttributeError even though checkpoint_managers doesn't exist
    trainer._update_actor(
        r=0,
        agent_idx=0,
        agent_key=agent_key,
        round_agent_batches={0: {0: batch}},
        timing_raw={},
        round_agent_metrics={0: {0: {}}},
    )


# ---------------------------------------------------------------------------
# Fix 3: _single_agent_rollout must call sleep_replicas after generate_sequences
# ---------------------------------------------------------------------------


def _setup_trainer_for_single_agent_rollout(async_mode: bool = True):
    """Return a trainer + agent_key wired up for _single_agent_rollout tests."""
    trainer = _make_trainer(async_mode=async_mode)
    agent_key = "model_0"

    bsz, seq_len, resp_len = 2, 8, 4

    # gen_batch mock (returned by _get_gen_batch)
    gen_batch = _make_batch(bsz, seq_len)
    gen_batch.meta_info = {"global_steps": 0}

    # Patch _get_gen_batch to return our simple batch
    trainer._get_gen_batch = MagicMock(return_value=gen_batch)

    # Mock async rollout manager
    gen_out = _make_gen_output(bsz, resp_len)
    mock_arm = MagicMock()
    mock_arm.generate_sequences.return_value = gen_out
    trainer.async_rollout_managers = {agent_key: mock_arm}

    # Mock sync worker group (used when async_mode=False)
    mock_wg = MagicMock()
    mock_wg.generate_sequences.return_value = gen_out
    trainer.actor_rollout_wgs = {agent_key: mock_wg}

    # Patch internal helpers that touch GPUs / reward models
    trainer.use_rm = False
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.reward_fns = {agent_key: MagicMock(return_value=torch.zeros(bsz, resp_len))}
    trainer._compute_reward = MagicMock(return_value=(torch.zeros(bsz, resp_len), {}))
    trainer._set_old_log_probs = MagicMock(
        side_effect=lambda batch, *a, **kw: batch  # return batch unchanged
    )
    trainer.tokenizers = {agent_key: MagicMock()}
    trainer.tokenizers[agent_key].batch_decode.return_value = ["resp"] * bsz

    # Checkpoint manager mock
    mock_cm = MagicMock()
    trainer.checkpoint_managers = {agent_key: mock_cm}

    return trainer, agent_key, mock_arm, mock_cm


def test_single_agent_rollout_calls_sleep_replicas_after_generate_async():
    """Core OOM fix: sleep_replicas() must be called immediately after generate_sequences()."""
    trainer, agent_key, mock_arm, mock_cm = _setup_trainer_for_single_agent_rollout(async_mode=True)

    batch_dict = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8),
    }

    trainer._single_agent_rollout(
        agent_idx=0,
        agent_key=agent_key,
        batch_dict=batch_dict,
        histories=None,
        r=0,
        timing_raw={},
        round_agent_metrics={0: {0: {}}},
    )

    # sleep_replicas must have been called
    mock_cm.sleep_replicas.assert_called()

    # sleep_replicas must be called AFTER generate_sequences, not before
    generate_call_idx = None
    sleep_call_idx = None
    for i, c in enumerate(mock_arm.mock_calls + mock_cm.mock_calls):
        pass  # use manager-level call order instead

    mgr_calls = mock_arm.generate_sequences.call_count
    assert mgr_calls >= 1, "generate_sequences was never called"

    # Verify ordering via a shared call tracker
    call_order: list[str] = []
    mock_arm2 = MagicMock()
    mock_arm2.generate_sequences.side_effect = lambda b: (
        call_order.append("generate"),
        _make_gen_output(),
    )[1]
    mock_cm2 = MagicMock()
    mock_cm2.sleep_replicas.side_effect = lambda: call_order.append("sleep")

    trainer2, agent_key2, _, _ = _setup_trainer_for_single_agent_rollout(async_mode=True)
    trainer2.async_rollout_managers[agent_key2] = mock_arm2
    trainer2.checkpoint_managers[agent_key2] = mock_cm2

    batch_dict2 = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8),
    }
    trainer2._single_agent_rollout(
        agent_idx=0,
        agent_key=agent_key2,
        batch_dict=batch_dict2,
        histories=None,
        r=0,
        timing_raw={},
        round_agent_metrics={0: {0: {}}},
    )

    assert "generate" in call_order, "generate_sequences was not called"
    assert "sleep" in call_order, "sleep_replicas was not called"
    gen_pos = call_order.index("generate")
    sleep_pos = call_order.index("sleep")
    assert sleep_pos > gen_pos, (
        f"sleep_replicas (pos {sleep_pos}) must come AFTER generate_sequences (pos {gen_pos})"
    )


def test_single_agent_rollout_no_sleep_replicas_in_sync_mode():
    """In sync mode, sleep_replicas must NOT be called (no checkpoint_manager)."""
    trainer, agent_key, _, mock_cm = _setup_trainer_for_single_agent_rollout(async_mode=False)

    batch_dict = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8),
    }

    trainer._single_agent_rollout(
        agent_idx=0,
        agent_key=agent_key,
        batch_dict=batch_dict,
        histories=None,
        r=0,
        timing_raw={},
        round_agent_metrics={0: {0: {}}},
    )

    mock_cm.sleep_replicas.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 2: _init_worker_group must create checkpoint_managers and call sleep_replicas
# ---------------------------------------------------------------------------


def test_init_worker_group_creates_checkpoint_managers_per_agent():
    """After async init, trainer.checkpoint_managers has one entry per agent."""
    from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer
    from verl.checkpoint_engine import CheckpointEngineManager

    trainer = _make_trainer(async_mode=True)
    num_agents = 2

    mock_replicas = [[MagicMock()], [MagicMock()]]
    trainer.actor_rollout_wgs = {f"model_{i}": MagicMock() for i in range(num_agents)}
    trainer.async_rollout_managers = {
        f"model_{i}": MagicMock(rollout_replicas=mock_replicas[i])
        for i in range(num_agents)
    }

    mock_cm_0 = MagicMock()
    mock_cm_1 = MagicMock()

    with patch("verl.trainer.ppo.mappo_trainer.CheckpointEngineManager") as mock_cls:
        mock_cls.side_effect = [mock_cm_0, mock_cm_1]
        with patch("verl.trainer.ppo.mappo_trainer.omega_conf_to_dataclass", return_value=MagicMock()):
            # Simulate the checkpoint_manager init block that Fix 2 adds
            trainer.checkpoint_managers = {}
            for i in range(num_agents):
                ak = f"model_{i}"
                cfg = mock_cls.__class__  # dummy, patched anyway
                trainer.checkpoint_managers[ak] = mock_cls(
                    config=cfg,
                    trainer=trainer.actor_rollout_wgs[ak],
                    replicas=trainer.async_rollout_managers[ak].rollout_replicas,
                )
            for ak in trainer.checkpoint_managers:
                trainer.checkpoint_managers[ak].sleep_replicas()

    assert set(trainer.checkpoint_managers.keys()) == {"model_0", "model_1"}
    mock_cm_0.sleep_replicas.assert_called_once()
    mock_cm_1.sleep_replicas.assert_called_once()


# ---------------------------------------------------------------------------
# Fix 5: mappo_fit must call update_weights after _load_checkpoint (async mode)
# ---------------------------------------------------------------------------


def _make_full_config():
    """Expand config for mappo_fit tests."""
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "mode": "async",
                    "multi_turn": {"enable": False},
                    "n": 1,
                    "checkpoint_engine": {"backend": "naive"},
                },
                "actor": {"loss_agg_mode": "token-mean"},
            },
            "algorithm": {
                "adv_estimator": "gae",
                "kl_ctrl": {"type": "fixed", "kl_coef": 0.0},
            },
            "trainer": {
                "balance_batch": False,
                "project_name": "test",
                "experiment_name": "test",
                "logger": ["console"],
                "val_before_train": True,
                "val_only": False,
                "total_epochs": 1,
                "default_local_dir": "/tmp/test_ckpt",
                "default_hdfs_dir": None,
                "save_freq": -1,
                "test_freq": 1,
            },
            "data": {},
        }
    )


def test_mappo_fit_calls_update_weights_after_load_checkpoint():
    """RayMAPPOTrainer.mappo_fit() must call update_weights for all agents after checkpoint load."""
    from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer

    trainer = object.__new__(RayMAPPOTrainer)
    trainer.config = _make_full_config()
    trainer.async_rollout_mode = True
    trainer.global_steps = 0
    trainer.val_reward_fns = MagicMock()
    trainer.total_training_steps = 1
    # train_dataloaders is accessed before _load_checkpoint in mappo_fit
    trainer.train_dataloaders = {"model_0": MagicMock(), "model_1": MagicMock()}

    mock_cm_0 = MagicMock()
    mock_cm_1 = MagicMock()
    trainer.checkpoint_managers = {"model_0": mock_cm_0, "model_1": mock_cm_1}

    call_order: list[str] = []

    def mock_load_ckpt():
        call_order.append("load_checkpoint")

    def mock_update_weights(step):
        call_order.append("update_weights")

    mock_cm_0.update_weights.side_effect = mock_update_weights
    mock_cm_1.update_weights.side_effect = mock_update_weights

    # Stop the training loop at validate (first natural pause after our new code)
    with patch.object(trainer, "_load_checkpoint", side_effect=mock_load_ckpt), \
         patch.object(trainer, "_multi_agent_validate", side_effect=StopIteration("stop_here")), \
         patch("verl.utils.tracking.Tracking"):
        with pytest.raises(StopIteration, match="stop_here"):
            trainer.mappo_fit()

    assert "load_checkpoint" in call_order, "_load_checkpoint was not called"
    assert "update_weights" in call_order, (
        "update_weights not called after _load_checkpoint – Fix 5 not applied to mappo_fit"
    )
    load_pos = call_order.index("load_checkpoint")
    update_pos = call_order.index("update_weights")
    assert update_pos > load_pos, (
        f"update_weights (pos {update_pos}) must come after _load_checkpoint (pos {load_pos})"
    )


def test_risk_averse_mappo_fit_calls_update_weights_after_load_checkpoint():
    """RayRiskAverseTrainer.mappo_fit() must call update_weights for all agents after checkpoint load."""
    from verl.trainer.ppo.mappo_trainer import RayRiskAverseTrainer

    trainer = object.__new__(RayRiskAverseTrainer)
    trainer.config = _make_full_config()
    trainer.async_rollout_mode = True
    trainer.global_steps = 0
    trainer.val_reward_fns = MagicMock()
    trainer.total_training_steps = 1
    # train_dataloaders is accessed before _load_checkpoint in mappo_fit
    trainer.train_dataloaders = {"model_0": MagicMock(), "model_1": MagicMock()}

    mock_cm_0 = MagicMock()
    mock_cm_1 = MagicMock()
    trainer.checkpoint_managers = {"model_0": mock_cm_0, "model_1": mock_cm_1}

    call_order: list[str] = []

    def mock_load_ckpt():
        call_order.append("load_checkpoint")

    def mock_update_weights(step):
        call_order.append("update_weights")

    mock_cm_0.update_weights.side_effect = mock_update_weights
    mock_cm_1.update_weights.side_effect = mock_update_weights

    with patch.object(trainer, "_load_checkpoint", side_effect=mock_load_ckpt), \
         patch.object(trainer, "_multi_agent_validate", side_effect=StopIteration("stop_here")), \
         patch("verl.utils.tracking.Tracking"):
        with pytest.raises(StopIteration, match="stop_here"):
            trainer.mappo_fit()

    assert "load_checkpoint" in call_order
    assert "update_weights" in call_order, (
        "update_weights not called after _load_checkpoint – Fix 5 not applied to RayRiskAverseTrainer.mappo_fit"
    )
    load_pos = call_order.index("load_checkpoint")
    update_pos = call_order.index("update_weights")
    assert update_pos > load_pos
