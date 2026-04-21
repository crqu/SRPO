"""Unit tests for LoRA engine arg construction and generate() LoRA guard.

Mirrors logic from:
  - vllm_async_server.py: compilation_config + enforce_eager + enable_lora args
  - vllm_async_server.py: generate() RuntimeError when LoRA not loaded
  - utils.py: update_weights_from_ipc remove+add ordering, _update_weights dispatch

No GPU, ray, or torch imports required.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, call

# ---------------------------------------------------------------------------
# Constants (from utils.py)
# ---------------------------------------------------------------------------

VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

# ---------------------------------------------------------------------------
# Helpers mirroring vllm_async_server.py engine arg construction
# ---------------------------------------------------------------------------

def _build_lora_engine_args(
    lora_as_adapter: bool,
    lora_rank: int,
    config_enforce_eager: bool,
    compilation_config: dict,
) -> tuple[dict, dict]:
    """Mirror the LoRA-related engine arg and compilation_config logic.

    Reproduces vllm_async_server.py lines 239-369 so tests run without
    importing the real class (which requires ray, torch, vllm).
    """
    if lora_as_adapter:
        if "cudagraph_mode" not in compilation_config:
            compilation_config["cudagraph_mode"] = "NONE"
        elif compilation_config["cudagraph_mode"] != "NONE":
            compilation_config["cudagraph_mode"] = "NONE"
    else:
        compilation_config.setdefault("cudagraph_mode", "PIECEWISE")

    args = {
        "enforce_eager": True if lora_as_adapter else config_enforce_eager,
    }

    if lora_rank > 0:
        allowed = sorted([1, 8, 16, 32, 64, 128, 256, 320, 512])
        max_lora_rank = next((r for r in allowed if r >= lora_rank), allowed[-1])
        args.update({
            "enable_lora": True,
            "max_loras": 1,
            "max_lora_rank": max_lora_rank,
        })

    return args, compilation_config


class TestLoraEngineArgs:
    def test_cudagraph_mode_forced_none_when_lora(self):
        """cudagraph_mode must be NONE when lora_as_adapter=True."""
        _, cc = _build_lora_engine_args(
            lora_as_adapter=True,
            lora_rank=32,
            config_enforce_eager=False,
            compilation_config={},
        )
        assert cc["cudagraph_mode"] == "NONE"

    def test_cudagraph_mode_overridden_to_none_even_if_set(self):
        """cudagraph_mode=PIECEWISE is overridden to NONE for LoRA."""
        _, cc = _build_lora_engine_args(
            lora_as_adapter=True,
            lora_rank=32,
            config_enforce_eager=False,
            compilation_config={"cudagraph_mode": "PIECEWISE"},
        )
        assert cc["cudagraph_mode"] == "NONE"

    def test_enforce_eager_forced_true_when_lora(self):
        """enforce_eager must be True when lora_as_adapter=True, regardless of config."""
        args, _ = _build_lora_engine_args(
            lora_as_adapter=True,
            lora_rank=32,
            config_enforce_eager=False,
            compilation_config={},
        )
        assert args["enforce_eager"] is True

    def test_enable_lora_and_max_rank_set_for_rank_32(self):
        """enable_lora=True and max_lora_rank=32 for lora_rank=32."""
        args, _ = _build_lora_engine_args(
            lora_as_adapter=True,
            lora_rank=32,
            config_enforce_eager=False,
            compilation_config={},
        )
        assert args["enable_lora"] is True
        assert args["max_loras"] == 1
        assert args["max_lora_rank"] == 32

    def test_no_lora_override_when_lora_disabled(self):
        """When lora_as_adapter=False, enforce_eager and cudagraph_mode respect config."""
        args, cc = _build_lora_engine_args(
            lora_as_adapter=False,
            lora_rank=0,
            config_enforce_eager=False,
            compilation_config={},
        )
        assert args["enforce_eager"] is False
        assert cc["cudagraph_mode"] == "PIECEWISE"
        assert "enable_lora" not in args


# ---------------------------------------------------------------------------
# Helper mirroring generate() LoRA check (vllm_async_server.py:517-527)
# ---------------------------------------------------------------------------

async def _check_lora_for_generate(engine, lora_as_adapter: bool):
    """Mirror the LoRA guard in vLLMHttpServer.generate()."""
    lora_request = None
    if lora_as_adapter:
        lora_loaded = VLLM_LORA_INT_ID in await engine.list_loras()
        if not lora_loaded:
            raise RuntimeError(
                f"LoRA adapter (id={VLLM_LORA_INT_ID}) not found in vLLM engine. "
                "Weight sync may have failed or lora_as_adapter config is inconsistent."
            )
        lora_request = (VLLM_LORA_NAME, VLLM_LORA_INT_ID, VLLM_LORA_PATH)
    return lora_request


class TestGenerateLoraCheck:
    def test_raises_when_lora_not_loaded(self):
        """RuntimeError raised when lora_as_adapter=True and LoRA absent from engine."""
        engine = AsyncMock()
        engine.list_loras = AsyncMock(return_value=set())

        with pytest.raises(RuntimeError, match=str(VLLM_LORA_INT_ID)):
            asyncio.run(_check_lora_for_generate(engine, lora_as_adapter=True))

    def test_proceeds_when_lora_loaded(self):
        """No error and lora_request returned when LoRA present in engine."""
        engine = AsyncMock()
        engine.list_loras = AsyncMock(return_value={VLLM_LORA_INT_ID})

        result = asyncio.run(_check_lora_for_generate(engine, lora_as_adapter=True))
        assert result is not None
        assert result[1] == VLLM_LORA_INT_ID

    def test_skips_check_when_not_lora_adapter(self):
        """list_loras never called when lora_as_adapter=False."""
        engine = AsyncMock()
        engine.list_loras = AsyncMock(return_value=set())

        result = asyncio.run(_check_lora_for_generate(engine, lora_as_adapter=False))
        assert result is None
        engine.list_loras.assert_not_called()


# ---------------------------------------------------------------------------
# Helper mirroring update_weights_from_ipc + _update_weights (utils.py:179-264)
# ---------------------------------------------------------------------------

def _run_update_weights_ipc(worker, weights, peft_config, base_sync_done):
    """Mirror the LoRA remove+add logic from update_weights_from_ipc and _update_weights.

    Reproduces utils.py lines 188-264 without importing vllm/torch.
    """
    # remove old lora before adding new one (utils.py:188-190)
    if peft_config and base_sync_done:
        worker.remove_lora(VLLM_LORA_INT_ID)

    # dispatch to lora or base weight path (_update_weights, utils.py:242-264)
    if peft_config and base_sync_done:
        lora_request = {
            "lora_name": VLLM_LORA_NAME,
            "lora_int_id": VLLM_LORA_INT_ID,
            "lora_path": VLLM_LORA_PATH,
            "peft_config": peft_config,
            "lora_tensors": dict(weights),
        }
        worker.add_lora(lora_request)
    else:
        worker.load_weights(weights)


class TestUpdateWeightsIpc:
    def test_add_lora_called_with_peft_config(self):
        """add_lora called with TensorLoRARequest-like dict when peft_config present."""
        worker = MagicMock()
        peft_cfg = {"rank": 32}
        weights = [("lora_A", "tensor_a"), ("lora_B", "tensor_b")]

        _run_update_weights_ipc(worker, weights, peft_config=peft_cfg, base_sync_done=True)

        worker.add_lora.assert_called_once()
        call_kwargs = worker.add_lora.call_args[0][0]
        assert call_kwargs["lora_int_id"] == VLLM_LORA_INT_ID
        assert call_kwargs["peft_config"] == peft_cfg
        assert call_kwargs["lora_tensors"] == dict(weights)

    def test_remove_lora_called_before_add_lora(self):
        """remove_lora must be called before add_lora to prevent stale adapter."""
        worker = MagicMock()
        peft_cfg = {"rank": 32}
        weights = [("lora_A", "tensor_a")]

        _run_update_weights_ipc(worker, weights, peft_config=peft_cfg, base_sync_done=True)

        expected_order = [
            call.remove_lora(VLLM_LORA_INT_ID),
            call.add_lora({"lora_name": VLLM_LORA_NAME, "lora_int_id": VLLM_LORA_INT_ID,
                           "lora_path": VLLM_LORA_PATH, "peft_config": peft_cfg,
                           "lora_tensors": dict(weights)}),
        ]
        actual = [c for c in worker.mock_calls if c[0] in ("remove_lora", "add_lora")]
        assert actual == expected_order

    def test_no_add_lora_without_peft_config(self):
        """Base weight path: add_lora never called, load_weights called instead."""
        worker = MagicMock()
        weights = [("weight.layer0", "tensor_0")]

        _run_update_weights_ipc(worker, weights, peft_config=None, base_sync_done=True)

        worker.add_lora.assert_not_called()
        worker.remove_lora.assert_not_called()
        worker.load_weights.assert_called_once_with(weights)
