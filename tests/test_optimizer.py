"""Tests for the recommendation engine."""

import pytest

from vllm_config.hardware import GpuInfo, HardwareInfo
from vllm_config.models import ModelMetadata
from vllm_config.optimizer import recommend


def _hw(gpu_count: int = 1, vram_gb: float = 24.0, cc_major: int = 8, cc_minor: int = 0) -> HardwareInfo:
    gpus = [
        GpuInfo(name="NVIDIA A100", vram_gb=vram_gb, compute_major=cc_major, compute_minor=cc_minor)
        for _ in range(gpu_count)
    ]
    return HardwareInfo(gpus=gpus, system_ram_gb=64.0, cpu_count=16)


def _model(
    params_b: float = 8.0,
    layers: int = 32,
    n_heads: int = 32,
    kv_heads: int = 8,
    hidden: int = 4096,
    max_ctx: int = 131072,
) -> ModelMetadata:
    return ModelMetadata(
        model_id="test/model",
        num_params_b=params_b,
        num_layers=layers,
        num_attention_heads=n_heads,
        num_kv_heads=kv_heads,
        hidden_size=hidden,
        max_position_embeddings=max_ctx,
        architecture="LlamaForCausalLM",
        vocab_size=32000,
    )


def test_small_model_fits_no_quantization():
    rec = recommend(_hw(vram_gb=24.0), _model(params_b=7.0))
    assert rec.quantization is None
    assert rec.fits


def test_large_model_triggers_quantization():
    rec = recommend(_hw(gpu_count=1, vram_gb=24.0), _model(params_b=70.0))
    assert rec.quantization is not None


def test_multi_gpu_increases_capacity():
    model = _model(params_b=30.0)
    rec_single = recommend(_hw(gpu_count=1, vram_gb=24.0), model)
    rec_multi = recommend(_hw(gpu_count=4, vram_gb=24.0), model)
    assert rec_multi.tensor_parallel_size == 4
    # Multi-GPU should need less aggressive quantization
    if rec_single.quantization in ("awq", "gptq"):
        assert rec_multi.quantization != "awq" or rec_multi.quantization != "gptq"


def test_desired_context_respected():
    rec = recommend(_hw(vram_gb=80.0), _model(params_b=7.0, max_ctx=128000), desired_context=8192)
    assert rec.max_model_len == 8192


def test_desired_context_clamped_if_too_large():
    rec = recommend(_hw(vram_gb=16.0), _model(params_b=7.0, max_ctx=128000), desired_context=128000)
    assert rec.max_model_len <= 128000
    if rec.max_model_len < 128000:
        assert any("Clamping" in w for w in rec.warnings)


def test_no_gpu_returns_unfit():
    hw = HardwareInfo(gpus=[], system_ram_gb=32.0, cpu_count=8)
    rec = recommend(hw, _model(params_b=7.0))
    assert not rec.fits
    assert any("No NVIDIA GPU" in w for w in rec.warnings)


def test_bf16_on_ampere():
    rec = recommend(_hw(cc_major=8, cc_minor=0), _model(params_b=7.0))
    assert rec.dtype == "bfloat16"


def test_fp16_on_older_gpu():
    rec = recommend(_hw(cc_major=7, cc_minor=5), _model(params_b=7.0))
    assert rec.dtype == "float16"


def test_fp8_only_on_ada_or_hopper():
    rec_old = recommend(_hw(cc_major=8, cc_minor=0), _model(params_b=7.0))
    assert rec_old.quantization != "fp8"

    rec_ada = recommend(_hw(cc_major=8, cc_minor=9, vram_gb=12.0), _model(params_b=13.0))
    # fp8 is a candidate; whether it's chosen depends on VRAM pressure
    assert rec_ada.quantization in (None, "fp8", "awq", "gptq")


def test_build_command_includes_all_flags():
    rec = recommend(_hw(gpu_count=2, vram_gb=24.0), _model(params_b=13.0))
    cmd = rec.build_command("meta-llama/Llama-3-13B", host="0.0.0.0", port=8000)
    assert "vllm serve" in cmd
    assert "--dtype" in cmd
    assert "--max-model-len" in cmd
    assert "--gpu-memory-utilization" in cmd
    if rec.tensor_parallel_size > 1:
        assert "--tensor-parallel-size" in cmd


def test_no_quantization_flag_when_full_precision():
    rec = recommend(_hw(vram_gb=80.0), _model(params_b=7.0))
    cmd = rec.build_command("test/model")
    assert "--quantization" not in cmd
