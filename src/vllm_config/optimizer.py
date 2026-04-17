"""Core recommendation engine: map hardware + model onto optimal vLLM flags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .hardware import HardwareInfo
from .models import ModelMetadata

# Bytes per parameter for each quantization scheme
_BYTES_PER_PARAM: dict[str, float] = {
    "none": 2.0,   # fp16 / bf16
    "fp8": 1.0,
    "int8": 1.0,
    "awq": 0.5,    # 4-bit
    "gptq": 0.5,   # 4-bit
}

# KV cache stored in fp16 (2 bytes per element)
_KV_BYTES = 2

# Fraction of VRAM reserved for CUDA context, activations, etc.
_VRAM_SAFETY_MARGIN = 0.10

# Minimum fraction of usable VRAM that must remain after model weights for KV cache
_KV_HEADROOM = 0.15


@dataclass
class Recommendation:
    dtype: str
    quantization: Optional[str]
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    estimated_model_vram_gb: float
    estimated_kv_cache_vram_gb: float
    fits: bool
    warnings: list[str] = field(default_factory=list)

    def build_command(
        self,
        model_id: str,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> str:
        parts = [f"vllm serve {model_id}"]
        parts.append(f"--host {host}")
        parts.append(f"--port {port}")
        parts.append(f"--dtype {self.dtype}")
        if self.quantization:
            parts.append(f"--quantization {self.quantization}")
        if self.tensor_parallel_size > 1:
            parts.append(f"--tensor-parallel-size {self.tensor_parallel_size}")
        parts.append(f"--max-model-len {self.max_model_len}")
        parts.append(f"--gpu-memory-utilization {self.gpu_memory_utilization:.2f}")
        return " \\\n  ".join(parts)


def recommend(
    hw: HardwareInfo,
    model: ModelMetadata,
    desired_context: Optional[int] = None,
) -> Recommendation:
    warnings: list[str] = []

    if hw.gpu_count == 0:
        return Recommendation(
            dtype="float16",
            quantization=None,
            tensor_parallel_size=1,
            max_model_len=desired_context or 2048,
            gpu_memory_utilization=0.90,
            estimated_model_vram_gb=0.0,
            estimated_kv_cache_vram_gb=0.0,
            fits=False,
            warnings=["No NVIDIA GPU detected. vLLM requires a CUDA-capable GPU."],
        )

    tensor_parallel = hw.gpu_count
    # Bottleneck is the GPU with least VRAM (all must hold the shard)
    vram_per_gpu = hw.min_vram_gb
    usable_vram_total = vram_per_gpu * tensor_parallel * (1 - _VRAM_SAFETY_MARGIN)

    cc_major, cc_minor = hw.compute_capability
    dtype = "bfloat16" if cc_major >= 8 else "float16"

    # FP8 requires Ada Lovelace (SM 8.9) or Hopper (SM 9.0+)
    fp8_supported = cc_major >= 9 or (cc_major == 8 and cc_minor >= 9)

    # Preference order: full precision → fp8 → awq/gptq
    quant_candidates: list[tuple[Optional[str], str]] = [
        (None, "none"),
    ]
    if fp8_supported:
        quant_candidates.append(("fp8", "fp8"))
    quant_candidates.extend([("awq", "awq"), ("gptq", "gptq")])

    _UNSET = object()
    chosen_quant: object = _UNSET
    model_vram_gb = 0.0

    for quant, key in quant_candidates:
        bpp = _BYTES_PER_PARAM[key]
        model_vram_gb = model.num_params_b * 1e9 * bpp / (1024**3)
        # Keep at least _KV_HEADROOM of usable VRAM free for KV cache
        if model_vram_gb <= usable_vram_total * (1 - _KV_HEADROOM):
            chosen_quant = quant  # may legitimately be None (= full precision)
            break

    if chosen_quant is _UNSET:
        # Still use AWQ as last resort and warn
        chosen_quant = "awq"
        bpp = _BYTES_PER_PARAM["awq"]
        model_vram_gb = model.num_params_b * 1e9 * bpp / (1024**3)
        warnings.append(
            f"Model ({model.num_params_b:.1f}B params) may not fit even with AWQ quantization "
            f"on {usable_vram_total:.1f} GB usable VRAM. "
            "Consider adding more GPUs or using a smaller model."
        )
    elif chosen_quant in ("awq", "gptq"):
        warnings.append(
            f"Using {chosen_quant.upper()} quantization (4-bit) to fit model weights into VRAM. "
            "Slight quality reduction is expected."
        )

    # Available VRAM for KV cache after model weights (split across GPUs for TP)
    remaining_total = usable_vram_total - model_vram_gb
    remaining_per_gpu = remaining_total / tensor_parallel

    # KV cache memory per token per GPU:
    # 2 (K+V) × kv_heads_per_gpu × head_dim × num_layers × bytes_per_element
    kv_heads_per_gpu = max(1, model.num_kv_heads // tensor_parallel)
    kv_bytes_per_token = (
        2 * kv_heads_per_gpu * model.head_dim * model.num_layers * _KV_BYTES
    )
    kv_gb_per_token = kv_bytes_per_token / (1024**3)

    if kv_gb_per_token > 0 and remaining_per_gpu > 0:
        max_ctx_from_vram = int(remaining_per_gpu / kv_gb_per_token)
    else:
        max_ctx_from_vram = 512

    max_ctx_from_vram = max(512, min(max_ctx_from_vram, model.max_position_embeddings))

    if desired_context:
        if desired_context > max_ctx_from_vram:
            warnings.append(
                f"Desired context {desired_context:,} tokens exceeds estimated capacity "
                f"({max_ctx_from_vram:,} tokens). Clamping to {max_ctx_from_vram:,}."
            )
            max_model_len = max_ctx_from_vram
        else:
            max_model_len = desired_context
    else:
        # Cap at 32K by default unless VRAM allows more; avoids unexpected OOM
        max_model_len = min(max_ctx_from_vram, 32768)
        if max_model_len < 512:
            max_model_len = 512
            warnings.append("Very limited VRAM for KV cache — using minimum context of 512 tokens.")

    kv_cache_vram_gb = kv_gb_per_token * max_model_len * tensor_parallel
    total_needed = model_vram_gb + kv_cache_vram_gb
    fits = total_needed <= usable_vram_total

    gpu_memory_utilization = 0.90
    if total_needed / usable_vram_total > 0.95:
        gpu_memory_utilization = 0.85
        warnings.append(
            "VRAM usage is very tight — lowering --gpu-memory-utilization to 0.85 "
            "to reduce out-of-memory risk."
        )

    final_quant: Optional[str] = chosen_quant  # type: ignore[assignment]
    return Recommendation(
        dtype=dtype,
        quantization=final_quant,
        tensor_parallel_size=tensor_parallel,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        estimated_model_vram_gb=model_vram_gb,
        estimated_kv_cache_vram_gb=kv_cache_vram_gb,
        fits=fits,
        warnings=warnings,
    )
