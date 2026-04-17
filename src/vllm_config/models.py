"""Fetch and parse model metadata from HuggingFace Hub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class ModelMetadata:
    model_id: str
    num_params_b: float
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    hidden_size: int
    max_position_embeddings: int
    architecture: str
    vocab_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def fetch_model_metadata(model_id: str, hf_token: Optional[str] = None) -> ModelMetadata:
    headers = {"Accept": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url, headers=headers, follow_redirects=True)
        resp.raise_for_status()
        config = resp.json()

    return _parse_config(model_id, config)


def _parse_config(model_id: str, config: dict) -> ModelMetadata:
    archs = config.get("architectures") or []
    arch = archs[0] if archs else "Unknown"

    hidden_size = config.get("hidden_size") or config.get("d_model") or 4096
    num_layers = (
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("num_layers")
        or 32
    )
    num_heads = config.get("num_attention_heads") or config.get("n_head") or 32
    num_kv_heads = config.get("num_key_value_heads") or num_heads
    max_pos = (
        config.get("max_position_embeddings")
        or config.get("max_seq_len")
        or config.get("seq_length")
        or 4096
    )
    vocab_size = config.get("vocab_size") or 32000

    num_params_b = _estimate_params(hidden_size, num_layers, num_heads, num_kv_heads, vocab_size)

    return ModelMetadata(
        model_id=model_id,
        num_params_b=num_params_b,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        max_position_embeddings=max_pos,
        architecture=arch,
        vocab_size=vocab_size,
    )


def _estimate_params(
    hidden: int,
    layers: int,
    n_heads: int,
    kv_heads: int,
    vocab: int,
) -> float:
    """Estimate parameter count (billions) from architecture dimensions.

    Uses typical LLaMA-style transformer layout (SwiGLU FFN, GQA).
    """
    head_dim = hidden // n_heads
    # Attention: Q + K + V + O projections
    attn = hidden * hidden + 2 * kv_heads * head_dim * hidden + hidden * hidden
    # SwiGLU FFN: gate + up + down, intermediate ≈ 8/3 * hidden
    intermediate = int(8 / 3 * hidden)
    ffn = hidden * intermediate * 2 + intermediate * hidden
    per_layer = attn + ffn
    # Embedding + LM head (usually tied, count once)
    embeddings = vocab * hidden
    total = layers * per_layer + embeddings
    return total / 1e9
