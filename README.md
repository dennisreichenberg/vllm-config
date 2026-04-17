# vllm-config

**Hardware-aware vLLM configuration optimizer.** Detects your GPUs and system memory, fetches model architecture from HuggingFace, and outputs a ready-to-paste `vllm serve` command with optimal parameters.

No more guessing `--tensor-parallel-size` or `--max-model-len` — just point it at your model and go.

## Install

```bash
pip install vllm-config
```

## Usage

```bash
vllm-config <model-id> [OPTIONS]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--context-length`, `-c` | auto | Desired context window in tokens |
| `--host` | `0.0.0.0` | Host for the generated `vllm serve` command |
| `--port`, `-p` | `8000` | Port for the generated `vllm serve` command |
| `--hf-token` | `$HF_TOKEN` | HuggingFace API token (required for gated models) |

## Examples

### Llama 3.1 8B on a single 24 GB GPU

```bash
vllm-config meta-llama/Llama-3.1-8B-Instruct
```

**Output:**
```
┌─ Detected Hardware ─────────────────────────────────────────────┐
│ GPU 0      NVIDIA RTX 4090  |  24.0 GB VRAM  |  Compute 8.9    │
│ System RAM 64.0 GB                                              │
│ CPU Cores  24                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─ Recommended Configuration ──────────────────────────────────┐
│ --dtype                    bfloat16                          │
│ --quantization             none (full precision)             │
│ --tensor-parallel-size     1                                 │
│ --max-model-len            32,768                            │
│ --gpu-memory-utilization   0.90                              │
│ Est. model VRAM            ~14.0 GB                          │
│ Est. KV cache VRAM         ~3.8 GB                           │
│ Compatibility              Fits in VRAM                      │
└──────────────────────────────────────────────────────────────┘

Ready-to-paste launch command:

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

---

### Mistral 7B with custom context length

```bash
vllm-config mistralai/Mistral-7B-Instruct-v0.3 --context-length 16384
```

---

### Qwen 2.5 72B on 4× A100 80 GB

```bash
vllm-config Qwen/Qwen2.5-72B-Instruct
```

Expected output includes `--tensor-parallel-size 4` and recommends full BF16 precision.

---

### Gated model (requires HuggingFace token)

```bash
export HF_TOKEN=hf_...
vllm-config meta-llama/Llama-3.1-70B-Instruct
# or
vllm-config meta-llama/Llama-3.1-70B-Instruct --hf-token hf_...
```

---

## How it works

1. **Hardware detection** — uses `pynvml` to read each GPU's name, VRAM, and CUDA compute capability. Falls back gracefully when no NVIDIA GPU is present.
2. **Model metadata** — fetches `config.json` from HuggingFace Hub (no model weights downloaded). Extracts hidden size, layer count, KV head count, and maximum native context length.
3. **Parameter estimation** — derives approximate parameter count from architecture dimensions using a standard LLaMA-style transformer formula.
4. **Recommendation engine** — selects the least-compressed dtype/quantization that fits within available VRAM (BF16/FP16 → FP8 → AWQ/GPTQ), picks `--tensor-parallel-size` equal to GPU count, and calculates the maximum feasible context window from remaining KV cache memory.

### Quantization selection logic

| GPU VRAM pressure | Recommended quantization |
|---|---|
| Model fits with ≥ 15% headroom | None (full precision) |
| Tight fit on SM 8.9+ (Ada/Hopper) | FP8 |
| Needs compression | AWQ (4-bit) |
| Emergency fallback | GPTQ (4-bit) |

### dtype selection

| GPU compute capability | dtype |
|---|---|
| SM 8.0+ (Ampere, Ada, Hopper) | `bfloat16` |
| SM 7.x and older (Volta, Turing) | `float16` |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (for hardware detection; estimation still works without)
- Internet access to fetch model config from HuggingFace

## License

MIT
