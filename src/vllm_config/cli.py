"""CLI entry point for vllm-config."""

from __future__ import annotations

from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .hardware import HardwareInfo, detect_hardware
from .models import ModelMetadata, fetch_model_metadata
from .optimizer import Recommendation, recommend

app = typer.Typer(
    name="vllm-config",
    help="Analyze your hardware and generate optimal vLLM startup parameters.",
    add_completion=False,
)
console = Console()
err = Console(stderr=True)


@app.command()
def run(
    model: str = typer.Argument(
        ...,
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B-Instruct).",
    ),
    context_length: Optional[int] = typer.Option(
        None,
        "--context-length",
        "-c",
        help="Desired context window in tokens. Auto-calculated if omitted.",
        min=64,
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Host for the vllm serve command."),
    port: int = typer.Option(8000, "--port", "-p", help="Port for the vllm serve command."),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf-token",
        envvar="HF_TOKEN",
        help="HuggingFace API token (for gated/private models).",
    ),
) -> None:
    """Generate an optimal vllm serve command for your hardware and model.

    \b
    Examples:
      vllm-config meta-llama/Llama-3.1-8B-Instruct
      vllm-config mistralai/Mistral-7B-Instruct-v0.3 --context-length 16384
      vllm-config Qwen/Qwen2.5-72B-Instruct -c 8192
      vllm-config meta-llama/Llama-3.1-70B-Instruct --hf-token hf_...
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]vllm-config[/bold cyan] — Hardware-aware vLLM configuration optimizer",
            border_style="cyan",
        )
    )
    console.print()

    with console.status("[bold]Detecting hardware...[/bold]"):
        hw = detect_hardware()

    _print_hardware_table(hw)

    with console.status(f"[bold]Fetching model config for [cyan]{model}[/cyan]...[/bold]"):
        try:
            meta = fetch_model_metadata(model, hf_token=hf_token)
        except Exception as exc:
            err.print(f"\n[red]Failed to fetch model config:[/red] {exc}")
            err.print(
                "[dim]Verify the model ID is correct. "
                "For gated models, set --hf-token or HF_TOKEN.[/dim]"
            )
            raise typer.Exit(1)

    _print_model_table(meta)

    rec = recommend(hw, meta, desired_context=context_length)
    _print_recommendation(rec, model, host, port)


def _print_hardware_table(hw: HardwareInfo) -> None:
    table = Table(title="Detected Hardware", box=box.ROUNDED, border_style="dim")
    table.add_column("Component", style="bold white")
    table.add_column("Details", style="cyan")

    if hw.gpus:
        for i, gpu in enumerate(hw.gpus):
            table.add_row(
                f"GPU {i}",
                f"{gpu.name}  |  {gpu.vram_gb:.1f} GB VRAM  |  "
                f"Compute {gpu.compute_major}.{gpu.compute_minor}",
            )
        if hw.gpu_count > 1:
            table.add_row("Total GPU VRAM", f"{hw.total_vram_gb:.1f} GB across {hw.gpu_count} GPUs")
    else:
        table.add_row("GPU", "[red]None detected (pynvml unavailable or no NVIDIA GPU)[/red]")

    table.add_row("System RAM", f"{hw.system_ram_gb:.1f} GB")
    table.add_row("CPU Cores", str(hw.cpu_count))

    console.print(table)
    console.print()


def _print_model_table(meta: ModelMetadata) -> None:
    table = Table(title=f"Model: {meta.model_id}", box=box.ROUNDED, border_style="dim")
    table.add_column("Property", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Architecture", meta.architecture)
    table.add_row("Parameters (estimated)", f"~{meta.num_params_b:.1f}B")
    table.add_row("Hidden Size", str(meta.hidden_size))
    table.add_row("Layers", str(meta.num_layers))
    table.add_row("Attention Heads", str(meta.num_attention_heads))
    table.add_row("KV Heads (GQA)", str(meta.num_kv_heads))
    table.add_row("Head Dim", str(meta.head_dim))
    table.add_row("Max Native Context", f"{meta.max_position_embeddings:,} tokens")

    console.print(table)
    console.print()


def _print_recommendation(
    rec: Recommendation,
    model_id: str,
    host: str,
    port: int,
) -> None:
    if rec.warnings:
        for w in rec.warnings:
            console.print(f"[yellow]⚠[/yellow]  {w}")
        console.print()

    status_color = "green" if rec.fits else "yellow"
    status_text = "Fits in VRAM" if rec.fits else "Tight — monitor VRAM usage"

    table = Table(title="Recommended Configuration", box=box.ROUNDED, border_style="cyan")
    table.add_column("Parameter", style="bold white")
    table.add_column("Value", style="cyan")
    table.add_column("Notes", style="dim")

    table.add_row("--dtype", rec.dtype, "Weight data type")
    quant_str = rec.quantization or "none (full precision)"
    table.add_row("--quantization", quant_str, "Quantization scheme")
    table.add_row(
        "--tensor-parallel-size",
        str(rec.tensor_parallel_size),
        f"Model sharded across {rec.tensor_parallel_size} GPU(s)",
    )
    table.add_row("--max-model-len", f"{rec.max_model_len:,}", "Context window tokens")
    table.add_row(
        "--gpu-memory-utilization",
        f"{rec.gpu_memory_utilization:.2f}",
        "Fraction of VRAM allocated to vLLM",
    )
    table.add_row("", "", "")
    table.add_row(
        "Est. model VRAM",
        f"~{rec.estimated_model_vram_gb:.1f} GB",
        "Weight memory (total across GPUs)",
    )
    table.add_row(
        "Est. KV cache VRAM",
        f"~{rec.estimated_kv_cache_vram_gb:.1f} GB",
        "KV cache at max context",
    )
    table.add_row(
        "Compatibility",
        f"[{status_color}]{status_text}[/{status_color}]",
        "",
    )

    console.print(table)
    console.print()

    cmd = rec.build_command(model_id, host, port)
    console.print("[bold]Ready-to-paste launch command:[/bold]")
    console.print()
    console.print(Syntax(cmd, "bash", theme="monokai", word_wrap=True))
    console.print()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
