"""Hardware detection: GPUs, VRAM, system RAM, CPU."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class GpuInfo:
    name: str
    vram_gb: float
    compute_major: int
    compute_minor: int


@dataclass
class HardwareInfo:
    gpus: list[GpuInfo]
    system_ram_gb: float
    cpu_count: int

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def total_vram_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    @property
    def min_vram_gb(self) -> float:
        return min((g.vram_gb for g in self.gpus), default=0.0)

    @property
    def gpu_name(self) -> str:
        return self.gpus[0].name if self.gpus else "No GPU"

    @property
    def compute_capability(self) -> tuple[int, int]:
        if not self.gpus:
            return (0, 0)
        return (self.gpus[0].compute_major, self.gpus[0].compute_minor)


def detect_hardware() -> HardwareInfo:
    gpus = _detect_gpus()
    ram_gb = _detect_ram()
    cpu_count = os.cpu_count() or 1
    return HardwareInfo(gpus=gpus, system_ram_gb=ram_gb, cpu_count=cpu_count)


def _detect_gpus() -> list[GpuInfo]:
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = mem.total / (1024**3)
            cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            gpus.append(
                GpuInfo(
                    name=name,
                    vram_gb=vram_gb,
                    compute_major=cc[0],
                    compute_minor=cc[1],
                )
            )
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        return []


def _detect_ram() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except Exception:
        pass
    return 0.0
