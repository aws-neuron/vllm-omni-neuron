# SPDX-License-Identifier: Apache-2.0
"""
vLLM Omni Neuron Environment Profiles

Per-model environment written to os.environ. Each field name IS the environment variable
name, and each is either forced (always overwrite) or set_default (write only when
unset) per Wan22EnvProfile.SET_DEFAULT.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import ClassVar, Protocol, runtime_checkable


@dataclass(frozen=True)
class EnvWrites:
    # assign = forced overwrite, setdefault = no-op if we have set other value
    assign: Mapping[str, str] = field(default_factory=dict)
    setdefault: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class EnvProfile(Protocol):
    def to_env(self) -> EnvWrites: ...


def merge(*sources: EnvWrites) -> EnvWrites:
    assign: dict[str, str] = {}
    setdefault: dict[str, str] = {}
    for source in sources:
        assign.update(source.assign)
        setdefault.update(source.setdefault)
    return EnvWrites(assign=assign, setdefault=setdefault)


def thread_limits(tp_size: int) -> EnvWrites:
    """Deferred OMP/MKL thread caps; torchrun's own value wins when it sets one."""
    if hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count() or 1
    threads = str(max(1, cpus // tp_size))
    return EnvWrites(setdefault={"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads})


def apply(*sources: EnvProfile | EnvWrites) -> None:
    """Merge every source and write it to os.environ."""
    writes = merge(*(s.to_env() if isinstance(s, EnvProfile) else s for s in sources))

    os.environ.update(writes.assign)
    for name, value in writes.setdefault.items():
        os.environ.setdefault(name, value)


@dataclass(frozen=True)
class Wan22EnvProfile:
    TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION: str | None = "1"
    VLLM_SLEEP_WHEN_IDLE: str | None = "1"
    VLLM_NEURON_COMPILATION_TIMEOUT: str | None = "1800"
    NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE: str | None = "167772160"
    # Must match --hbm-scratchpad-page-size in NEURON_CC_FLAGS below; change both together.
    NEURON_SCRATCHPAD_PAGE_SIZE: str | None = "2048"
    NEURON_RT_DBG_CC_DMA_PACKET_SIZE: str | None = "2048"
    NEURON_LOGICAL_NC_CONFIG: str | None = "2"
    NEURON_PLATFORM_TARGET_OVERRIDE: str | None = None
    NEURON_CC_FLAGS: str | None = "-O1 --hbm-scratchpad-page-size=2048"
    VLLM_NEURON_BACKEND: str | None = "neuron_native"
    TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS: str | None = "True"
    # CWD-relative, so callers whose working dir is not the plugin root override it.
    TORCH_NEURONX_DEBUG_DIR: str | None = "./compile_dir"

    extra: EnvWrites = field(default_factory=EnvWrites)

    # Pre-defined set for setDefault field
    SET_DEFAULT: ClassVar[frozenset[str]] = frozenset(
        {
            "NEURON_PLATFORM_TARGET_OVERRIDE",
            "NEURON_CC_FLAGS",
            "VLLM_NEURON_BACKEND",
        }
    )

    def to_env(self) -> EnvWrites:
        """Split the fields into forced and SET_DEFAULT writes; a None field is skipped."""
        assign: dict[str, str] = {}
        setdefault: dict[str, str] = {}
        for spec in fields(self):
            if spec.name == "extra":
                continue
            value = getattr(self, spec.name)
            if value is None:
                continue
            target = setdefault if spec.name in self.SET_DEFAULT else assign
            target[spec.name] = value
        return merge(EnvWrites(assign=assign, setdefault=setdefault), self.extra)


WAN22_T2V = Wan22EnvProfile()
WAN22_I2V = Wan22EnvProfile()
