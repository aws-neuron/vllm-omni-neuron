# SPDX-License-Identifier: Apache-2.0
"""NeuronOmniPlatform — vllm-omni platform for AWS Trainium/Inferentia."""

import torch
from vllm_neuron import envs
from vllm_neuron.vllm.platform import NeuronPlatform
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

from vllm_omni_neuron.lite_compat import (
    device_count as lite_device_count,
)
from vllm_omni_neuron.lite_compat import (
    get_hbm_memory_gb,
    get_platform_target,
    is_lite_runtime,
)


class NeuronOmniPlatform(OmniPlatform, NeuronPlatform):
    """vllm-omni platform for AWS Neuron (Trainium/Inferentia) hardware."""

    _omni_enum = OmniPlatformEnum.OOT
    device_type = "neuron"
    device_control_env_var = "NEURON_VISIBLE_DEVICES"

    @staticmethod
    def _expand_device_ranges(devices: str) -> str:
        """Expand ranges so stage configs can keep compact ``"N-M"`` notation.

        vllm-omni only comma-splits NEURON_VISIBLE_DEVICES; lists pass through.
        """
        ids: list[str] = []
        for token in (t.strip() for t in devices.split(",")):
            if not token:
                continue
            if "-" in token:
                start, end = (int(x.strip()) for x in token.split("-", 1))
                ids.extend(str(i) for i in range(start, end + 1))
            else:
                ids.append(token)
        return ",".join(ids)

    @classmethod
    def set_device_control_env_var(cls, devices: str | int | None) -> None:
        import os

        if isinstance(devices, str):
            devices = cls._expand_device_ranges(devices)
        os.environ[cls.device_control_env_var] = str(devices)

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/model_executor/stage_configs"

    @classmethod
    def get_diffusion_worker_cls(cls) -> str:
        return "vllm_omni_neuron.diffusion.worker.diffusion_worker.NeuronDiffusionWorker"

    @classmethod
    def get_diffusion_model_runner_cls(cls) -> str:
        return "vllm_omni_neuron.diffusion.worker.diffusion_model_runner.NeuronDiffusionModelRunner"

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def get_compile_backend(cls) -> str | None:
        if envs.VLLM_NEURON_CPU_MODE:
            return None
        return envs.get_compile_backend_name()

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        return "vllm_omni_neuron.diffusion.attention.backends.sdpa.NeuronSDPABackend"

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if envs.VLLM_NEURON_CPU_MODE:
            return torch.device("cpu")
        # Mirrors CudaOmniPlatform.get_torch_device: accept an optional local_rank,
        # index the device with it when given; return the unindexed device otherwise.
        if local_rank is None:
            return torch.device("neuron")
        return torch.device("neuron", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        if envs.VLLM_NEURON_CPU_MODE:
            return 0
        if is_lite_runtime():
            return lite_device_count()
        return NeuronPlatform.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        return None

    @classmethod
    def synchronize(cls) -> None:
        pass

    @classmethod
    def empty_cache(cls) -> None:
        pass

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        return cls.get_device_total_memory(0)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Per-core HBM in bytes, looked up by platform target.

        Uses the Lite platform API for target detection
        (NEURON_PLATFORM_TARGET_OVERRIDE env var, else NRT auto-detect)
        and the per-target GB table. Falls back to trn2's 24 GiB when
        neither source is available (CPU mode without the override env
        var set, for example).
        """
        try:
            target = get_platform_target()
            gb = get_hbm_memory_gb(target, default=24)
        except (RuntimeError, ImportError):
            gb = 24
        return gb * 1024**3

    @classmethod
    def reset_peak_memory_stats(cls, device_id: int = 0) -> None:
        # No Neuron equivalent of torch.cuda.reset_peak_memory_stats; no-op.
        pass

    @classmethod
    def max_memory_allocated(cls, device_id: int = 0) -> int:
        # No Neuron peak-allocation counter; return 0 so diffusion accounting
        # treats us as having nothing live and moves on.
        return 0

    @classmethod
    def max_memory_reserved(cls, device_id: int = 0) -> int:
        # No Neuron reserved-memory counter; no-op equivalent.
        return 0

    @classmethod
    def set_device(cls, device: torch.device | int | None = None) -> None:
        pass  # Neuron device selection is handled via NEURON_RT_VISIBLE_CORES.

    @classmethod
    def is_initialized(cls) -> bool:
        """Whether the accelerator context is initialized.

        vllm-omni 0.24's ``get_local_device()`` calls this; the base
        ``OmniPlatform`` leaves it as ``None`` (proxied to the ``neuron`` torch
        device module, which does not implement it), so the call raises
        ``TypeError: 'NoneType' object is not callable``. Neuron device
        selection is env-driven (one core per process, no per-process current
        device), so report not-initialized and let ``get_local_device()`` fall
        back to ``LOCAL_RANK``.
        """
        return False

    @classmethod
    def current_device(cls) -> int:
        """Return the current device index.

        Only reached when ``is_initialized()`` is True; provided for API
        completeness. One core per process under Neuron, so index 0.
        """
        return 0

    # dist_backend is read as a plain attribute by init_distributed_environment
    dist_backend: str = envs.get_dist_backend()
