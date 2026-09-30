# SPDX-License-Identifier: Apache-2.0
"""NeuronDiffusionWorker — DiffusionWorker override for AWS Neuron hardware."""

from __future__ import annotations

import os

from vllm.config import CompilationConfig, VllmConfig
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_neuron import envs
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker
from vllm_omni.diffusion.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.platforms import current_omni_platform

from vllm_omni_neuron.backend import uses_native_compilation_backend
from vllm_omni_neuron.diffusion.distributed.parallel_state import override_groups_with_physical_mesh
from vllm_omni_neuron.lite_compat import initialize as initialize_lite
from vllm_omni_neuron.lite_compat import is_lite_runtime


def _limit_lite_worker_threads() -> None:
    """Keep one Lite worker per NeuronCore from oversubscribing the host."""
    if not is_lite_runtime():
        return

    # Frontend thread settings are inherited by every worker. Match
    # vllm_neuron's MPExecutor worker setup before Lite initializes its pools.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import torch

    torch.set_num_threads(1)


class NeuronDiffusionWorker(DiffusionWorker, NeuronWorker):
    """DiffusionWorker subclass for AWS Neuron (Trainium/Inferentia).

    Inherits runtime init methods (_get_visible_devices, _set_efa_affinity,
    _set_cpu_affinity, _patch_in_same_node_as_function) from NeuronWorker,
    overriding where diffusion-specific behavior differs.

    TODO: Ideally, we should not have to do this.
    We should have a common set of APIs for hardware visibility
    and initialization in a separate hardware utils.py in the text plugin or a common package
    that can be reused by all packages' and frameworks' workers.
    """

    def init_device(self) -> None:
        _limit_lite_worker_threads()

        vllm_config = VllmConfig(compilation_config=CompilationConfig())
        vllm_config.parallel_config.world_size = self.od_config.parallel_config.world_size
        vllm_config.parallel_config.nnodes = getattr(self.od_config, "nnodes", 1)
        vllm_config.parallel_config.tensor_parallel_size = (
            self.od_config.parallel_config.tensor_parallel_size
        )
        vllm_config.parallel_config.data_parallel_size = (
            self.od_config.parallel_config.data_parallel_size
        )
        self.vllm_config = vllm_config

        native_compilation = uses_native_compilation_backend()
        lite_runtime = is_lite_runtime()

        if not envs.VLLM_NEURON_CPU_MODE:
            if not native_compilation:
                if not lite_runtime:
                    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "1"
                import torch
                import vllm_neuron  # noqa: F401
                from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap

                visible_devices = self._get_visible_devices()
                if os.environ.get("NEURON_SKIP_EFA_AFFINITY", "0") == "0":
                    self._set_efa_affinity(visible_devices)
                self._set_cpu_affinity()
                os.environ["NEURON_RT_VISIBLE_CORES"] = str(visible_devices[self.local_rank])
                os.environ["NEURON_RT_MAP_HBM"] = "1"
            else:
                # Set per-worker device visibility
                visible_devices = self._get_visible_devices()
                from vllm_neuron.vllm.platform import NeuronPlatform

                NeuronPlatform.set_device_count(len(visible_devices))
                os.environ["NEURON_RT_VISIBLE_CORES"] = str(visible_devices[self.local_rank])
                os.environ.pop("NEURON_LIBRARY_PATH", None)
                os.environ.pop("NEURON_RT_ROOT_COMM_ID", None)
                # Also sets TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE; see initialize().
                initialize_lite()

        self.device = current_omni_platform.get_torch_device(self.local_rank)
        current_omni_platform.set_device(self.device)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.od_config.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.od_config.parallel_config.world_size)

        self._patch_in_same_node_as_function()

        init_distributed_environment(
            world_size=self.od_config.parallel_config.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            backend=envs.get_dist_backend(),
        )

        if not envs.VLLM_NEURON_CPU_MODE and not native_compilation:
            rendezvous_ccom_bootstrap()
            import torch

            runtime = torch.classes.neuron.Runtime()
            runtime.initialize()

        if not envs.VLLM_NEURON_CPU_MODE and (not native_compilation or lite_runtime):
            # torch_neuronx and Lite both patch F.gelu with a wrapper around the C
            # builtin, which Dynamo cannot trace in fullgraph mode. Restore with the
            # aten op, which is dispatcher-aware and accepts the same kwargs. Runs
            # after whichever backend installed its wrapper (Lite patches on import,
            # via initialize_lite() above; torch_neuronx in runtime.initialize()).
            import torch

            torch.nn.functional.gelu = torch.ops.aten.gelu.default

        parallel_config = self.od_config.parallel_config
        initialize_model_parallel(
            data_parallel_size=parallel_config.data_parallel_size,
            cfg_parallel_size=parallel_config.cfg_parallel_size,
            sequence_parallel_size=parallel_config.sequence_parallel_size,
            ulysses_degree=parallel_config.ulysses_degree,
            ring_degree=parallel_config.ring_degree,
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        )
        # On the Trn2 64-core 8x8 fabric, rebuild TP/CP/CFG onto the physical-mesh
        # layout so the live groups match the replica_groups registered for
        # compilation (no-op on Trn3 and other non-mesh layouts). Without this,
        # Trn2 cfg-parallel collectives route into the wrong slots and corrupt the
        # output. Resolve the CP degree the same way initialize_model_parallel does
        # (it defaults to ulysses*ring when sequence_parallel_size is unset). Upstream
        # names this "sequence parallelism"; we use it as context parallelism (CP).
        cp_size = parallel_config.sequence_parallel_size or (
            (parallel_config.ulysses_degree or 1) * (parallel_config.ring_degree or 1)
        )
        override_groups_with_physical_mesh(
            tp_size=parallel_config.tensor_parallel_size,
            cp_size=cp_size,
            cfg_size=parallel_config.cfg_parallel_size or 1,
        )
        init_workspace_manager(self.device)

    def _create_profiler(self):
        """Create the Neuron runtime profiler for diffusion workers."""
        profiler_config = self.od_config.profiler_config
        if getattr(profiler_config, "profiler", None) != "cuda":
            return super()._create_profiler()

        from vllm_neuron.vllm.worker.neuron_profiler import (
            NeuronProfiler,
            NeuronProfilerConfig,
        )

        neuron_config = NeuronProfilerConfig(
            self.od_config.additional_config.get("neuron_profiler")
        )
        if neuron_config.neuron_cores is None:
            if self.local_rank != 0:
                return None
        elif self.local_rank not in neuron_config.neuron_cores:
            return None
        return NeuronProfiler(profiler_config, neuron_config)

    def init_lora_manager(self) -> None:
        """No-op: LoRA not supported on Neuron."""
        pass

    def sleep(self, level: int = 1) -> bool:
        raise NotImplementedError("sleep() is not supported on Neuron hardware")

    def wake_up(self, tags: list[str] | None = None) -> bool:
        raise NotImplementedError("wake_up() is not supported on Neuron hardware")
