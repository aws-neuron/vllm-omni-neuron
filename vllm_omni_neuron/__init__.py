# SPDX-License-Identifier: Apache-2.0
"""vllm_omni_neuron — Neuron platform plugin for vllm-omni."""

from . import bootstrap as _bootstrap  # noqa: F401  isort: skip  must precede vllm_neuron imports

import warnings

from vllm_neuron import _is_cpu_mode, _is_neuron_dev


def _register_pipelines() -> None:
    """Register all Neuron-specific diffusion pipelines into the vllm-omni registry.

    Each module under vllm_omni_neuron.diffusion.models may expose a
    PIPELINE_REGISTRY list of dicts with keys: model_arch, class_name, and
    optionally post_process_func_name.
    """
    import importlib
    import pkgutil

    from vllm_omni.diffusion.registry import register_diffusion_model

    import vllm_omni_neuron.diffusion.models as _models_pkg

    for module_info in pkgutil.iter_modules(_models_pkg.__path__, _models_pkg.__name__ + "."):
        module = importlib.import_module(module_info.name)
        for entry in getattr(module, "PIPELINE_REGISTRY", []):
            register_diffusion_model(
                model_arch=entry["model_arch"],
                module_name=module_info.name,
                class_name=entry["class_name"],
                **{k: v for k, v in entry.items() if k not in ("model_arch", "class_name")},
            )


_topologies_registered = False
_topologies_registering = False


def register_pipeline_topologies() -> None:
    """Register stage topologies for checkpoints vllm-omni does not know.

    Without a topology, config resolution falls through to the generic diffusion
    path, which never reads the deploy YAML: the pipeline then loads unsharded on
    a single core and dies filling that core's HBM. The general-plugins hook runs
    *after* the engine has resolved stage configs, so this must be called from
    platform-plugin activation, which precedes it. Idempotent.
    """
    global _topologies_registered, _topologies_registering
    # Importing the registry can re-enter platform resolution, hence a separate
    # in-progress guard: it terminates that recursion, while the success flag is
    # what makes a later caller (the general-plugins hook) retry after a failure.
    if _topologies_registered or _topologies_registering:
        return
    _topologies_registering = True
    try:
        from vllm_omni.config.pipeline_registry import register_pipeline

        from vllm_omni_neuron.diffusion.pipeline_topologies import WAN2_2_I2V_PIPELINE

        register_pipeline(WAN2_2_I2V_PIPELINE)
        _topologies_registered = True
    finally:
        _topologies_registering = False


def register_neuron_pipelines() -> None:
    """vllm_omni.general_plugins entry point — registers Neuron pipelines early,
    before platform resolution, avoiding circular imports."""
    register_pipeline_topologies()
    _register_pipelines()


def neuron_omni_platform_plugin() -> str | None:
    """Entry point for vllm_omni.platform_plugins.

    Returns the dotted class path for NeuronOmniPlatform when running on
    Neuron hardware or in CPU mode, otherwise returns None.
    """
    if not _is_cpu_mode() and not _is_neuron_dev():
        warnings.warn(
            "No Neuron devices found and VLLM_NEURON_CPU_MODE is not set. "
            "Skipping Neuron plugin registration.",
            category=UserWarning,
            stacklevel=2,
        )
        return None
    # Earliest hook that runs in the process that resolves stage configs. A
    # failure here must not cost us the platform itself (the caller treats any
    # exception as "plugin not applicable"), so it only degrades to the
    # general-plugins timing.
    try:
        register_pipeline_topologies()
    except Exception:  # noqa: BLE001 - diagnostic only; registration retries later
        warnings.warn(
            "Could not register Neuron pipeline topologies at platform activation; "
            "deploy-YAML topologies may be ignored for out-of-tree checkpoints.",
            category=UserWarning,
            stacklevel=2,
        )
    return "vllm_omni_neuron.platform.NeuronOmniPlatform"


def _repair_neuron_device_module() -> None:
    """Repair the Torch ``neuron`` device module at import time.

    ``ensure_neuron_amp_device_module`` attaches the ``get_amp_supported_dtype``
    func that vllm-omni 0.24's ``rope.py`` requires at import (it decorates rotary
    classes with ``@torch.amp.autocast("neuron")``). This is needed under BOTH the
    Lite runtime AND CPU mode (both import that rope module), so it is not gated on
    Lite. The shim is a numeric no-op.

    ``ensure_current_device_index`` is Lite-specific, so it stays gated on the Lite
    env. Applied at package import (not only in the worker) so mp-spawned test
    workers, which re-import this package but never run the worker's init path, are
    covered too.
    """
    from vllm_omni_neuron.lite_compat import (
        ensure_current_device_index,
        ensure_neuron_amp_device_module,
        is_lite_runtime,
    )

    ensure_neuron_amp_device_module()
    if is_lite_runtime():
        ensure_current_device_index()


_repair_neuron_device_module()

from vllm_omni_neuron import log_dedup as _log_dedup  # noqa: E402  isort: skip

_log_dedup.install()
