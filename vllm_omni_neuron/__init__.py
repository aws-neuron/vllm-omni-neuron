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


def register_neuron_pipelines() -> None:
    """vllm_omni.general_plugins entry point — registers Neuron pipelines early,
    before platform resolution, avoiding circular imports."""
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
