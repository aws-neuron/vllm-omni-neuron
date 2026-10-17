# SPDX-License-Identifier: Apache-2.0
"""Synthesize torch_neuronx.{pyhlo,xla_impl} when libtorch-neuronx-lite is absent.

NKI (``nki.framework.torch_xla``) and ``vllm_neuron`` import
``torch_neuronx.pyhlo`` and ``torch_neuronx.xla_impl`` at module-load time.
Those names ship in the ``libtorch-neuronx-lite`` wheel. On native-backend
dev setups, lite is uninstalled so the editable ``torch_neuronx`` package
(native path) can claim the PrivateUse1 backend slot. That package keeps
protos under ``torch_neuronx.protos.*`` instead and has no ``xla_impl``
package, so the imports fail. ``vllm_neuron/nki/nki_hop.py`` swallows the
ImportError and sets ``compile_nki = None``, surfacing later as a misleading
``TypeError("'NoneType' object is not callable")`` deep in Dynamo.

This module fabricates the missing namespaces in ``sys.modules`` so the
imports resolve. Auto-disables when lite is installed (CI; future dev pods)
or when ``torch_neuronx`` already provides the names natively. Must be
imported before any consumer touches ``vllm_neuron.nki``.
"""

import importlib.util
import sys
import types
from typing import Any, Callable

_installed = False


def _spec_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _make_xla_only_stub(symbol: str) -> Callable[..., Any]:
    def _err(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            f"{symbol} was invoked on the native torch_neuronx path. "
            "This symbol only resolves when libtorch-neuronx-lite is "
            "installed (XLA-routed backend). Install lite to use the XLA "
            "path, or stay on VLLM_NEURON_BACKEND=neuron_native."
        )

    return _err


def _install_pyhlo() -> None:
    if _spec_available("torch_neuronx.pyhlo"):
        return
    try:
        from torch_neuronx.protos import hlo_pb2
        from torch_neuronx.protos.xla import xla_data_pb2
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "torch_neuronx.protos is not available. Install the editable "
            "torch_neuronx package with protobuf stubs, or install "
            "libtorch-neuronx-lite which provides torch_neuronx.pyhlo directly."
        ) from e

    mod = types.ModuleType("torch_neuronx.pyhlo")
    mod.__doc__ = "Synthetic shim from vllm_omni_neuron.bootstrap."
    mod.hlo_pb2 = hlo_pb2
    mod.xla_data_pb2 = xla_data_pb2
    sys.modules["torch_neuronx.pyhlo"] = mod


def _install_xla_impl() -> None:
    if _spec_available("torch_neuronx.xla_impl"):
        return

    pkg = types.ModuleType("torch_neuronx.xla_impl")
    pkg.__doc__ = "Synthetic shim from vllm_omni_neuron.bootstrap."
    pkg.__path__ = []  # mark as package
    sys.modules["torch_neuronx.xla_impl"] = pkg

    base = types.ModuleType("torch_neuronx.xla_impl.base")
    base.xla_hlo_call = _make_xla_only_stub("xla_hlo_call")
    base.xla_call = _make_xla_only_stub("xla_call")
    base.AwsNeuronCustomLoweringType = _make_xla_only_stub("AwsNeuronCustomLoweringType")
    sys.modules["torch_neuronx.xla_impl.base"] = base

    cct = types.ModuleType("torch_neuronx.xla_impl.custom_call_targets")
    # Mirrors nki/framework/jax.py:125 so backend_config matches.
    cct.AwsNeuronNkiKernel = "AwsNeuronCustomNativeKernel"
    sys.modules["torch_neuronx.xla_impl.custom_call_targets"] = cct


def install() -> None:
    global _installed
    if _installed:
        return
    if _spec_available("libtorch_neuronx_lite"):
        _installed = True
        return
    if not _spec_available("torch_neuronx"):
        _installed = True
        return
    _install_pyhlo()
    _install_xla_impl()
    _installed = True


install()
