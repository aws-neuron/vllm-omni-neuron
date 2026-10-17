# SPDX-License-Identifier: Apache-2.0
"""Wan attention compilation settings for Lite's NKI bridge."""

import hashlib
from functools import wraps
from importlib import import_module

_WAN_ATTENTION_KERNELS = {
    (
        "vllm_omni_neuron.kernels.nkilib.experimental.attention.ring_attention_const_max_fwd",
        "ring_attention_const_max_fwd",
    ),
    (
        "vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer",
        "_wan_attention_kernel",
    ),
}
_CACHE_POLICY = "wan-trn3-attention:non_ssa=1:address_rotation=enable:v1"


def _uses_wan_attention_policy(func, target: str) -> bool:
    return (
        target == "trn3"
        and (
            getattr(func, "__module__", None),
            getattr(func, "__name__", None),
        )
        in _WAN_ATTENTION_KERNELS
    )


def install_wan_attention_compile_policy() -> None:
    """Restore the staging buffer policy for the two Wan attention kernels.

    NKI 0.7 defaults to SSA legalization and automatic address rotation. In Wan
    these reuse scratch buffers across EXP operations and serialize attention.
    The previous non-SSA legalization and enabled address rotation retain the
    independent buffers. Kernel arithmetic and the const-max source stay intact.

    Lite currently discards CompileKernel options in wrap_nki(), so configure
    its private compilation bridge until it can carry per-kernel options.
    Its persistent cache also needs the policy in its key: changing constructor
    settings alone would reuse entries compiled with the previous defaults.
    Torch's AOT cache precedes NKI lowering, so its process-wide cache tag must
    also include the policy. This invalidates prior AOT entries once, including
    TRN2 entries; TRN2's NKI options and NKI cache keys remain unchanged.
    """
    try:
        bridge = import_module("libtorch_neuronx_lite.nki.nki_compile")
    except ModuleNotFoundError as exc:
        if exc.name and (
            exc.name == "libtorch_neuronx_lite" or exc.name.startswith("libtorch_neuronx_lite.nki")
        ):
            return  # Older SDKs use the previous bridge and buffer defaults.
        raise

    original_kernel = bridge.CompileKernel
    if getattr(original_kernel, "_wan_attention_compile_policy", False):
        return
    fields = getattr(original_kernel, "__dataclass_fields__", {})
    if not {"non_ssa_legalization", "address_rotation"} <= fields.keys():
        return

    cache = import_module("libtorch_neuronx_lite.nki.nki_cache")
    address_rotation = import_module("nki.framework.kernel").AddressRotation
    original_key = cache.create_nki_cache_key
    torch_cache = import_module("torch.compiler.config")
    previous_tag = torch_cache.cache_key_tag
    if _CACHE_POLICY not in previous_tag.split("|"):
        torch_cache.cache_key_tag = "|".join(filter(None, (previous_tag, _CACHE_POLICY)))

    def compile_kernel(*args, **kwargs):
        if _uses_wan_attention_policy(kwargs.get("func"), kwargs.get("target")):
            kwargs.update(non_ssa_legalization=True, address_rotation=address_rotation.ENABLE)
        return original_kernel(*args, **kwargs)

    @wraps(original_key)
    def create_cache_key(func, args, grid, frontend_name):
        key = original_key(func, args, grid, frontend_name=frontend_name)
        if key is not None and _uses_wan_attention_policy(func, cache.get_platform_target()):
            return hashlib.sha256(f"{key}|{_CACHE_POLICY}".encode()).hexdigest()[:32]
        return key

    # Install once, before registering Wan's operators. Neither wrapper mutates
    # shared options while compiling, so concurrent compilations remain isolated.
    compile_kernel._wan_attention_compile_policy = True
    bridge.CompileKernel = compile_kernel
    cache.create_nki_cache_key = create_cache_key
