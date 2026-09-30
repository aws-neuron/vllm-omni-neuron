# SPDX-License-Identifier: Apache-2.0
"""ComfyUI FP8-scaled Wan2.2 checkpoint -> Neuron parameter mapping.

Source checkpoint: ``Comfy-Org/Wan_2.2_ComfyUI_Repackaged``,
``split_files/diffusion_models/wan2.2_t2v_{high,low}_noise_14B_fp8_scaled.safetensors``.
High-noise maps to ``transformer``, low-noise to ``transformer_2``.

The two files are structurally identical: 1908 tensors each
(407 ``F8_E4M3`` + 812 ``F32`` + 689 ``F16``), a uniform 47-key set across all
40 blocks, plus 28 top-level keys.

Naming is **native ComfyUI/Wan**, not Diffusers -- ``self_attn.{q,k,v,o}`` rather
than ``attn1.to_{q,k,v}`` / ``attn1.to_out.0``, ``ffn.{0,2}`` rather than
``ffn.net.0.proj`` / ``ffn.net.2``, and the block norms are ordered differently
(see the renames in :func:`build_fp8_dequant_mappings`).

Per quantized linear the checkpoint stores four keys::

    <prefix>.weight        F8_E4M3  [out, in]   per-tensor absmax quantized
    <prefix>.scale_weight  F32      []          scalar dequant scale
    <prefix>.scale_input   F32      []          scalar activation scale (unused)
    <prefix>.bias          F16      [out]

``scale_input`` is a calibrated activation scale that this path does not use -- the
BF16 forward needs no activation quantization. ``scaled_fp8`` is an empty
(``shape []``) marker tensor ComfyUI writes to flag the format.

This module is pure name/shape bookkeeping: it performs no device work and imports
nothing from ``torch`` / ``torch_neuronx``, so it is fully unit-testable on a CPU host.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Collection

# ---------------------------------------------------------------------------
# Checkpoint facts (validated against the real safetensors headers)
# ---------------------------------------------------------------------------

#: Files under ``split_files/diffusion_models/`` and the module each maps to.
FP8_CHECKPOINT_FILES: dict[str, str] = {
    "transformer": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "transformer_2": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
}

#: Subdirectory holding the DiT files inside the HF repo.
FP8_CHECKPOINT_SUBDIR = "split_files/diffusion_models"

#: FP8 weight dtype as safetensors records it in the header.
FP8_DTYPE_STR = "F8_E4M3"

#: Number of transformer blocks in each expert.
NUM_BLOCKS = 40
SUPPORTED_NATIVE_FP8_MODULES = frozenset({"attn1", "attn2", "ffn"})


class Fp8CheckpointError(RuntimeError):
    """Raised when a checkpoint does not look like the expected Wan2.2 ComfyUI FP8 file."""


# ---------------------------------------------------------------------------
# Safetensors header helpers + checkpoint sanity checks
# ---------------------------------------------------------------------------


def read_safetensors_header(path: str) -> dict[str, dict]:
    """Read a ``.safetensors`` file's JSON header (``{name: {dtype, shape, ...}}``).

    Parses the length-prefixed JSON header directly (stdlib only, no torch/safetensors
    dependency), with ``__metadata__`` removed.
    """
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


# ---------------------------------------------------------------------------
# Neuron model-parameter mapping (dequant-at-load path)
#
# Maps each ``WanTransformer3DModel`` parameter to the ComfyUI checkpoint key(s) that
# feed it, so ``load_weights`` can dequant every quantized Linear to BF16. ComfyUI uses
# native-Wan naming and orders the block norms differently from this model, so three
# keys are renamed relative to a plain pass-through. A wrong or missing rename would
# silently drop a tensor under ``strict=False`` (garbage output, no error), so the load
# path asserts full bijectivity against the model's parameters:
#
#   comfy ``blocks.N.norm3.{weight,bias}`` -> model ``blocks.N.norm2.{weight,bias}``
#   comfy ``blocks.N.modulation``          -> model ``blocks.N.scale_shift_table``
#   comfy ``head.modulation``              -> model top-level ``scale_shift_table``
#
# Kept torch-free so the correspondence is unit-testable against the checkpoint header
# alone, with no hardware / vllm_omni.
# ---------------------------------------------------------------------------

#: ComfyUI top-level DEQUANT leaf -> this model's replicated ``nn.Linear`` prefix.
#: These sit outside the TP-sharded projection kernels (the condition embedder and the
#: final head) and are dequantized whole (no transpose, no shard). The model-side names
#: are the Diffusers ``WanTimeTextImageEmbedding`` / ``proj_out`` names the plain BF16
#: path already relies on via name-matching, so any drift is caught by the bijectivity
#: check.
FP8_TOP_LINEAR_MAP: dict[str, str] = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "head.head": "proj_out",
}


def fp8_transformed_parameter_names(
    num_layers: int = NUM_BLOCKS,
    native_fp8_modules: Collection[str] = (),
) -> list[str]:
    """Parameters populated by native-pack or CPU-dequant FP8 transforms.

    CPU-dequant modules contribute their weight parameter. Native modules additionally
    contribute runtime weight-scale parameters populated directly from the checkpoint.
    """
    native_fp8_modules = frozenset(native_fp8_modules)
    unsupported = native_fp8_modules - SUPPORTED_NATIVE_FP8_MODULES
    if unsupported:
        raise ValueError(f"unsupported native FP8 modules: {sorted(unsupported)}")

    names: list[str] = []
    for i in range(num_layers):
        b = f"blocks.{i}"
        names += [
            f"{b}.attn1.qkv_proj_weight",
            f"{b}.attn1.o_proj_weight",
            f"{b}.attn2.q_proj_weight",
            f"{b}.attn2.k_proj_weight",
            f"{b}.attn2.v_proj_weight",
            f"{b}.attn2.o_proj_weight",
            f"{b}.ffn.up_proj_weight",
            f"{b}.ffn.down_proj_weight",
        ]
        if "attn1" in native_fp8_modules:
            names += [
                f"{b}.attn1.qkv_proj_w_scale",
                f"{b}.attn1.o_proj_w_scale",
            ]
        if "attn2" in native_fp8_modules:
            names += [f"{b}.attn2.{projection}_proj_w_scale" for projection in ("q", "k", "v", "o")]
        if "ffn" in native_fp8_modules:
            names += [
                f"{b}.ffn.up_proj_w_scale",
                f"{b}.ffn.down_proj_w_scale",
            ]
    names += [f"{prefix}.weight" for prefix in FP8_TOP_LINEAR_MAP.values()]
    return names


def build_fp8_dequant_mappings(num_layers: int = NUM_BLOCKS) -> dict[str, str | list[str]]:
    """Model-parameter -> ComfyUI checkpoint key(s) for the fp8_row_mx load path.

    Quantized weights map to ``[weight, scale_weight]`` (fused self-attn QKV to the six
    keys ``[qW, qS, kW, kS, vW, vS]``); biases / norms / modulation map to a single F16
    key. Mirrors the plain BF16 ``load_weights`` mapping shape, swapped to the ComfyUI
    namespace with the norm/modulation renames applied. Pure (no torch / model) so
    bijectivity and coverage are testable against the checkpoint header alone.

    NOTE: this targets Wan2.2 T2V (``added_kv_proj_dim`` unset) -- the ComfyUI checkpoint
    has no image cross-attention (``add_k/add_v``) keys.
    """
    m: dict[str, str | list[str]] = {}
    for i in range(num_layers):
        b = f"blocks.{i}"
        # --- self-attention: fused QKV (three separately-scaled sources) ----------
        m[f"{b}.attn1.qkv_proj_weight"] = [
            f"{b}.self_attn.q.weight",
            f"{b}.self_attn.q.scale_weight",
            f"{b}.self_attn.k.weight",
            f"{b}.self_attn.k.scale_weight",
            f"{b}.self_attn.v.weight",
            f"{b}.self_attn.v.scale_weight",
        ]
        m[f"{b}.attn1.qkv_proj_bias"] = [
            f"{b}.self_attn.q.bias",
            f"{b}.self_attn.k.bias",
            f"{b}.self_attn.v.bias",
        ]
        m[f"{b}.attn1.o_proj_weight"] = [
            f"{b}.self_attn.o.weight",
            f"{b}.self_attn.o.scale_weight",
        ]
        m[f"{b}.attn1.o_proj_bias"] = f"{b}.self_attn.o.bias"
        m[f"{b}.attn1.norm_q.weight"] = f"{b}.self_attn.norm_q.weight"
        m[f"{b}.attn1.norm_k.weight"] = f"{b}.self_attn.norm_k.weight"
        # --- cross-attention: Q separate, K/V separate (this model does NOT fuse) --
        for proj in ("q", "k", "v"):
            m[f"{b}.attn2.{proj}_proj_weight"] = [
                f"{b}.cross_attn.{proj}.weight",
                f"{b}.cross_attn.{proj}.scale_weight",
            ]
            m[f"{b}.attn2.{proj}_proj_bias"] = f"{b}.cross_attn.{proj}.bias"
        m[f"{b}.attn2.norm_q.weight"] = f"{b}.cross_attn.norm_q.weight"
        m[f"{b}.attn2.norm_k.weight"] = f"{b}.cross_attn.norm_k.weight"
        m[f"{b}.attn2.o_proj_weight"] = [
            f"{b}.cross_attn.o.weight",
            f"{b}.cross_attn.o.scale_weight",
        ]
        m[f"{b}.attn2.o_proj_bias"] = f"{b}.cross_attn.o.bias"
        # --- cross-attn norm rename: comfy norm3 -> model norm2 --------------------
        m[f"{b}.norm2.weight"] = f"{b}.norm3.weight"
        m[f"{b}.norm2.bias"] = f"{b}.norm3.bias"
        # --- FFN -------------------------------------------------------------------
        m[f"{b}.ffn.up_proj_weight"] = [f"{b}.ffn.0.weight", f"{b}.ffn.0.scale_weight"]
        m[f"{b}.ffn.up_proj_bias"] = f"{b}.ffn.0.bias"
        m[f"{b}.ffn.down_proj_weight"] = [f"{b}.ffn.2.weight", f"{b}.ffn.2.scale_weight"]
        m[f"{b}.ffn.down_proj_bias"] = f"{b}.ffn.2.bias"
        # --- modulation rename: comfy modulation -> model scale_shift_table --------
        m[f"{b}.scale_shift_table"] = f"{b}.modulation"

    # --- top level ----------------------------------------------------------------
    m["patch_embedding.weight"] = "patch_embedding.weight"
    m["patch_embedding.bias"] = "patch_embedding.bias"
    m["scale_shift_table"] = "head.modulation"
    for comfy_leaf, model_prefix in FP8_TOP_LINEAR_MAP.items():
        m[f"{model_prefix}.weight"] = [
            f"{comfy_leaf}.weight",
            f"{comfy_leaf}.scale_weight",
        ]
        m[f"{model_prefix}.bias"] = f"{comfy_leaf}.bias"
    return m


def build_fp8_mappings(
    num_layers: int = NUM_BLOCKS,
    native_fp8_modules: Collection[str] = (),
) -> dict[str, str | list[str]]:
    """Build mappings for native ROW_MX and CPU-dequant modules.

    Packed weights and their runtime scales are separate model parameters, so
    each maps only to the checkpoint tensor it consumes. This avoids reading a
    large FP8 weight again merely to construct its small scale parameter.
    """
    native_fp8_modules = frozenset(native_fp8_modules)
    unsupported = native_fp8_modules - SUPPORTED_NATIVE_FP8_MODULES
    if unsupported:
        raise ValueError(f"unsupported native FP8 modules: {sorted(unsupported)}")

    mappings = build_fp8_dequant_mappings(num_layers)
    for i in range(num_layers):
        block = f"blocks.{i}"

        if "attn1" in native_fp8_modules:
            fused_sources = mappings.pop(f"{block}.attn1.qkv_proj_weight")
            assert isinstance(fused_sources, list) and len(fused_sources) == 6
            mappings[f"{block}.attn1.qkv_proj_weight"] = fused_sources[0::2]
            mappings[f"{block}.attn1.qkv_proj_w_scale"] = fused_sources[1::2]

            name = f"{block}.attn1.o_proj_weight"
            sources = mappings[name]
            assert isinstance(sources, list) and len(sources) == 2
            mappings[name] = sources[0]
            mappings[f"{block}.attn1.o_proj_w_scale"] = sources[1]

        if "attn2" in native_fp8_modules:
            for projection in ("q", "k", "v"):
                name = f"{block}.attn2.{projection}_proj_weight"
                sources = mappings[name]
                assert isinstance(sources, list) and len(sources) == 2
                mappings[name] = sources[0]
                mappings[f"{block}.attn2.{projection}_proj_w_scale"] = sources[1]

            name = f"{block}.attn2.o_proj_weight"
            sources = mappings[name]
            assert isinstance(sources, list) and len(sources) == 2
            mappings[name] = sources[0]
            mappings[f"{block}.attn2.o_proj_w_scale"] = sources[1]

        if "ffn" in native_fp8_modules:
            for projection in ("up", "down"):
                name = f"{block}.ffn.{projection}_proj_weight"
                sources = mappings[name]
                assert isinstance(sources, list) and len(sources) == 2
                mappings[name] = sources[0]
                mappings[f"{block}.ffn.{projection}_proj_w_scale"] = sources[1]
    return mappings


def validate_fp8_checkpoint(header: dict[str, dict], num_layers: int = NUM_BLOCKS) -> None:
    """Validate the checkpoint tensors consumed by the FP8 dequant loaders.

    Checks the mapped contract rather than the exact file histogram, so reduced-layer
    checkpoints remain valid. Every quantized weight must be two-dimensional F8_E4M3 and
    have a scalar F32 ``scale_weight``.
    """
    mappings = build_fp8_dequant_mappings(num_layers)
    for param_name in fp8_transformed_parameter_names(num_layers):
        sources = mappings[param_name]
        if not isinstance(sources, list) or len(sources) % 2:
            raise RuntimeError(f"invalid FP8 mapping for {param_name}: {sources}")

        for weight_key, scale_key in zip(sources[0::2], sources[1::2]):
            weight = header.get(weight_key)
            if weight is None:
                raise Fp8CheckpointError(f"missing FP8 weight: {weight_key}")
            if weight.get("dtype") != FP8_DTYPE_STR:
                raise Fp8CheckpointError(
                    f"{weight_key} must be {FP8_DTYPE_STR}, got {weight.get('dtype')}"
                )
            if len(weight.get("shape", [])) != 2:
                raise Fp8CheckpointError(
                    f"{weight_key} must be 2-D [out, in], got shape {weight.get('shape')}"
                )

            scale = header.get(scale_key)
            if scale is None:
                raise Fp8CheckpointError(f"missing FP8 weight scale: {scale_key}")
            if scale.get("dtype") != "F32":
                raise Fp8CheckpointError(f"{scale_key} must be F32, got {scale.get('dtype')}")
            if scale.get("shape") not in ([], [1]):
                raise Fp8CheckpointError(
                    f"{scale_key} must be scalar, got shape {scale.get('shape')}"
                )
