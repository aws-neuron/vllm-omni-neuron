# SPDX-License-Identifier: Apache-2.0
"""Wan 2.2 T2V-A14B on Neuron via the Omni entrypoint.

Usage:
    python examples/wan22/run.py                          # Full: 81 frames, 480x832
    python examples/wan22/run.py --dev                    # Dev: 5 frames, 240x416
    python examples/wan22/run.py --tensor-parallel-size 8 # 8-way TP
    python examples/wan22/run.py --profile                # Profile: warmup + timed inference
"""

import argparse
import os
import time
from dataclasses import replace

import torch
import yaml

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from vllm_omni_neuron import env_profiles

# torch_neuronx patches F.gelu with a wrapper around the C builtin;
# Dynamo cannot trace through the C function in fullgraph mode.
# Restore with the aten op which is dispatcher-aware and accepts kwargs.
torch.nn.functional.gelu = torch.ops.aten.gelu.default

# Parse args early so env vars are set before Omni spawns workers
parser = argparse.ArgumentParser(description="Wan 2.2 T2V-A14B on Neuron")
parser.add_argument("--dev", action="store_true", help="Dev mode: fewer frames, lower resolution")
parser.add_argument(
    "--num-layers",
    type=int,
    default=None,
    help="Number of transformer layers (default: use model config, typically 40)",
)
parser.add_argument("--tensor-parallel-size", type=int, default=8)
parser.add_argument(
    "--platform-target",
    choices=["trn2", "trn3"],
    default=None,
    help="Override Neuron platform detection. Required only when the runtime cannot "
    "detect the target, such as with an editable torch-neuronx build.",
)
parser.add_argument("--height", type=int, default=None)
parser.add_argument("--width", type=int, default=None)
parser.add_argument("--num-frames", type=int, default=None)
parser.add_argument(
    "--model-path",
    type=str,
    default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    help="Local path to Wan 2.2 checkpoint. Downloads from HF if not set.",
)
parser.add_argument(
    "--comfyui-fp8-model-path",
    type=str,
    default="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
    help="Local root or Hugging Face repository containing the ComfyUI FP8 "
    "high-noise and low-noise DiT checkpoints.",
)
parser.add_argument("--output", type=str, default="wan_output.mp4", help="Output video file path")
parser.add_argument(
    "--quantization",
    type=str,
    default=None,
    choices=["bf16", "fp8_row_mx"],
    help="DiT quantization mode. 'fp8_row_mx' loads a ComfyUI per-tensor FP8-scaled "
    "checkpoint and runs QKV, output, and FFN projections with native ROW_MX kernels. "
    "Default (None) = plain BF16.",
)
parser.add_argument(
    "--modules-to-not-convert",
    nargs="*",
    choices=["attn1", "attn2", "ffn"],
    default=None,
    help="FP8 checkpoint modules to CPU-dequant. By default all projection groups use native "
    "ROW_MX kernels; use 'attn1 attn2 ffn' for the full CPU-dequant reference.",
)
parser.add_argument(
    "--stage-config",
    type=str,
    default=None,
    help="Path to the stage config YAML. Defaults to wan22_stage.yaml next to this script.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="A fluffy orange cat walking gracefully across a sunny garden path, high quality, detailed",
)
parser.add_argument(
    "--cache-backend",
    type=str,
    default=None,
    choices=["cache_dit"],
    help="Cache backend for acceleration. Options: 'cache_dit'. Default: None.",
)
parser.add_argument(
    "--cache-summary",
    action="store_true",
    help="Log cache-dit statistics (cached vs computed steps, residual diffs) after each request.",
)
parser.add_argument(
    "--no-cfg",
    action="store_true",
    help="Disable classifier-free guidance (skip negative prompt pass). Sets guidance_scale=1.0.",
)
parser.add_argument(
    "--guidance-scale",
    type=float,
    default=5.0,
    help="Guidance scale for CFG (default: 5.0). Ignored when --no-cfg is set.",
)
parser.add_argument(
    "--profile",
    action="store_true",
    help="Profile mode: run warmup inference after compilation, then measure timed inference runs.",
)
args = parser.parse_args()

# Number of timed inference runs after warmup (keep low — model is not yet optimized).
# Override via WAN_PROFILE_NUM_RUNS to average several warm runs (e.g. when comparing
# two attention kernels, where run-to-run variance matters).
PROFILE_NUM_RUNS = int(os.environ.get("WAN_PROFILE_NUM_RUNS", "1"))

# Flag to show detailed memory info per NEFF
# os.environ["NEURON_RT_LOG_LEVEL_TDRV"] = "info"

# Set thread limits BEFORE spawning workers — ensures env vars are inherited
# by child processes before they import torch (which reads OMP_NUM_THREADS).
_tp = args.tensor_parallel_size
# Use predefined wan22 t2v env vars; --platform-target is None unless passed, and a
# None field is skipped, so the profile's trn2 default is not written here.
env_profiles.apply(
    replace(env_profiles.WAN22_T2V, NEURON_PLATFORM_TARGET_OVERRIDE=args.platform_target),
    env_profiles.thread_limits(_tp),
)


def _load_model_config(stage_cfg_path: str) -> dict:
    """Return the stage YAML's engine_args.model_config block, or {} if absent."""
    with open(stage_cfg_path) as f:
        stage_cfg = yaml.safe_load(f)
    return dict(stage_cfg["stage_args"][0]["engine_args"].get("model_config") or {})


def main():
    stage_cfg = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "wan22_stage.yaml"
    )

    cache_config = None
    if args.cache_backend == "cache_dit":
        cache_config = {
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 4,
            "max_cached_steps": 20,
            "residual_diff_threshold": 0.24,
            "max_continuous_cached_steps": 3,
            "enable_taylorseer": False,
        }

    model_config = _load_model_config(stage_cfg)
    if args.num_layers is not None:
        model_config["num_layers"] = args.num_layers
    if args.quantization is not None:
        model_config["quantization"] = args.quantization
    if args.modules_to_not_convert is not None:
        model_config["modules_to_not_convert"] = args.modules_to_not_convert
    model_config["comfyui_fp8_model_path"] = args.comfyui_fp8_model_path

    # The orchestrator waits at most _HANDSHAKE_POLL_TIMEOUT_S (hardcoded 600s in
    # vllm_omni) for the diffusion worker to signal READY. A COLD 480p/720p compile
    # (~40-50 min) far exceeds that, so on a cold NEFF cache the parent gives up
    # mid-compile and tears the worker down. Raise the ceiling in-process before Omni
    # init (the orchestrator thread reads the patched value). Warm re-runs finish well
    # under it. Mirrors eval/vbench/generate.py.
    _cold_timeout = int(os.environ.get("WAN_HANDSHAKE_TIMEOUT_S", "3600"))
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as _sdp

        if getattr(_sdp, "_HANDSHAKE_POLL_TIMEOUT_S", 0) < _cold_timeout:
            _sdp._HANDSHAKE_POLL_TIMEOUT_S = _cold_timeout
            print(f"[init] raised diffusion handshake timeout to {_cold_timeout}s (cold compile)")
    except Exception as e:
        print(f"[init] could not patch handshake timeout ({e!r}); using default 600s")

    omni = Omni(
        model=args.model_path,
        stage_configs_path=stage_cfg,
        stage_init_timeout=_cold_timeout,
        init_timeout=_cold_timeout,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        enable_cache_dit_summary=args.cache_summary,
        model_config=model_config,
    )

    if args.dev:
        height, width, num_frames, num_steps = 128, 208, 5, 10
    else:
        height, width, num_frames, num_steps = 480, 832, 81, 40

    height = args.height or height
    width = args.width or width
    num_frames = args.num_frames or num_frames

    params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        guidance_scale=1.0 if args.no_cfg else args.guidance_scale,
        seed=42,
    )

    print(f"Generating: {num_frames} frames, {height}x{width}, {num_steps} steps")
    result = omni.generate({"prompt": args.prompt}, params)

    if args.profile:
        # Warmup run (compilation already happened during the first generate above)
        print("\n[Profile] Warmup inference complete.")

        # Timed inference run(s)
        run_times = []
        for i in range(PROFILE_NUM_RUNS):
            print(f"\n[Profile] Timed run {i + 1}/{PROFILE_NUM_RUNS}...")
            t_start = time.perf_counter()
            result = omni.generate({"prompt": args.prompt}, params)
            t_end = time.perf_counter()
            elapsed = t_end - t_start
            run_times.append(elapsed)
            print(f"[Profile] Run {i + 1}: {elapsed:.2f}s")

        avg_time = sum(run_times) / len(run_times)
        print(f"\n{'=' * 60}")
        print(f"[Profile] Results ({num_frames} frames, {height}x{width}, {num_steps} steps):")
        print(f"  Runs:         {PROFILE_NUM_RUNS}")
        print(f"  Avg latency:  {avg_time:.2f}s")
        if PROFILE_NUM_RUNS > 1:
            print(f"  Min latency:  {min(run_times):.2f}s")
            print(f"  Max latency:  {max(run_times):.2f}s")
        print(f"{'=' * 60}")

    frames = result[0].request_output.images
    print(f"frames type: {type(frames)}, len: {len(frames)}")
    if frames:
        print(
            f"frames[0] type: {type(frames[0])}, shape/size: {getattr(frames[0], 'shape', None) or getattr(frames[0], 'size', None)}"
        )
    import numpy as np
    from diffusers.utils import export_to_video

    # vllm-omni 0.24 returns each output as a numpy array in (batch, T, H, W, C)
    # layout (channel-last, frame axis at index 1) — earlier builds returned a torch
    # tensor in (batch, C, T, H, W). Normalize either into (T, H, W, C) frames.
    for out_idx, output in enumerate(frames):
        video = output.detach().cpu().numpy() if hasattr(output, "detach") else np.asarray(output)
        if video.ndim == 5:
            video = video[0]  # drop batch -> 4D
        # 4D now: either (C, T, H, W) (channel-first) or (T, H, W, C) (channel-last).
        if video.shape[0] in (3, 4) and video.shape[-1] not in (3, 4):
            video = np.transpose(video, (1, 2, 3, 0))  # (C,T,H,W) -> (T,H,W,C)
        if np.issubdtype(video.dtype, np.floating):
            # [-1,1] model output -> [0,1] for the encoder; already-[0,1] stays put.
            if float(video.min()) < 0.0:
                video = np.clip(video, -1.0, 1.0) * 0.5 + 0.5
            video = np.clip(video, 0.0, 1.0)
        video_array = video.astype(np.float32)
        path = args.output if out_idx == 0 else f"{args.output}_{out_idx}.mp4"
        export_to_video(list(video_array), path, fps=16)
        print(f"Saved {video_array.shape[0]} frames to {path}")


if __name__ == "__main__":
    main()
