# SPDX-License-Identifier: Apache-2.0
"""Wan 2.2 I2V-A14B (image-to-video) on Neuron via the Omni entrypoint.

Usage:
    python examples/wan22/run_i2v.py                               # uses i2v_input.JPG
    python examples/wan22/run_i2v.py --image path/to/first_frame.png
    python examples/wan22/run_i2v.py --dev                         # small/fast
    python examples/wan22/run_i2v.py --last-image out.png          # FLF2V
    python examples/wan22/run_i2v.py --profile                     # warmup + timed inference

Mirrors ``run.py`` (T2V) but drives the I2V pipeline: the conditioning image is
passed through ``multi_modal_data`` and the model class is
``Wan22I2VPipeline`` (resolved to NeuronWanI2VPipeline by the plugin).
"""

import argparse
import os
import time

import PIL.Image
import torch
import yaml

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from vllm_omni_neuron import env_profiles

# torch_neuronx patches F.gelu with a wrapper around the C builtin; Dynamo
# cannot trace through the C function in fullgraph mode. Restore the aten op.
torch.nn.functional.gelu = torch.ops.aten.gelu.default

parser = argparse.ArgumentParser(description="Wan 2.2 I2V-A14B on Neuron")
parser.add_argument("--dev", action="store_true", help="Dev mode: fewer frames, lower resolution")
parser.add_argument("--tensor-parallel-size", type=int, default=8)
parser.add_argument(
    "--num-layers",
    type=int,
    default=None,
    help="Number of transformer layers (default: use model config, typically 40)",
)
parser.add_argument("--height", type=int, default=None)
parser.add_argument("--width", type=int, default=None)
parser.add_argument("--num-frames", type=int, default=None)
parser.add_argument(
    "--model-path",
    type=str,
    default="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    help="Local path or HF id of the Wan 2.2 I2V checkpoint.",
)
parser.add_argument(
    "--image",
    type=str,
    default="i2v_input.JPG",
    help="Path to the conditioning (first-frame) image. "
    "Defaults to i2v_input.JPG next to this script.",
)
parser.add_argument(
    "--last-image",
    type=str,
    default=None,
    help="Optional last-frame image for first-last-frame (FLF2V) conditioning.",
)
parser.add_argument("--output", type=str, default="wan_i2v_output.mp4", help="Output video path")
parser.add_argument(
    "--stage-config",
    type=str,
    default=None,
    help="Path to the stage config YAML. Defaults to wan22_i2v_stage.yaml next to this script.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
)
parser.add_argument(
    "--no-cfg",
    action="store_true",
    help="Disable classifier-free guidance. Sets guidance_scale=1.0.",
)
parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale (default: 5.0).")
parser.add_argument(
    "--profile",
    action="store_true",
    help="Profile mode: run warmup inference after compilation, then measure timed inference runs.",
)
args = parser.parse_args()

# Number of timed inference runs after warmup (keep low — model is not yet optimized).
PROFILE_NUM_RUNS = 1

_tp = args.tensor_parallel_size
# Use predefined wan22 i2v env vars; kept identical to run.py so I2V and T2V
# compile under the same runtime configuration.
env_profiles.apply(
    env_profiles.WAN22_I2V,
    env_profiles.thread_limits(_tp),
)


def _load_model_config(stage_cfg_path: str) -> dict:
    """Return the stage YAML's engine_args.model_config block, or {} if absent."""
    with open(stage_cfg_path) as f:
        stage_cfg = yaml.safe_load(f)
    return dict(stage_cfg["stage_args"][0]["engine_args"].get("model_config") or {})


def main():
    stage_cfg = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "wan22_i2v_stage.yaml"
    )

    model_config = _load_model_config(stage_cfg)
    if args.num_layers is not None:
        model_config["num_layers"] = args.num_layers

    # A cold 480p/720p compile far exceeds vllm_omni's default 600s worker
    # handshake timeout, so the parent tears the worker down mid-compile. Raise
    # the ceiling before Omni init (the orchestrator reads the patched value);
    # warm re-runs finish well under it. Mirrors run.py.
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

    # Resolve a relative --image against this script's directory so the default
    # (i2v_input.JPG) works regardless of the current working directory.
    image_path = args.image
    if not os.path.isabs(image_path) and not os.path.exists(image_path):
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)

    print(f"Using conditioning image: {image_path}")
    image = PIL.Image.open(image_path).convert("RGB")
    multi_modal_data = {"image": image}
    if args.last_image:
        multi_modal_data["last_image"] = PIL.Image.open(args.last_image).convert("RGB")

    print(f"Generating I2V: {num_frames} frames, {height}x{width}, {num_steps} steps")
    inputs = {"prompt": args.prompt, "multi_modal_data": multi_modal_data}
    result = omni.generate(inputs, params)

    if args.profile:
        # Warmup run (compilation already happened during the first generate above)
        print("\n[Profile] Warmup inference complete.")

        # Timed inference run(s)
        run_times = []
        for i in range(PROFILE_NUM_RUNS):
            print(f"\n[Profile] Timed run {i + 1}/{PROFILE_NUM_RUNS}...")
            t_start = time.perf_counter()
            result = omni.generate(inputs, params)
            elapsed = time.perf_counter() - t_start
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
