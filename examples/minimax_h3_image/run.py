# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image text-to-image on Neuron via the Omni entrypoint.

Usage:
    python examples/minimax_h3_image/run.py --output out.png               # 2048x2048
    python examples/minimax_h3_image/run.py --height 1024 --width 1024 \
        --min-generation-area 1024 --output out_1024.png                  # native 1024x1024
    python examples/minimax_h3_image/run.py --profile                       # warm timing
"""

import argparse
import os
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import torch
import yaml
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from vllm_omni_neuron import env_profiles

torch.nn.functional.gelu = torch.ops.aten.gelu.default

parser = argparse.ArgumentParser(description="MiniMax-H3-Image on Neuron")
parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3-Image")
parser.add_argument("--stage-config", default=None)
parser.add_argument("--prompt", default="A highland cow standing in a misty meadow at sunrise, photorealistic, detailed fur")
parser.add_argument("--prompt-file", default=None, help="Read the prompt from a file (overrides --prompt)")
parser.add_argument("--image", action="append", default=None,
                    help="Reference image for editing; repeat for <Picture 2>, <Picture 3>, ...")
parser.add_argument("--mask", default=None,
                    help="Inpainting mask for <Picture 1> (white = regenerate, black = keep)")
parser.add_argument("--outpaint", default=None,
                    help="Outpainting: extend <Picture 1> by LEFT,TOP,RIGHT,BOTTOM pixels (sets the output size)")
parser.add_argument("--mask-mode", choices=["blend", "paste"], default="blend",
                    help="blend: keep the unmasked latents at every step and paste back; paste: paste back only")
parser.add_argument("--mask-feather", type=int, default=32, help="Feather of the pixel composite, in pixels")
parser.add_argument("--height", type=int, default=2048)
parser.add_argument("--width", type=int, default=2048)
parser.add_argument("--min-generation-area", type=int, default=2048 * 2048)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-layers", type=int, default=None, help="Truncate the DiT (debug)")
parser.add_argument("--quantization", choices=["bf16", "fp8_ffn"], default=None,
                    help="DiT FFN precision; fp8_ffn runs the FFN as FP8 W8A8 (Trn2 nkilib ROW kernel)")
parser.add_argument("--cache-steps", default=None,
                    help="Comma-separated steps whose middle block groups reuse the previous step's residual, e.g. 3,5")
parser.add_argument("--output", default="minimax_h3_image.png")
parser.add_argument("--profile", action="store_true", help="Run timed warm generations after the first")
parser.add_argument("--profile-runs", type=int, default=3)
parser.add_argument("--throughput", type=int, default=0,
                    help="After the first generation, submit N requests at once "
                         "and report images/s; with a multi-replica stage config they run concurrently")
args = parser.parse_args()

env_profiles.apply(env_profiles.MINIMAX_H3_IMAGE, env_profiles.thread_limits(16))


def main():
    stage_cfg = args.stage_config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimax_h3_image_stage.yaml")
    with open(stage_cfg) as f:
        model_config = dict(yaml.safe_load(f)["stages"][0]["engine_args"].get("model_config") or {})
    if args.num_layers is not None:
        model_config["num_layers"] = args.num_layers
    if args.quantization is not None:
        model_config["quantization"] = args.quantization
    if args.cache_steps:
        model_config["cache_steps"] = [int(s) for s in args.cache_steps.split(",")]

    cold_timeout = int(os.environ.get("H3_HANDSHAKE_TIMEOUT_S", "3600"))
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as _sdp

        if getattr(_sdp, "_HANDSHAKE_POLL_TIMEOUT_S", 0) < cold_timeout:
            _sdp._HANDSHAKE_POLL_TIMEOUT_S = cold_timeout
    except Exception as e:  # noqa: BLE001
        print(f"[init] could not patch handshake timeout ({e!r})")

    omni = Omni(
        model=args.model_path,
        deploy_config=stage_cfg,
        stage_init_timeout=cold_timeout,
        init_timeout=cold_timeout,
        model_config=model_config,
    )
    params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed,
        extra_args={"min_generation_area": args.min_generation_area},
    )
    prompt = open(args.prompt_file).read() if args.prompt_file else args.prompt
    request = {"prompt": prompt}
    if args.image:
        from PIL import Image

        request["multi_modal_data"] = {"image": [Image.open(p) for p in args.image]}
    if args.mask or args.outpaint:
        from PIL import Image, ImageFilter

        if not args.image:
            raise SystemExit("--mask / --outpaint need --image (the image to inpaint or extend)")
        src = Image.open(args.image[0]).convert("RGB")
        if args.outpaint:
            left, top, right, bottom = (int(v) for v in args.outpaint.split(","))
            w, h = src.width + left + right, src.height + top + bottom
            w32, h32 = -(-w // 32) * 32, -(-h // 32) * 32  # canvas sizes are multiples of 32
            right, bottom = right + w32 - w, bottom + h32 - h
            # init canvas: the image stretched over the new area (only seen inside the mask)
            init = src.resize((w32, h32), Image.BILINEAR).filter(ImageFilter.GaussianBlur(32))
            init.paste(src, (left, top))
            mask = Image.new("L", (w32, h32), 255)
            mask.paste(0, (left, top, left + src.width, top + src.height))
            params.height, params.width = h32, w32
            params.extra_args["init_image"] = init
            print(f"[outpaint] {src.size} -> {(w32, h32)} (left {left}, top {top}, right {right}, bottom {bottom})")
        else:
            mask = Image.open(args.mask)
        params.extra_args.update(mask=mask, mask_mode=args.mask_mode, mask_feather=args.mask_feather)
    t0 = time.perf_counter()
    result = omni.generate(request, params)
    print(f"first generation (includes compilation on a cold cache): {time.perf_counter() - t0:.2f}s")
    if args.profile:
        times = []
        for i in range(args.profile_runs):
            t0 = time.perf_counter()
            result = omni.generate(request, params)
            times.append(time.perf_counter() - t0)
            print(f"[profile] run {i + 1}: {times[-1]:.3f}s")
        print(f"[profile] {args.height}x{args.width}: avg {sum(times) / len(times):.3f}s min {min(times):.3f}s")

    if args.throughput:
        reqs = [dict(request) for _ in range(args.throughput)]
        omni.generate(reqs, params)  # warm every replica (each compiles / loads on first use)
        t0 = time.perf_counter()
        outs = omni.generate(reqs, params)
        dt = time.perf_counter() - t0
        print(f"[throughput] {len(outs)} images {args.height}x{args.width} in {dt:.2f}s: {len(outs) / dt:.3f} images/s")

    images = result[0].images
    image = images[0] if isinstance(images, list) else images
    if isinstance(image, list):
        image = image[0]
    if hasattr(image, "save"):
        image.save(args.output)
    else:
        import numpy as np
        from PIL import Image

        arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
        while arr.ndim > 3:
            arr = arr[0]
        if arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
