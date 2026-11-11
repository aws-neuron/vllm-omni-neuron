# SPDX-License-Identifier: Apache-2.0
"""Single-device FLUX.2 Klein Diffusers adapter smoke test through Omni."""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--prompt", default="A golden retriever surfing a wave, photorealistic")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("flux2-klein-output"))
    args = parser.parse_args()

    import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    args.output_dir.mkdir(parents=True, exist_ok=True)
    omni = Omni(
        model=args.model,
        diffusion_load_format="diffusers",
        dtype="bfloat16",
        tensor_parallel_size=1,
        stage_init_timeout=1800,
        init_timeout=1800,
    )
    try:
        for index in range(args.runs):
            # Fresh sampling params keep generator state independent per request.
            params = OmniDiffusionSamplingParams(
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                seed=args.seed,
                generator_device="cpu",
            )
            start = time.perf_counter()
            results = omni.generate({"prompt": args.prompt}, params)
            elapsed = time.perf_counter() - start
            image = results[0].images[0]
            if image.size != (args.width, args.height):
                raise RuntimeError(f"Unexpected output size: {image.size}")
            image.save(args.output_dir / f"request-{index}.png")
            print(
                f"request={index} elapsed={elapsed:.3f}s (first request includes cold compilation)"
            )

    finally:
        omni.close()


if __name__ == "__main__":
    main()
