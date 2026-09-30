# SPDX-License-Identifier: Apache-2.0
"""
Step-Level Tensor Capture & Comparison.

Runs the Wan 2.2 pipeline on Neuron via the Omni entrypoint with tensor_capture
configured. The NeuronDiffusionModelRunner registers ModelCapture hooks on transformers
before compile — hooks are traced into the compiled graph and captures are written
to disk automatically during the denoising loop.

Optionally compares the captured noise predictions against a previously saved
baseline (--baseline-dir) and expected (--expected-dir) run using three-way
stepwise comparison (FP32 baseline vs BF16 CPU vs BF16 Neuron).

Usage (requires Neuron hardware):
    # Capture only:
    python examples/wan22/run_step_capture.py --capture-dir /tmp/captures
    python examples/wan22/run_step_capture.py --modules "blocks.0-3" --num-steps 10

    # Capture + compare against saved baselines:
    python examples/wan22/run_step_capture.py --capture-dir /tmp/captures_neuron \
        --baseline-dir /tmp/captures_fp32 --expected-dir /tmp/captures_bf16_cpu
"""

import argparse
import os

from vllm_omni_neuron import env_profiles

parser = argparse.ArgumentParser(description="Capture block outputs per denoising step")
parser.add_argument("--num-steps", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--capture-dir",
    type=str,
    default="/tmp/diffusion_captures",
    help="Output directory for captured tensors",
)
parser.add_argument(
    "--modules", type=str, default="blocks.0-1", help="Comma-separated module patterns to capture"
)
parser.add_argument("--model-path", type=str, default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
parser.add_argument("--prompt", type=str, default="A cat sitting on a windowsill watching rain")
parser.add_argument(
    "--baseline-dir",
    type=str,
    default=None,
    help="FP32 baseline capture dir for three-way comparison",
)
parser.add_argument(
    "--expected-dir", type=str, default=None, help="BF16 CPU capture dir for three-way comparison"
)
parser.add_argument(
    "--prompt-hash",
    type=str,
    default="prompt_0",
    help="Prompt hash subdirectory to load for comparison",
)
args = parser.parse_args()

_tp = 8
env_profiles.apply(env_profiles.WAN22_T2V, env_profiles.thread_limits(_tp))


def main():
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    modules = [m.strip() for m in args.modules.split(",")]
    print(f"Model: {args.model_path}")
    print(f"Capture modules: {modules}")
    print(f"Capture dir: {args.capture_dir}")

    stage_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wan22_stage.yaml")

    omni = Omni(
        model=args.model_path,
        stage_configs_path=stage_cfg,
        stage_init_timeout=1200,
        init_timeout=1200,
        model_config={
            "tensor_capture": {
                "modules": modules,
                "capture_dir": args.capture_dir,
            },
        },
    )

    params = OmniDiffusionSamplingParams(
        height=128,
        width=208,
        num_frames=5,
        num_inference_steps=args.num_steps,
        guidance_scale=1.0,
        seed=args.seed,
    )

    print(f"\nRunning {args.num_steps}-step denoising with capture enabled...")
    omni.generate({"prompt": args.prompt}, params)
    print("Generation complete.")

    # Show captured files
    print(f"\nCaptures saved to: {args.capture_dir}/")
    if os.path.exists(args.capture_dir):
        for root, dirs, files in os.walk(args.capture_dir):
            level = root.replace(args.capture_dir, "").count(os.sep)
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = "  " * (level + 1)
            for f in sorted(files)[:5]:
                print(f"{sub_indent}{f}")
            if len(files) > 5:
                print(f"{sub_indent}... and {len(files) - 5} more")
    else:
        print("  (no captures found)")

    # Three-way comparison if baseline and expected dirs are provided
    if args.baseline_dir and args.expected_dir:
        from pathlib import Path

        from vllm_omni_neuron.diffusion.accuracy.step_capture import DiffusionCaptureWriter
        from vllm_omni_neuron.diffusion.accuracy.step_compare import (
            compare_denoising_runs,
            print_stepwise_report,
        )

        print("\n" + "=" * 80)
        print("RUNNING THREE-WAY STEP COMPARISON")
        print("=" * 80)

        def _load_preds(capture_dir, prompt_hash):
            prompt_dir = Path(capture_dir) / prompt_hash
            step_dirs = sorted(prompt_dir.glob("step_*"))
            preds, timesteps = [], []
            for step_dir in step_dirs:
                meta = DiffusionCaptureWriter.read_step_meta(step_dir)
                if meta is None:
                    continue
                timesteps.append(meta["timestep"])
                tensors = DiffusionCaptureWriter.read_step_tensors(step_dir, tp_rank=0)
                if "noise_prediction" in tensors:
                    preds.append(tensors["noise_prediction"])
            return preds, timesteps

        baseline_preds, timesteps = _load_preds(args.baseline_dir, args.prompt_hash)
        expected_preds, _ = _load_preds(args.expected_dir, args.prompt_hash)
        actual_preds, _ = _load_preds(args.capture_dir, args.prompt_hash)

        print(
            f"Loaded {len(baseline_preds)} baseline, {len(expected_preds)} expected, "
            f"{len(actual_preds)} actual steps"
        )

        if baseline_preds and expected_preds and actual_preds:
            report = compare_denoising_runs(baseline_preds, expected_preds, actual_preds, timesteps)
            print_stepwise_report(report)
        else:
            print("Skipping comparison: missing noise_prediction tensors in one or more dirs")


if __name__ == "__main__":
    main()
