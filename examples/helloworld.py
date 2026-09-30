# SPDX-License-Identifier: Apache-2.0
"""Helloworld example: run ColumnParallelLinear(128, 512) through the Omni
entrypoint with tensor_parallel_size=N.

Usage:
    VLLM_NEURON_CPU_MODE=1 python examples/helloworld.py          # CPU mode, TP=4
    python examples/helloworld.py                          # real Neuron hardware
    python examples/helloworld.py --stage-config path/to/stage.yaml
"""

import argparse
import json
import os
import tempfile

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

_DEFAULT_STAGE_CONFIG = os.path.join(os.path.dirname(__file__), "helloworld_stage.yaml")


def _make_model_dir() -> str:
    """Create a minimal local model dir so OmniDiffusion skips HF resolution."""
    d = tempfile.mkdtemp(prefix="helloworld_model_")
    with open(os.path.join(d, "model_index.json"), "w") as f:
        json.dump({"_class_name": "HelloWorldPipeline", "_diffusers_version": "0.0.0"}, f)
    os.makedirs(os.path.join(d, "transformer"), exist_ok=True)
    with open(os.path.join(d, "transformer", "config.json"), "w") as f:
        json.dump({}, f)
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage-config",
        type=str,
        default=_DEFAULT_STAGE_CONFIG,
        help="Path to stage YAML file (default: examples/helloworld_stage.yaml)",
    )
    args = parser.parse_args()

    model_dir = _make_model_dir()
    omni = Omni(
        model=model_dir,
        stage_configs_path=args.stage_config,
        stage_init_timeout=600,
        init_timeout=600,
    )
    result = omni.generate(
        {"prompt": "hello"},
        OmniDiffusionSamplingParams(),
    )
    print(f"generate() returned: {result}")


if __name__ == "__main__":
    main()
