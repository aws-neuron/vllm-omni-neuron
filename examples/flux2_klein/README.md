# FLUX.2 Klein Diffusers adapter proof of concept

This is an experimental **single-device, BF16, text-to-image** path for
`black-forest-labs/FLUX.2-klein-4B`. It delegates the complete denoising loop to
Diffusers, reuses vLLM-Omni request/output conversion, and compiles the Diffusers
text encoder, transformer, VAE decoder, scheduler step, and tensor preparation
regions through the native Neuron compiler. Tokenization and loop orchestration
remain in Python. Compilation uses static shapes (`dynamic=False`); a new
shape or scheduler step count can trigger additional compilation. It does not introduce a second Klein implementation.

## Environment

Use the plugin's supported vLLM/vLLM-Neuron/vLLM-Omni versions from
`requirements/core.txt` and a native Neuron SDK environment. Install the Diffusers
`support-neuron-vllm-omni` branch and this plugin's `add-diffusers-support` branch
editable into that environment. The hardware test uses installed vLLM-Omni
`0.29.0rc1`, vLLM `0.29.0`, and vLLM-Neuron `0.29.0.1.2.0+5f6fb4a`.
No vLLM-Omni source modifications are required by this PoC.

```bash
python -m pip install -e ~/diffusers
python -m pip install -e ~/vllm-omni-neuron
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_NEURON_BACKEND=neuron_native
export VLLM_NEURON_LIBTORCH_NEURONX_LITE=1
export VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND=1
export TORCH_CACHING_PRECOMPILE=0
export NEURON_VISIBLE_DEVICES=0
unset NEURON_RT_VISIBLE_CORES  # the plugin assigns this in each worker
export TORCH_NEURONX_NEFF_CACHE_DIR=/tmp/flux2-klein-neff-cache
```

Select a free core before running. Do not reuse a core allocated by another
workload. Cache persistence depends on where the cache directory is mounted.
The legacy XLA compilation route is deliberately rejected for this PoC.
The native route in the pinned vLLM-Neuron release uses libtorch-neuronx-lite.
Do not export NEURON_RT_VISIBLE_CORES in the Omni parent process; use
NEURON_VISIBLE_DEVICES and let the plugin assign cores to workers.

## Standalone baseline

```bash
NEURON_RT_VISIBLE_CORES=0 python ~/diffusers/examples/inference/flux2_klein_neuron.py \
    --compile-transformer --height 256 --width 256 --runs 2
```

## Offline serving through Omni

```bash
python examples/flux2_klein/run.py --height 256 --width 256 --runs 2
```

Startup warmup and the first request trigger compilation. The second uses the same seed and shape
with a fresh CPU generator. Compare both outputs with the standalone baseline;
record cold and warm timings separately. Larger shapes may require additional
memory or different attention kernels and are not validated by this PoC.

## Online serving (not yet validated)

```bash
vllm serve black-forest-labs/FLUX.2-klein-4B --omni \
    --diffusion-load-format diffusers --dtype bfloat16 \
    --tensor-parallel-size 1
```

```bash
curl http://localhost:8000/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d '{"model":"black-forest-labs/FLUX.2-klein-4B","prompt":"A golden retriever surfing a wave, photorealistic","size":"256x256","num_inference_steps":4,"guidance_scale":1.0,"seed":42,"response_format":"b64_json"}'
```

## Scope and validation

No TP, CP, CFG parallelism, DP, CPU offload, VAE tiling/slicing, quantization,
Cache-DiT, native tensor capture, or step-wise continuous batching. Keep
`enforce_eager` disabled: the upstream adapter rejects that option. CPU-mode
unit tests skip Neuron compilation; they do not establish hardware support.
Klein 9B and base variants are not validated.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_NEURON_CPU_MODE=1 \
    python -m pytest test/unit/test_diffusers_adapter.py
```

Before expanding support, run the relevant offline and online smoke checks on
free Neuron hardware in a matching serving environment. Check output
quality, same-seed reproducibility, repeated requests, and cache reuse across
process restarts. This is an experimental PoC, not a production release.

TP can be added later at adapter setup time by connecting the plugin's worker
mesh to Diffusers parallelism. Increasing `tensor_parallel_size` alone is
explicitly rejected and does not shard this pipeline.

### Hardware validation (2026-11-11)

The standalone Diffusers baseline ran the real checkpoint revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294` on a Trainium2 device:
256×256, BF16, four steps, seed 42, no TP. Two requests produced identical PNGs;
the first took 115.268 seconds including compilation and the second 0.495 seconds.
This baseline used torch-neuronx 2.14 with torch 2.14, separate from the pinned
serving runtime (torch 2.13 and libtorch-neuronx-lite 2.13).

The full offline path also completed successfully (exit code 0):
`Omni.generate` → Neuron worker/adapter → Diffusers → Neuron → PNG output.
It used the same checkpoint, dimensions, four steps, and seed above.

| Measurement | Result |
| --- | --- |
| Engine initialization including warmup | 58.97 s |
| First four-step request | 9.072 s |
| Second four-step request | 0.514 s |
| Same-seed repeat outputs | Pixel-identical, identical PNG hashes |
| PNG SHA256 | `c7f58456dade739ade79252fa54f979e7ce28d89de80832acf00004abd6c7d54` |

These are smoke-test timings, not a clean cold-start benchmark: compiler artifacts
from debugging runs were present. The standalone and Omni images are not pixel-identical
(mean absolute RGB difference 10.95/255); their runtime/compiler stacks differ.
The Omni image was visually checked and depicts the requested dog on a wave.
Numerical parity across runtime versions has not been established.

Runtime fixes include contiguous packed latents, static compilation shapes,
compiled dtype conversions, and a text-to-image startup warmup. The runtime still
emits an `aten::unfold` warning from prompt repetition, but both requests succeed.

23 adapter tests passed against the actual installed serving packages, including
a tiny real Klein pipeline, without stubbing serving imports. Four Diffusers prompt
layout regression cases passed, as did Ruff and whitespace checks.

Local run artifacts are in `/tmp/flux2-klein-poc-omni/`; the full log is
`/tmp/flux2-klein-poc-omni.log` and the tested environment launcher is
`/tmp/run_flux2_klein_omni_poc.sh`. The isolated serving dependencies live in
`/tmp/flux2-klein-serving-deps`; they do not modify the existing Python environment.
The run used torch `2.13.0+cpu`, libtorch-neuronx-lite
`2.13.0.1.0.3112+653bc48f`, and Transformers `5.17.0`.

HTTP serving, larger resolutions, TP, and cache reuse across restarts remain
outside the completed validation.
