# vLLM Omni Neuron Plugin (Beta)

The vLLM Omni Neuron plugin is the recommended serving solution for diffusion and
multimodal generation models on AWS Trainium. It extends
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) with a Neuron
backend, providing the same vLLM Omni APIs and configuration you are already
familiar with.

## Supported models

Models listed below are tested end to end on Neuron hardware with correctness
validation and performance benchmarking.

| Family | Model | Generation | Instance | Correctness | Perf Test | Perf Tuning |
|---|---|---|---|---|---|---|
| Wan2.2 | T2V-A14B | Text to video | Trn2, Trn3 | ✅ | ✅ | In progress |
| Wan2.2 | I2V-A14B | Image to video | Trn2, Trn3 | ✅ | ✅ | In progress |

- **Correctness** — Accuracy validation passing (VBench and reference comparison)
- **Perf Test** — Performance benchmark tests tracked across releases
- **Perf Tuning** — Active optimization work being done

See the [model cards](docs/models/) for recommended configurations, accuracy
results, and known limitations.

> **Note:** This list includes only the models currently supported in the latest
> plugin version and does not include models under development or on the roadmap.

## Setup

Choose manual installation on a prepared Neuron host, or use the published vLLM
Neuron Deep Learning Container (DLC). Pip installs the declared dependencies.

Follow the [setup guide](docs/getting-started/setup-guide.md) for both flows,
host prerequisites, persistent caches, and installation verification.

On a host prepared with the [SDK 2.32 driver, runtime, and tools](docs/getting-started/setup-guide.md),
activate Python 3.13 and install the plugin and its Python dependencies:

```bash
git clone -b release-0.24.0.0.1.0 https://github.com/aws-neuron/vllm-omni-neuron.git
cd vllm-omni-neuron
python -m pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
```

See [Version](#version) for the compatibility matrix.

## Quick start

After completing the DLC setup, run a short text-to-video smoke test. For manual installation,
omit `docker exec vllm-omni-neuron` and replace `/workspace` with
`$VLLM_OMNI_HOME` in the commands below.

```bash
docker exec vllm-omni-neuron \
  python /workspace/plugin/examples/wan22/run.py \
  --dev \
  --output /workspace/output/wan-t2v-dev.mp4
```

Generate an 81-frame video at the default resolution of 480 × 832:

```bash
docker exec vllm-omni-neuron \
  python /workspace/plugin/examples/wan22/run.py \
  --output /workspace/output/wan-t2v.mp4
```

For image-to-video generation, online serving, custom prompts, output shapes,
and generation controls, use the tutorials and feature guide below.

## Features

Feature support is model-specific. The table reflects capabilities integrated
and tested on Neuron hardware for at least one supported model.

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-video (T2V) | ✅ |
| | Image-to-video (I2V) | ✅ |
| **Execution** | Offline `Omni.generate` | ✅ |
| | Online asynchronous generation API | ✅ |
| | Sequential multi-stage pipelines | ✅ |
| | Continuous request batching | Limited |
| | Stepwise streaming output | ❌ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Context parallelism (CP) | ✅ |
| | Megatron sequence parallelism (SP) | ✅ |
| | Classifier-free guidance parallelism (CFGP) | ✅ |
| | VAE patch parallelism | ✅ |
| **Compilation** | `torch.compile` | ✅ |
| | Compile cache (reusable Neuron artifacts) | ✅ |
| **Acceleration** | Cache-DiT | ✅ |
| **Precision** | BF16 | ✅ |
| | FP8 ROW_MX projections (Wan2.2 T2V, Trn3) | ✅ |

- **Supported** — integrated and tested for at least one model.
- **Limited** — concurrent requests are accepted, but the current Wan pipeline
  executes them serially rather than as a true request batch.
- **Not supported** — unavailable on the current Neuron model pipelines.

See the [features guide](docs/guides/features-guide.md) for model availability,
configuration, and trade-offs.

## Optimized NKI kernels

The plugin uses NKI kernels vendored in this repository for const-max ring
attention, fused adaptive LayerNorm with FP8 quantization, FP8 QKV
projection, and FP8 MLP. It also calls kernels from the installed NKI Library
for other attention, output-projection, and BF16 MLP paths. The vendored
kernels can be read and adapted for other models.

See the [vendored kernel reference](docs/model-dev/kernels/) for what each
documented kernel computes, its design decisions, and how to adapt it to
another model.

## Documentation

Full documentation sources are in [`docs/`](docs/). These docs are published to
[awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-omni-neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-omni-neuron/docs/index.html).

**Deploying supported models:**

| Folder | Contents |
|---|---|
| [`docs/getting-started/`](docs/getting-started/) | Installation, verification, and quickstarts |
| [`docs/tutorials/`](docs/tutorials/) | Generation, deployment, and optimization |
| [`docs/models/`](docs/models/) | Supported model configurations, accuracy, and limitations |
| [`docs/guides/`](docs/guides/) | Feature configuration and operational guidance |

**Implementing new models & optimizing performance:**

| Folder | Contents |
|---|---|
| [`docs/model-dev/`](docs/model-dev/) | Model onboarding, accuracy debugging, and kernel implementations |
| [`docs/design/`](docs/design/) | Plugin architecture and parallelism design |

Start with:

- [Set up vLLM Omni Neuron](docs/getting-started/setup-guide.md)
- [Offline Wan2.2 quickstart](docs/getting-started/quickstart-offline-serving-wan22.md)
- [Online Wan2.2 quickstart](docs/getting-started/quickstart-online-serving-wan22.md)
- [Wan2.2 deployment tutorial](docs/tutorials/tutorial-wan22-14b.md)
- [Feature guide](docs/guides/features-guide.md)

## Version

The vLLM Omni Neuron plugin version follows the format
`<vLLM Omni version>.<plugin version>` (e.g., `0.24.0.0.1.0` means vLLM Omni
0.24.0, plugin version 0.1.0).

| vLLM Omni Neuron Plugin | vLLM Omni Version | Neuron SDK | Instance Support | Status | Documentation |
|---|---|---|---|---|---|
| 0.24.0.0.1.0 (latest) | 0.24.0 | 2.32 | Trn2, Trn3 | Beta | [vLLM Omni Neuron docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-omni-neuron/docs/index.html) |

## Issues

Report bugs or request features: [GitHub Issues](https://github.com/aws-neuron/vllm-omni-neuron/issues)

## Code of Conduct

This project follows the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information, see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq)
or contact <opensource-codeofconduct@amazon.com>.

## License

Apache-2.0. See [LICENSE](LICENSE).
