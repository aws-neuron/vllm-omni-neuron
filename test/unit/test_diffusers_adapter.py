# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle tests; model loading and Neuron compilation are mocked."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from diffusers import Flux2KleinPipeline
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import (
    DiffusersAdapterPipeline,
)

from vllm_omni_neuron.diffusion.models.diffusers_adapter import NeuronDiffusersAdapterPipeline


@pytest.fixture
def config():
    return SimpleNamespace(
        model="black-forest-labs/FLUX.2-klein-4B",
        model_class_name="Flux2KleinPipeline",
        dtype=torch.bfloat16,
        parallel_config=SimpleNamespace(
            world_size=1, tensor_parallel_size=1, cfg_parallel_size=1, sequence_parallel_size=1
        ),
        diffusion_offload_config=None,
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        vae_use_slicing=False,
        vae_use_tiling=False,
        cache_backend=None,
        enforce_eager=False,
        quantization_config=None,
        diffusers_load_kwargs={},
        diffusers_call_kwargs={},
        diffusers_pipeline_cls=Flux2KleinPipeline,
        model_config={},
        enable_diffusion_pipeline_profiler=False,
        diffusion_attention_config=SimpleNamespace(default=None),
        output_type=None,
    )


@pytest.fixture
def adapter(config):
    result = NeuronDiffusersAdapterPipeline(od_config=config, device=torch.device("cpu"))
    result._pipeline = MagicMock(spec=Flux2KleinPipeline)
    result._pipeline.transformer = MagicMock()
    result._pipeline.text_encoder = None
    result._pipeline.vae = SimpleNamespace(decode=MagicMock())
    result._pipeline.scheduler = SimpleNamespace(step=MagicMock())
    result._pipeline._execution_device = torch.device("cpu")
    return result


def test_to_moves_wrapped_pipeline(adapter):
    assert adapter.to(torch.device("cpu"), torch.bfloat16) is adapter
    adapter._pipeline.to.assert_called_once_with(device=torch.device("cpu"), dtype=torch.bfloat16)
    assert adapter.device == torch.device("cpu")


def test_to_flushes_neuron_copies(adapter, monkeypatch):
    synchronize = MagicMock()
    monkeypatch.setattr(torch, "neuron", SimpleNamespace(synchronize=synchronize), raising=False)
    adapter._pipeline._execution_device = SimpleNamespace(type="neuron")
    adapter.to("neuron:0")
    synchronize.assert_called_once_with()


def test_compile_wraps_transformer_and_decoder_and_returns_adapter(adapter, monkeypatch):
    original = adapter._pipeline.transformer
    compiled = MagicMock()
    compile_model = MagicMock(return_value=compiled)
    monkeypatch.setattr(torch, "compile", compile_model)
    assert adapter.compile(backend="neuron", options={"compiler_workdir": "/tmp/test"}) is adapter
    assert compile_model.call_count == 6
    compile_model.assert_any_call(
        original,
        backend="neuron",
        options={"compiler_workdir": "/tmp/test"},
        fullgraph=True,
        dynamic=False,
    )
    assert adapter._pipeline.transformer is compiled


def test_native_attention(adapter):
    adapter._set_attention_backend()
    adapter._pipeline.transformer.set_attention_backend.assert_called_once_with("native")


def test_explicit_unsupported_attention_is_not_silently_ignored(adapter):
    adapter.od_config.diffusion_attention_config.default = SimpleNamespace(backend="FLASH_ATTN")
    with pytest.raises(ValueError, match="TORCH_SDPA"):
        adapter._set_attention_backend()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("enable_cpu_offload", True, "CPU offload"),
        ("enable_layerwise_offload", True, "CPU offload"),
        ("vae_use_tiling", True, "VAE"),
        ("vae_use_slicing", True, "VAE"),
        ("dtype", torch.float32, "bfloat16"),
        ("diffusers_load_kwargs", {"device_map": "balanced"}, "placement"),
        ("diffusers_load_kwargs", {"quantization_config": {}}, "unquantized"),
        ("model_config", {"tensor_capture": {"modules": ["blocks.0"]}}, "tensor_capture"),
    ],
)
def test_rejects_unsupported_options(config, field, value, match):
    setattr(config, field, value)
    with pytest.raises(ValueError, match=match):
        NeuronDiffusersAdapterPipeline(od_config=config, device=torch.device("cpu"))


def test_rejects_multiple_workers(config):
    config.parallel_config.world_size = 2
    with pytest.raises(ValueError, match="world_size=1"):
        NeuronDiffusersAdapterPipeline(od_config=config, device=torch.device("cpu"))


def test_preserves_upstream_tp_rejection(config):
    config.parallel_config.tensor_parallel_size = 2
    with pytest.raises(NotImplementedError, match="Tensor parallel"):
        NeuronDiffusersAdapterPipeline(od_config=config, device=torch.device("cpu"))


def test_rejects_other_pipeline_families(adapter, monkeypatch):
    monkeypatch.setattr(
        DiffusersAdapterPipeline, "_load_pipeline_from_pretrained", lambda *args: object()
    )
    with pytest.raises(ValueError, match="Flux2KleinPipeline"):
        adapter._load_pipeline_from_pretrained("other-model", {})


def test_forward_preserves_cpu_generator_and_wraps_output(adapter):
    generator = torch.Generator("cpu").manual_seed(42)
    adapter._accept_call_kwargs = {"prompt", "generator", "height", "width", "num_inference_steps"}
    sampling = SimpleNamespace(
        generator=generator,
        generator_device="cpu",
        seed=42,
        output_type=None,
        num_outputs_per_prompt=1,
        enable_frame_interpolation=False,
        height=256,
        width=256,
        num_inference_steps=4,
    )
    adapter._pipeline.return_value = SimpleNamespace(images=["generated-image"])
    result = adapter.forward(
        SimpleNamespace(prompts=["a cat"], sampling_params=sampling, is_dummy_run=lambda: False)
    )
    kwargs = adapter._pipeline.call_args.kwargs
    assert kwargs["generator"] is generator
    assert kwargs["prompt"] == "a cat"
    assert kwargs["height"] == kwargs["width"] == 256
    assert result.output == ["generated-image"]


def test_runner_selects_adapter_without_native_loader(config, monkeypatch):
    from vllm_omni_neuron.diffusion.worker import diffusion_model_runner as runner_module

    runner = object.__new__(runner_module.NeuronDiffusionModelRunner)
    runner.od_config = config
    runner.device = torch.device("cpu")
    lifecycle = []

    def load_weights(adapter):
        lifecycle.append("load")
        adapter._pipeline = MagicMock(spec=Flux2KleinPipeline)
        adapter._pipeline._execution_device = torch.device("cpu")
        adapter._pipeline.to.side_effect = lambda **kwargs: lifecycle.append("move")

    monkeypatch.setattr(NeuronDiffusersAdapterPipeline, "load_weights", load_weights)
    native_loader = MagicMock(side_effect=AssertionError("native loader must not run"))
    monkeypatch.setattr(runner_module, "DiffusersPipelineLoader", native_loader)
    compile_adapter = MagicMock(side_effect=AssertionError("CPU mode must not compile for Neuron"))
    monkeypatch.setattr(NeuronDiffusersAdapterPipeline, "compile", compile_adapter)
    runner.load_model(load_format="diffusers")
    assert lifecycle == ["load", "move"]
    assert isinstance(runner.pipeline, NeuronDiffusersAdapterPipeline)
    assert runner.cache_backend is None


@pytest.mark.parametrize("compile_transformer", [False, True])
def test_tiny_klein_cpu_matches_direct_diffusers(config, monkeypatch, compile_transformer):
    import inspect

    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2Transformer2DModel,
    )

    torch.manual_seed(0)
    pipe = Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        text_encoder=None,
        tokenizer=None,
        transformer=Flux2Transformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=16,
            timestep_guidance_channels=256,
            axes_dims_rope=[4, 4, 4, 4],
            guidance_embeds=False,
        ),
        vae=AutoencoderKLFlux2(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
        ),
        is_distilled=True,
    ).to(dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    adapter = NeuronDiffusersAdapterPipeline(od_config=config, device=torch.device("cpu"))
    monkeypatch.setattr(adapter, "_load_pipeline_from_pretrained", lambda *args: pipe)
    adapter.load_weights()
    adapter.to("cpu")
    assert adapter._accept_call_kwargs == set(inspect.signature(pipe.__call__).parameters)
    assert not pipe.transformer.training
    embeddings = torch.randn(1, 4, 16, dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = pipe(
            prompt_embeds=embeddings,
            height=8,
            width=8,
            num_inference_steps=2,
            guidance_scale=1.0,
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="pt",
        ).images
    sampling = SimpleNamespace(
        prompt_embeds=embeddings,
        height=8,
        width=8,
        num_inference_steps=2,
        guidance_scale=1.0,
        generator=None,
        generator_device="cpu",
        seed=42,
        output_type="pt",
        num_outputs_per_prompt=1,
        enable_frame_interpolation=False,
    )
    if compile_transformer:
        adapter.compile(backend="eager")
    actual = adapter.forward(
        SimpleNamespace(
            prompts=[{"prompt": None}], sampling_params=sampling, is_dummy_run=lambda: False
        )
    ).output
    assert actual.shape == (1, 3, 8, 8)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_startup_warmup_is_text_to_image(adapter, monkeypatch):
    monkeypatch.setattr(
        DiffusersAdapterPipeline,
        "_build_call_kwargs",
        lambda *args: {"prompt": "dummy", "image": object(), "height": 512, "width": 512},
    )
    kwargs = adapter._build_call_kwargs(SimpleNamespace(is_dummy_run=lambda: True))
    assert "image" not in kwargs
    assert kwargs["height"] == kwargs["width"] == 256
    assert kwargs["guidance_scale"] == 1.0


def test_user_image_editing_is_rejected(adapter, monkeypatch):
    monkeypatch.setattr(
        DiffusersAdapterPipeline, "_build_call_kwargs", lambda *args: {"image": object()}
    )
    with pytest.raises(ValueError, match="text-to-image"):
        adapter._build_call_kwargs(SimpleNamespace(is_dummy_run=lambda: False))


def test_compiled_latent_packing_materializes_contiguous_output(adapter):
    adapter._pipeline._pack_latents = Flux2KleinPipeline._pack_latents
    adapter.compile(backend="eager")
    latents = torch.randn(1, 4, 4, 4)
    expected = Flux2KleinPipeline._pack_latents(latents)
    actual = adapter._pipeline._pack_latents(latents)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected)
