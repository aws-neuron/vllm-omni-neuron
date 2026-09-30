# SPDX-License-Identifier: Apache-2.0
"""NeuronWanPipeline — Wan 2.2 T2V pipeline for Neuron hardware.

Uses our verified custom TP WanTransformer3DModel with raw nn.Parameters.

Inherits forward(), encode_prompt(), prepare_latents(), check_inputs(),
predict_noise() from vllm-omni's Wan22Pipeline. Overrides __init__()
(Neuron device handling + transformer init) and load_weights()
(path-based TP-sharded loading).
"""

import json
import logging
import math
import os
import time
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoTokenizer
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from vllm_omni.diffusion.models.wan2_2 import Wan22Pipeline
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    get_wan22_post_process_func,
    get_wan22_pre_process_func,  # noqa: F401 — required by registry
    load_transformer_config,
)

from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
    NeuronAutoencoderKLWan,
)
from vllm_omni_neuron.diffusion.distributed.cfg_parallel import NeuronCFGParallelMixin
from vllm_omni_neuron.diffusion.models.umt5_encoder.umt5_encoder import (
    NeuronTextEncoderWrapper,
)
from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
from vllm_omni_neuron.diffusion.quantization.comfy_fp8_checkpoint import (
    FP8_CHECKPOINT_FILES,
    FP8_CHECKPOINT_SUBDIR,
)
from vllm_omni_neuron.lite_compat import is_lite_runtime

logger = logging.getLogger(__name__)

# CFG evaluates at most two encoder contexts per expert (the positive prompt and the
# negative prompt), so two live entries per expert is the whole working set. The cap is a
# backstop: if the context tensors turn out not to be reused across steps the cache would
# otherwise grow by one full K/V set per step (hundreds of MB each).
_MAX_CROSS_ATTENTION_KV_CACHE_ENTRIES_PER_EXPERT = 2


class _CrossAttentionKVCacheEntry(NamedTuple):
    """One expert's projected cross-attention K/V for one encoder context.

    ``context`` is kept both to verify the cache hit by tensor identity and to hold a
    reference to the keyed tensors, so their ``id()`` cannot be recycled by a different
    tensor while this entry is live.
    """

    context: tuple[torch.Tensor, ...]
    kv_cache: tuple[torch.Tensor, ...]


PIPELINE_REGISTRY = [
    {
        "model_arch": "Wan22Pipeline",
        "class_name": "NeuronWanPipeline",
        "pre_process_func_name": "get_wan22_pre_process_func",
        "post_process_func_name": "get_neuron_wan22_post_process_func",
    },
]


def get_neuron_wan22_post_process_func(od_config):
    """Pass the requested output type to Wan's upstream post-processor."""
    video_post_process = get_wan22_post_process_func(od_config)

    def post_process_func(video, output_type: str = "np", sampling_params=None):
        output_type = getattr(sampling_params, "output_type", None) or output_type
        return video_post_process(
            video,
            output_type=output_type,
            sampling_params=sampling_params,
        )

    return post_process_func


def _create_transformer_from_config(config: dict) -> WanTransformer3DModel:
    """Create Neuron WanTransformer3DModel from a diffusers config dict."""
    kwargs = {}
    for key in [
        "patch_size",
        "num_attention_heads",
        "attention_head_dim",
        "in_channels",
        "out_channels",
        "text_dim",
        "freq_dim",
        "ffn_dim",
        "num_layers",
        "cross_attn_norm",
        "eps",
        "image_dim",
        "added_kv_proj_dim",
        "rope_max_seq_len",
        "pos_embed_seq_len",
        # Not diffusers config fields — supplied via the stage config's model_config
        # overrides (engine_args.model_config), which are merged into the transformer
        # config above before this allowlist is applied.
        "quantization",
        "modules_to_not_convert",
        "tp_sequence_parallel",
    ]:
        if key in config:
            val = config[key]
            if key == "patch_size":
                val = tuple(val)
            kwargs[key] = val
    return WanTransformer3DModel(**kwargs)


def _compile_lite_helper(helper, *args, **kwargs):
    options = {**(kwargs.pop("options", {}) or {}), "model_name": f"wan{helper.__name__}"}
    kwargs = {"dynamic": False, "fullgraph": True, **kwargs}
    return torch.compile(helper, *args, options=options, **kwargs)


class NeuronFlowUniPCMultistepScheduler(FlowUniPCMultistepScheduler):
    """UniPC scheduler with device-resident Lite schedules and compiled step math."""

    @staticmethod
    def _unipc_coefficients(sigmas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute predictor and corrector scalar inputs for every scheduler step."""
        lambdas = torch.log(1 - sigmas) - torch.log(sigmas)
        h = lambdas[1:] - lambdas[:-1]
        h_phi = torch.expm1(-h)
        ones = torch.ones_like(h)
        zeros = torch.zeros_like(h)

        predictor_rk = (lambdas[:-2] - lambdas[1:-1]) / h[1:]
        predictor_rk = torch.cat((ones[:1], predictor_rk[:-1], ones[:1]))[: h.shape[0]]
        predictor_rho = torch.cat((zeros[:1], torch.full_like(predictor_rk[1:-1], 0.5), zeros[:1]))[
            : h.shape[0]
        ]
        predictor_coefficients = torch.stack(
            (
                sigmas[1:],
                sigmas[:-1],
                1 - sigmas[1:],
                h_phi,
                predictor_rk,
                predictor_rho,
            ),
            dim=1,
        )

        if h.shape[0] == 1:
            corrector_coefficients = torch.zeros(
                (1, 7),
                dtype=sigmas.dtype,
                device=sigmas.device,
            )
        else:
            corrector_h = h[:-1]
            corrector_h_phi = h_phi[:-1]
            corrector_rk = (lambdas[:-3] - lambdas[1:-2]) / corrector_h[1:]
            corrector_rk = torch.cat((ones[:1], corrector_rk))[: corrector_h.shape[0]]
            hh = -corrector_h[1:]
            h_phi_k = corrector_h_phi[1:] / hh - 1
            b0 = h_phi_k / corrector_h_phi[1:]
            h_phi_k = h_phi_k / hh - 0.5
            b1 = h_phi_k * 2 / corrector_h_phi[1:]
            rho_history = (b1 - b0) / (corrector_rk[1:] - 1)
            rho_current = b0 - rho_history
            rho_history = torch.cat((zeros[:1], rho_history))[: corrector_h.shape[0]]
            rho_current = torch.cat((torch.full_like(zeros[:1], 0.5), rho_current))[
                : corrector_h.shape[0]
            ]
            corrector_coefficients = torch.stack(
                (
                    sigmas[1:-1],
                    sigmas[:-2],
                    1 - sigmas[1:-1],
                    corrector_h_phi,
                    corrector_rk,
                    rho_history,
                    rho_current,
                ),
                dim=1,
            )
        return predictor_coefficients, corrector_coefficients

    @staticmethod
    def _unipc_schedule(
        parameters: torch.Tensor, num_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_max, sigma_min, shift, exp_mu, num_train_timesteps, sigma_last, dynamic = (
            parameters.unbind()
        )
        index = torch.arange(num_steps, dtype=torch.float32, device=parameters.device)
        sigmas = sigma_max - (sigma_max - sigma_min) * index / num_steps
        static_sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        dynamic_sigmas = exp_mu / (exp_mu + sigmas.reciprocal() - 1)
        sigmas = torch.where(dynamic != 0, dynamic_sigmas, static_sigmas)
        timesteps = (sigmas * num_train_timesteps).to(torch.int64)
        sigmas = torch.cat((sigmas, sigma_last.unsqueeze(0)))
        predictor_coefficients, corrector_coefficients = (
            NeuronFlowUniPCMultistepScheduler._unipc_coefficients(sigmas)
        )
        return timesteps, sigmas, predictor_coefficients, corrector_coefficients

    @staticmethod
    def _unipc_custom_schedule(
        sigmas: torch.Tensor, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shift, exp_mu, num_train_timesteps, sigma_last, dynamic = parameters.unbind()
        static_sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        dynamic_sigmas = exp_mu / (exp_mu + sigmas.reciprocal() - 1)
        sigmas = torch.where(dynamic != 0, dynamic_sigmas, static_sigmas)
        timesteps = (sigmas * num_train_timesteps).to(torch.int64)
        sigmas = torch.cat((sigmas, sigma_last.unsqueeze(0)))
        predictor_coefficients, corrector_coefficients = (
            NeuronFlowUniPCMultistepScheduler._unipc_coefficients(sigmas)
        )
        return timesteps, sigmas, predictor_coefficients, corrector_coefficients

    @staticmethod
    def _unipc_convert(
        model_output: torch.Tensor,
        sample: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return (sample - coefficients[0] * model_output).to(sample.dtype)

    @staticmethod
    def _unipc_predictor(
        sample: torch.Tensor,
        model_output: torch.Tensor,
        previous_model_output: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        sigma_t, sigma_s0, alpha_t, h_phi, rk, rho = coefficients.unbind()
        rho = rho.to(sample.dtype)
        x_t = sigma_t / sigma_s0 * sample - alpha_t * h_phi * model_output
        history = (previous_model_output - model_output) / rk
        return (x_t - alpha_t * h_phi * (rho * history)).to(sample.dtype)

    @staticmethod
    def _unipc_corrector(
        last_sample: torch.Tensor,
        this_model_output: torch.Tensor,
        previous_model_output: torch.Tensor,
        older_model_output: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        sigma_t, sigma_s0, alpha_t, h_phi, rk, rho_history, rho_current = coefficients.unbind()
        rho_history = rho_history.to(last_sample.dtype)
        rho_current = rho_current.to(last_sample.dtype)
        x_t = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi * previous_model_output
        history = (older_model_output - previous_model_output) / rk
        current = this_model_output - previous_model_output
        return (x_t - alpha_t * h_phi * (rho_history * history + rho_current * current)).to(
            last_sample.dtype
        )

    def compile(self, *args, **kwargs):
        if not is_lite_runtime():
            return self
        if not (
            self.predict_x0
            and self.config.prediction_type == "flow_prediction"
            and not self.config.thresholding
            and self.config.solver_order == 2
            and self.config.solver_type == "bh2"
            and self.config.lower_order_final
            and self.solver_p is None
        ):
            raise NotImplementedError("Lite compiled UniPC only supports the Wan scheduler")

        for attr, helper in (
            ("_compiled_schedule", self._unipc_schedule),
            ("_compiled_custom_schedule", self._unipc_custom_schedule),
            ("_compiled_convert_model_output", self._unipc_convert),
            ("_compiled_predictor", self._unipc_predictor),
            ("_compiled_corrector", self._unipc_corrector),
        ):
            setattr(self, attr, _compile_lite_helper(helper, *args, **kwargs))
        return self

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        shift: float | None = None,
    ) -> None:
        if not is_lite_runtime():
            return super().set_timesteps(
                num_inference_steps, device=device, sigmas=sigmas, mu=mu, shift=shift
            )

        target_device = torch.device(device) if device is not None else None
        compiled_schedule = getattr(self, "_compiled_schedule", None)
        compiled_custom_schedule = getattr(self, "_compiled_custom_schedule", None)
        if (
            target_device is None
            or target_device.type == "cpu"
            or compiled_schedule is None
            or compiled_custom_schedule is None
        ):
            return super().set_timesteps(
                num_inference_steps, device=device, sigmas=sigmas, mu=mu, shift=shift
            )

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("Must pass a value for `mu` when `use_dynamic_shifting` is True")
        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = self.sigma_min
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0.0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be 'zero' or 'sigma_min', "
                f"got {self.config.final_sigmas_type}"
            )

        dynamic = self.config.use_dynamic_shifting
        shift_value = self.config.shift if shift is None else shift
        exp_mu = math.exp(mu) if dynamic else 1.0
        if sigmas is None:
            if num_inference_steps is None:
                raise ValueError("num_inference_steps is required when sigmas are not provided")
            parameters = torch.tensor(
                [
                    self.sigma_max,
                    self.sigma_min,
                    shift_value,
                    exp_mu,
                    self.config.num_train_timesteps,
                    sigma_last,
                    float(dynamic),
                ],
                dtype=torch.float32,
                device=target_device,
            )
            outputs = compiled_schedule(parameters, num_inference_steps)
        else:
            sigma_tensor = torch.tensor(sigmas, dtype=torch.float32, device=target_device)
            parameters = torch.tensor(
                [
                    shift_value,
                    exp_mu,
                    self.config.num_train_timesteps,
                    sigma_last,
                    float(dynamic),
                ],
                dtype=torch.float32,
                device=target_device,
            )
            outputs = compiled_custom_schedule(sigma_tensor, parameters)

        (
            self.timesteps,
            self.sigmas,
            self._predictor_coefficients,
            self._corrector_coefficients,
        ) = outputs
        self.num_inference_steps = self.timesteps.shape[0]
        self.model_outputs = [None] * self.config.solver_order
        self.timestep_list = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = self._step_index = self._begin_index = None

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if (
            getattr(self, "_compiled_convert_model_output", None) is None
            or getattr(self, "_predictor_coefficients", None) is None
            or sample is None
            or model_output.dtype != sample.dtype
            or args
            or kwargs
        ):
            return super().convert_model_output(
                model_output,
                *args,
                sample=sample,
                **kwargs,
            )
        return self._run_compiled_convert_model_output(model_output, sample=sample)

    def _run_compiled_convert_model_output(
        self,
        model_output: torch.Tensor,
        *,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        assert self.step_index is not None
        coefficients = self._predictor_coefficients[self.step_index, 1:2]
        return self._compiled_convert_model_output(model_output, sample, coefficients)

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor | None = None,
        order: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        model_outputs = getattr(self, "model_outputs", None)
        current_model_output = model_outputs[-1] if model_outputs else None
        if (
            getattr(self, "_compiled_predictor", None) is None
            or getattr(self, "_predictor_coefficients", None) is None
            or sample is None
            or order not in (1, 2)
            or current_model_output is None
            or current_model_output.dtype != sample.dtype
            or args
            or kwargs
        ):
            return super().multistep_uni_p_bh_update(
                model_output,
                *args,
                sample=sample,
                order=order,
                **kwargs,
            )
        return self._run_compiled_predictor(sample=sample, order=order)

    def _run_compiled_predictor(
        self,
        *,
        sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        compiled = self._compiled_predictor
        assert self.step_index is not None

        current_model_output = self.model_outputs[-1]
        previous_model_output = (
            self.model_outputs[-2]
            if order == 2
            else current_model_output.view_as(current_model_output)
        )
        assert current_model_output is not None and previous_model_output is not None

        coefficients = self._predictor_coefficients[self.step_index]
        return compiled(sample, current_model_output, previous_model_output, coefficients)

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor | None = None,
        this_sample: torch.Tensor | None = None,
        order: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        model_outputs = getattr(self, "model_outputs", None)
        previous_model_output = model_outputs[-1] if model_outputs else None
        if (
            getattr(self, "_compiled_corrector", None) is not None
            and getattr(self, "_corrector_coefficients", None) is not None
            and last_sample is not None
            and this_sample is not None
            and order in (1, 2)
            and this_model_output.dtype == last_sample.dtype
            and previous_model_output is not None
            and previous_model_output.dtype == last_sample.dtype
            and not args
            and not kwargs
        ):
            return self._run_compiled_corrector(
                this_model_output=this_model_output,
                last_sample=last_sample,
                order=order,
            )

        # Under Lite, rebuild the solve result in ``x``'s dtype rather than letting the
        # corrector's trailing ``.to(device).to(x.dtype)`` do it: Lite reaches a narrower
        # dtype only by building a new tensor, never by converting one. Leaving the result
        # float32 promotes ``x_t``, whose own closing cast is the same unsupported
        # narrowing, so float32 latents would reach the next step.
        _orig = torch.linalg.solve
        effective_last_sample = last_sample
        if effective_last_sample is None and len(args) > 1:
            effective_last_sample = args[1]
        rebuild_dtype = (
            effective_last_sample.dtype
            if is_lite_runtime() and isinstance(effective_last_sample, torch.Tensor)
            else None
        )

        def _solve_matching_dtype(A, B, **kw):
            out = _orig(A, B.to(A.dtype), **kw)
            if rebuild_dtype is None:
                return out
            return torch.tensor(list(out.unbind()), dtype=rebuild_dtype, device=out.device)

        torch.linalg.solve = _solve_matching_dtype
        try:
            return super().multistep_uni_c_bh_update(
                this_model_output,
                *args,
                last_sample=last_sample,
                this_sample=this_sample,
                order=order,
                **kwargs,
            )
        finally:
            torch.linalg.solve = _orig

    def _run_compiled_corrector(
        self,
        *,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        assert self.step_index is not None

        previous_model_output = self.model_outputs[-1]
        older_model_output = (
            self.model_outputs[-2]
            if order == 2
            else previous_model_output.view_as(previous_model_output)
        )
        assert previous_model_output is not None and older_model_output is not None

        coefficients = self._corrector_coefficients[self.step_index - 1]
        if order == 1 and self.step_index != 1:
            coefficients = torch.cat((coefficients[:4], self._corrector_coefficients[0, 4:]))
        return self._compiled_corrector(
            last_sample,
            this_model_output,
            previous_model_output,
            older_model_output,
            coefficients,
        )


class NeuronWanPipeline(NeuronCFGParallelMixin, Wan22Pipeline):
    """Wan 2.2 T2V pipeline for Neuron.

    Subclasses vllm-omni's Wan22Pipeline, inheriting forward(), encode_prompt(),
    prepare_latents(), check_inputs(), predict_noise(). Overrides only __init__()
    for Neuron device handling and load_weights() for TP-sharded weight routing.

    Mixes in NeuronCFGParallelMixin (ahead of Wan22Pipeline in the MRO) so the
    base denoise loop's ``predict_noise_maybe_with_cfg`` resolves to the Neuron
    CFG-parallel path: when ``cfg_parallel_size > 1`` each cfg-parallel replica runs
    one CFG branch and a fullgraph-compiled ``all_gather`` + combine fuses them. With
    ``cfg_parallel_size == 1`` the override defers to the base (sequential CFG), so
    single-replica behavior is unchanged. Wan22Pipeline already inherits
    CFGParallelMixin, so the C3 linearization is well-defined.
    """

    @staticmethod
    def _cfg_combine(
        positive_noise_pred: torch.Tensor,
        negative_noise_pred: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return negative_noise_pred + scale[0] * (positive_noise_pred - negative_noise_pred)

    def combine_cfg_noise(
        self,
        positive_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        negative_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        true_cfg_scale: float,
        cfg_normalize: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        compiled = getattr(self, "_compiled_cfg_combine", None)
        if compiled is None or cfg_normalize:
            return super().combine_cfg_noise(
                positive_noise_pred,
                negative_noise_pred,
                true_cfg_scale,
                cfg_normalize,
            )
        if isinstance(positive_noise_pred, tuple) or isinstance(negative_noise_pred, tuple):
            if not (
                isinstance(positive_noise_pred, tuple)
                and isinstance(negative_noise_pred, tuple)
                and len(positive_noise_pred) == len(negative_noise_pred) == 1
                and isinstance(positive_noise_pred[0], torch.Tensor)
                and isinstance(negative_noise_pred[0], torch.Tensor)
            ):
                return super().combine_cfg_noise(
                    positive_noise_pred,
                    negative_noise_pred,
                    true_cfg_scale,
                    cfg_normalize,
                )
            positive_noise_pred = positive_noise_pred[0]
            negative_noise_pred = negative_noise_pred[0]
        scale = torch.tensor([true_cfg_scale], dtype=torch.float32).to(positive_noise_pred.device)
        return compiled(positive_noise_pred, negative_noise_pred, scale)

    def __init__(self, *, od_config, prefix: str = ""):
        # Skip Wan22Pipeline.__init__ (GPU-centric). Call nn.Module directly.
        nn.Module.__init__(self)
        self.od_config = od_config
        self.device = get_local_device()

        model = od_config.model
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        local_files_only = os.path.isdir(model)

        # TODO: Support expand_timesteps for TI2V models (read from model_index.json)
        self.expand_timesteps = False
        self.has_transformer_2 = False
        if local_files_only:
            model_index_path = os.path.join(model, "model_index.json")
            if os.path.exists(model_index_path):
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.expand_timesteps = model_index.get("expand_timesteps", False)
            transformer_2_path = os.path.join(model, "transformer_2")
            self.has_transformer_2 = os.path.exists(transformer_2_path)
        else:
            try:
                from huggingface_hub import hf_hub_download

                model_index_path = hf_hub_download(repo_id=model, filename="model_index.json")
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.expand_timesteps = model_index.get("expand_timesteps", False)
                    transformer_2_info = model_index.get("transformer_2", [None, None])
                    self.has_transformer_2 = transformer_2_info[0] is not None
            except Exception:
                pass

        self.boundary_ratio = od_config.boundary_ratio

        # Determine which transformers to load
        load_transformer = self.boundary_ratio != 1.0 if self.boundary_ratio is not None else True
        load_transformer_2 = self.has_transformer_2 and (
            self.boundary_ratio != 0.0 if self.boundary_ratio is not None else True
        )

        tp_group = dist.group.WORLD if dist.is_initialized() else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only
        )
        self.text_encoder = NeuronTextEncoderWrapper(
            model_path=model,
            dtype=dtype,
            tp_group=tp_group,
        )

        # VAE patch parallelism: when vae_patch_parallel_size > 1, VAE decode is
        # sharded across the VAE-parallel group. The distributed executor's
        # gather/all_reduce/broadcast run over that group, so every member builds
        # and runs the VAE in lockstep; the merged frames land on rank 0. Outside
        # patch-parallel mode the VAE is unsharded and lives on rank 0 only, to
        # avoid redundant NEFF compilation.
        #
        # The requested vae_patch_parallel_size is clamped to the world size.
        # Surplus ranks join collectives (keeping the group in lockstep) but
        # decode nothing.
        vae_pp_size = int(getattr(od_config.parallel_config, "vae_patch_parallel_size", 1) or 1)
        if dist.is_initialized() and vae_pp_size > 1:
            requested = vae_pp_size
            vae_pp_size = min(vae_pp_size, dist.get_world_size())
            logger.info(
                "VAE patch parallel: requested=%d, actual group size=%d",
                requested,
                vae_pp_size,
            )
        self.vae_patch_parallel = dist.is_initialized() and vae_pp_size > 1
        # Rank that holds the final decoded frames and writes perf metrics.
        self.is_output_rank = not dist.is_initialized() or dist.get_rank() == 0

        # Build the VAE-parallel subgroup. new_group() is collective, so ALL
        # ranks must call it (even non-members, who receive a handle they never
        # use). Non-patch-parallel mode needs no subgroup.
        self._vae_group = None
        if self.vae_patch_parallel:
            vae_ranks = list(range(vae_pp_size))
            self._vae_group = dist.new_group(ranks=vae_ranks)
            self.is_vae_rank = dist.get_rank() in vae_ranks
        else:
            self.is_vae_rank = self.is_output_rank

        if self.is_vae_rank:
            vae_cls = (
                DistributedAutoencoderKLWan if self.vae_patch_parallel else NeuronAutoencoderKLWan
            )
            self.vae = vae_cls.from_pretrained(
                model,
                subfolder="vae",
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )
            if self.vae_patch_parallel:
                # Patch parallelism dispatches on tiled_decode(), which requires
                # tiling. Bind the executor to the VAE subgroup and record the
                # parallel degree (== subgroup size).
                self.vae.init_distributed(group=self._vae_group)
                self.vae.set_parallel_size(vae_pp_size)

        # Initialize transformers (weights loaded via load_weights later)
        model_config_overrides = dict(od_config.model_config or {})
        self._comfyui_fp8_model_path = model_config_overrides.pop("comfyui_fp8_model_path", None)

        if load_transformer:
            tf_config = load_transformer_config(model, "transformer", local_files_only)
            tf_config.update(model_config_overrides)
            self.transformer = _create_transformer_from_config(tf_config)
        else:
            self.transformer = None  # type: ignore[assignment]

        if load_transformer_2:
            tf2_config = load_transformer_config(model, "transformer_2", local_files_only)
            tf2_config.update(model_config_overrides)
            self.transformer_2 = _create_transformer_from_config(tf2_config)
        else:
            self.transformer_2 = None  # type: ignore[assignment]

        # Store active transformer config
        if self.transformer is not None:
            self.transformer_config = self.transformer.config
        elif self.transformer_2 is not None:
            self.transformer_config = self.transformer_2.config
        else:
            raise RuntimeError("No transformer loaded")

        # Scheduler
        flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0
        self.scheduler = NeuronFlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
            solver_order=2,
        )
        # vllm-omni 0.24's inherited forward() reads self._sample_solver /
        # self._flow_shift (and rebuilds the scheduler when a request overrides
        # them). Mirror the base __init__ so the attributes exist; keeping them in
        # sync with the scheduler above means the base leaves our Neuron scheduler
        # in place for the default unipc/flow_shift request.
        self._sample_solver = "unipc"
        self._flow_shift = flow_shift

        if self.is_vae_rank:
            self.vae_scale_factor_temporal = getattr(self.vae.config, "scale_factor_temporal", 4)
            self.vae_scale_factor_spatial = getattr(self.vae.config, "scale_factor_spatial", 8)
        else:
            self.vae_scale_factor_temporal = 4
            self.vae_scale_factor_spatial = 8

        # TODO: Implement custom warmup logic for Neuron HW
        self.skip_warmup = True

        # Private state used by inherited forward()
        self._guidance_scale = None
        self._guidance_scale_2 = None
        self._num_timesteps = None
        self._current_timestep = None

        # CPU offload of the DiT during VAE decode. Opt-in via vllm-omni's
        # ``enable_cpu_offload`` flag (default False)
        # When enabled on the VAE rank, the DiT transformer(s) are moved to CPU to
        # free device HBM for VAE decode, then restored afterwards. This is useful
        # on platforms with limited HBM per core.
        self._cpu_offload = bool(getattr(od_config, "enable_cpu_offload", False))

        # Performance metrics state
        self._text_encode_seconds: float | None = None
        self._last_perf_metrics: dict | None = None
        self._collect_perf = bool(os.environ.get("WORKLOAD_OUTPUT_RW"))
        # Only one expert is active at a time; context identity distinguishes positive
        # and negative CFG branches within that expert.
        self._cross_attention_kv_cache_expert_id: int | None = None
        self._cross_attention_kv_cache: dict[tuple, _CrossAttentionKVCacheEntry] = {}
        # id(expert) -> its compiled first-step full DiT graph. The graph returns
        # ``(noise_prediction, *cross_attention_kv_cache)``.
        self._compiled_cross_attention_kv_generators: dict[int, Callable[..., tuple]] = {}
        self._warned_unstable_cross_attention_context = False

    def to(self, *args, **kwargs):
        """Move text encoder, VAE, and transformer(s) to the target device."""
        for attr in ("text_encoder", "vae", "transformer", "transformer_2"):
            if attr == "vae" and not self.is_vae_rank:
                continue
            t = getattr(self, attr, None)
            if t is not None:
                t.to(*args, **kwargs)
        return self

    def _encode_prompt(
        self,
        prompts: list[str],
        max_sequence_length: int,
        num_videos_per_prompt: int,
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Tokenize, encode on Neuron, and tile a single batch of prompts.

        The compiled encoder graph zeros the padded positions of its output
        internally (equivalent to the old truncate-to-seq_len + zero-repad, but
        done in-graph on-device), so this method only tokenizes, runs the
        encoder, and tiles per ``num_videos_per_prompt`` — no CPU round-trip and
        no data-dependent slicing.
        """
        batch_size = len(prompts)
        cleaned = [self._prompt_clean(p) for p in prompts]
        text_inputs = self.tokenizer(
            cleaned,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask

        # The compiled encoder graph already zeros the padded positions of its
        # output (see NeuronTextEncoderWrapper._encode), which is the whole of
        # the old CPU truncate+repad post-process. So no D2H/H2D round-trip and
        # no data-dependent slice here — just tile per video and reshape.
        embeds = self.text_encoder(ids.to(target_device), mask.to(target_device)).last_hidden_state
        embeds = self._cast_device_dtype(embeds, target_dtype)
        _, seq_len, _ = embeds.shape
        if num_videos_per_prompt > 1:
            embeds = self._tile_prompt_embeds(embeds, num_videos_per_prompt)
        embeds = embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        return embeds

    def _tile_prompt_embeds(self, embeds: torch.Tensor, repeats: int) -> torch.Tensor:
        """Tile prompt embeddings in a compiled Neuron graph."""
        compiled = getattr(self, "_compiled_tile_prompt_embeds", None)
        key = (repeats, tuple(embeds.shape), embeds.dtype)
        if compiled is None or getattr(self, "_compiled_tile_prompt_embeds_key", None) != key:
            from vllm_neuron.envs import get_compile_backend_name

            def tile(tensor):
                return tensor.repeat(1, repeats, 1)

            compiled = torch.compile(
                tile,
                backend=get_compile_backend_name(),
                fullgraph=True,
                dynamic=False,
                options={"model_name": "wan_prompt_embed_tile"},
            )
            self._compiled_tile_prompt_embeds = compiled
            self._compiled_tile_prompt_embeds_key = key
        return compiled(embeds)

    def encode_prompt(self, *args, device=None, dtype=None, **kwargs):
        """Encode positive (and optionally negative) prompts for Neuron."""
        if self._collect_perf:
            t_start = time.perf_counter()

        target_device = device or self.device
        target_dtype = (
            (dtype or self.transformer.dtype)
            if self.transformer is not None
            else (dtype or self.text_encoder.dtype)
        )

        prompt = kwargs.get("prompt", args[0] if args else None)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        max_sequence_length = kwargs.get("max_sequence_length", 512)
        num_videos_per_prompt = kwargs.get("num_videos_per_prompt", 1)
        do_classifier_free_guidance = kwargs.get("do_classifier_free_guidance", True)
        negative_prompt = kwargs.get("negative_prompt", None)

        prompt_embeds = self._encode_prompt(
            prompt,
            max_sequence_length,
            num_videos_per_prompt,
            target_device,
            target_dtype,
        )

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            # TODO: we can persistent and reuse the embeds of the empty string
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            negative_prompt_embeds = self._encode_prompt(
                negative_prompt,
                max_sequence_length,
                num_videos_per_prompt,
                target_device,
                target_dtype,
            )

        if self._collect_perf:
            self._text_encode_seconds = time.perf_counter() - t_start
        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _cast_latents(latents: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return latents.to(dtype=dtype)

    @staticmethod
    def _randn_latents(shape, *, generator, device, dtype):
        """Draw latent noise while preserving caller generator state."""
        if isinstance(generator, list):
            if len(generator) != shape[0]:
                raise ValueError(
                    f"Generator list length {len(generator)} does not match batch size {shape[0]}"
                )
            return torch.cat(
                [
                    torch.randn(
                        (1, *shape[1:]),
                        generator=item,
                        device=device,
                        dtype=dtype,
                    )
                    for item in generator
                ]
            )
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """Create seeded float32 noise and cast it without leaving the target device."""
        transformer_dtype = (
            self.transformer.dtype if self.transformer is not None else torch.bfloat16
        )
        target_device = torch.device(device)
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if latents is None:
            latents = self._randn_latents(
                shape,
                generator=generator,
                device=target_device,
                dtype=torch.float32,
            )
        else:
            latents = latents.to(device=target_device)

        return self._cast_device_dtype(latents, transformer_dtype)

    def _cast_device_dtype(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Change a device-resident tensor's dtype, in-graph under Lite.

        Lite rejects an uncompiled dtype-changing copy on a Neuron tensor with
        ``Expected self.dtype() == dst.dtype() to be true, but got false``, so the
        cast has to run inside a compiled graph. The helper is compiled on first use
        so callers that compile components individually, rather than through
        ``compile()``, still get an in-graph cast instead of a hard failure.

        Args:
            tensor: Tensor to cast; may live on CPU or the Neuron device.
            dtype: Target dtype.

        Returns:
            ``tensor`` itself when it already has ``dtype``, else the cast tensor.
        """
        if tensor.dtype == dtype:
            return tensor
        if is_lite_runtime():
            compiled_cast = getattr(self, "_compiled_latent_cast", None)
            if compiled_cast is None:
                from vllm_neuron.envs import get_compile_backend_name

                compiled_cast = _compile_lite_helper(
                    self._cast_latents, backend=get_compile_backend_name()
                )
                self._compiled_latent_cast = compiled_cast
            return compiled_cast(tensor, dtype)
        return tensor.to(dtype=dtype)

    def _move_transformers_to(self, device: torch.device) -> None:
        """Move DiT transformer weights to the given device."""
        for t in (self.transformer, self.transformer_2):
            if t is not None:
                t.to(device)

    def _decode_latents(self, latents):
        """Unscale latents and decode to video frames on the Neuron device.

        Performs the latent unscaling (latents / latents_std + latents_mean)
        explicitly on-device in the pipeline's dtype before calling vae.decode,
        rather than delegating to the base class which would pull latents to CPU
        for the dtype cast.
        """
        latents_mean = torch.tensor(
            self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            self.vae.config.latents_std, device=latents.device, dtype=latents.dtype
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents = latents / latents_std + latents_mean
        return self.vae.decode(latents, return_dict=False)[0]

    def _release_scheduler_history(self) -> None:
        """Drop the UniPC solver history once the denoise loop is done with it.

        ``model_outputs`` and ``last_sample`` are read only inside ``scheduler.step``,
        and every ``forward`` calls ``set_timesteps`` (``pipeline_wan2_2.py:466``), which
        resets them anyway. After the loop they are dead references pinning
        ``solver_order + 1`` device latents, so drop them before the DiT offload.
        """
        self.scheduler.model_outputs = [None] * self.scheduler.config.solver_order
        self.scheduler.last_sample = None

    def _clear_cross_attention_kv_cache(self) -> None:
        self._cross_attention_kv_cache_expert_id = None
        self._cross_attention_kv_cache.clear()

    def _cross_attention_kv_entry_for(
        self, current_model, kwargs
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[int, ...],
        _CrossAttentionKVCacheEntry | None,
    ]:
        """Look up per-block K/V for one expert and encoder context.

        Wan 2.2 switches from the high-noise expert to the low-noise expert once, so only
        one expert's cache needs to stay live. Within that expert, sequential CFG uses two
        stable encoder contexts (positive and negative prompts).
        """
        expert_id = id(current_model)
        if self._cross_attention_kv_cache_expert_id != expert_id:
            self._cross_attention_kv_cache.clear()
            self._cross_attention_kv_cache_expert_id = expert_id

        context = tuple(
            tensor
            for tensor in (
                kwargs.get("encoder_hidden_states"),
                kwargs.get("encoder_hidden_states_image"),
            )
            if tensor is not None
        )
        # ``id()`` avoids a device sync. Keeping the context tensors in the entry prevents
        # their ids from being recycled while the entry is live.
        key = tuple(id(tensor) for tensor in context)
        entry = self._cross_attention_kv_cache.get(key)
        if entry is not None and all(
            cached is live for cached, live in zip(entry.context, context, strict=True)
        ):
            return context, key, entry
        return context, key, None

    def _store_cross_attention_kv(
        self,
        context: tuple[torch.Tensor, ...],
        key: tuple[int, ...],
        kv_cache: tuple[torch.Tensor, ...],
    ) -> None:
        self._cross_attention_kv_cache[key] = _CrossAttentionKVCacheEntry(context, kv_cache)
        if len(self._cross_attention_kv_cache) > _MAX_CROSS_ATTENTION_KV_CACHE_ENTRIES_PER_EXPERT:
            if not self._warned_unstable_cross_attention_context:
                self._warned_unstable_cross_attention_context = True
                logger.warning(
                    "Cross-attention K/V cache exceeded %d entries for one expert, so the "
                    "encoder context tensors are not being reused across denoise steps. The "
                    "cache is being re-projected every step; expect no speedup from it.",
                    _MAX_CROSS_ATTENTION_KV_CACHE_ENTRIES_PER_EXPERT,
                )
            # dicts preserve insertion order; discard the oldest context.
            self._cross_attention_kv_cache.pop(next(iter(self._cross_attention_kv_cache)))

    def _predict_noise_and_generate_cross_attention_kv(
        self,
        current_model,
        kwargs,
        context: tuple[torch.Tensor, ...],
        key: tuple[int, ...],
    ) -> torch.Tensor:
        """Run the first-step full DiT graph and remember the K/V it returns."""
        cache_generator = self._compiled_cross_attention_kv_generators.get(id(current_model))
        if cache_generator is None:
            cache_generator = current_model.forward_and_cache_cross_attention_kv
        result = cache_generator(**kwargs)
        if not isinstance(result, tuple):
            raise TypeError("cache-generating DiT must return a tuple")

        expected_outputs = current_model.cross_attention_kv_cache_size + 1
        if len(result) != expected_outputs:
            raise RuntimeError(
                f"cache-generating DiT returned {len(result)} outputs, "
                f"expected {expected_outputs} (noise prediction plus K/V)"
            )

        noise_pred, *kv_cache = result
        self._store_cross_attention_kv(context, key, tuple(kv_cache))
        return noise_pred

    def predict_noise(self, current_model=None, **kwargs):
        timestep = kwargs["timestep"]
        lite_runtime = is_lite_runtime()
        if lite_runtime:
            kwargs["timestep"] = (
                timestep[0].unsqueeze(0)
                if timestep.ndim == 1
                else torch.stack([timestep[0]] * timestep.shape[0])
            )
        else:
            kwargs["timestep"] = timestep.clone(memory_format=torch.contiguous_format)

        # Match the upstream default before the cache needs the selected expert.
        if current_model is None:
            current_model = self.transformer
        context, key, entry = self._cross_attention_kv_entry_for(current_model, kwargs)
        if entry is None:
            noise_pred = self._predict_noise_and_generate_cross_attention_kv(
                current_model,
                kwargs,
                context,
                key,
            )
        else:
            kwargs["cross_attention_kv_cache"] = entry.kv_cache
            noise_pred = super().predict_noise(current_model=current_model, **kwargs)
        if lite_runtime:
            while isinstance(noise_pred, tuple):
                if not noise_pred:
                    raise RuntimeError("Lite transformer returned an empty output tuple")
                noise_pred = noise_pred[0]
            if not isinstance(noise_pred, torch.Tensor):
                raise TypeError("Lite transformer output must resolve to a tensor")
        return noise_pred

    def scheduler_step_maybe_with_cfg(
        self,
        noise_pred,
        t,
        latents,
        do_true_cfg,
        per_request_scheduler=None,
        generator=None,
    ):
        latents = super().scheduler_step_maybe_with_cfg(
            noise_pred,
            t,
            latents,
            do_true_cfg,
            per_request_scheduler=per_request_scheduler,
            generator=generator,
        )
        if is_lite_runtime():
            # Lite's D2D copy waits for the source future, providing queue
            # backpressure without transferring data to the host.
            outputs = latents if isinstance(latents, tuple) else (latents,)
            for output in outputs:
                source = output.view(-1)[:1]
                torch.empty_like(source).copy_(source)
        return latents

    def forward(self, req):
        """Run pipeline with VAE decode and per-stage performance timing.

        Wraps the parent forward() with timing instrumentation to capture
        text encoding, denoising, and VAE decode latencies. Metrics are written
        to a JSON file at ``$WORKLOAD_OUTPUT_RW/metrics/pipeline_perf_metrics.json``
        (or ``/tmp/metrics/pipeline_perf_metrics.json`` as fallback) for downstream
        consumption by benchmarks.
        """
        # Skip warmup request (empty request_ids & dummy run prompt)
        if getattr(self, "skip_warmup", False) and getattr(req, "request_ids") == ["dummy_req_id"]:
            prompt = (
                req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            )
            if prompt == "dummy run":
                logger.info("Skipping warmup request on Neuron pipeline")
                return DiffusionOutput(output=None)

        if self._collect_perf:
            t_start = time.perf_counter()

        # --- Text Encoding + Denoising (via parent forward) ---
        # encode_prompt is instrumented to capture self._text_encode_seconds.
        # We derive denoise_seconds = parent_forward_time - text_encode_seconds.
        if self._collect_perf:
            t_parent_start = time.perf_counter()
        # vllm-omni 0.24 dropped the forward(output_type=...) kwarg; latent output
        # is now selected via req.sampling_params.output_type. The Neuron pipeline
        # always needs latents back so it can run VAE decode itself below.
        sampling_params = getattr(req, "sampling_params", None)
        requested_output_type = getattr(sampling_params, "output_type", None)
        wants_latents = (
            requested_output_type or getattr(self.od_config, "output_type", None)
        ) == "latent"
        if sampling_params is not None:
            sampling_params.output_type = "latent"
        self._clear_cross_attention_kv_cache()
        try:
            result = super().forward(req)
        finally:
            if sampling_params is not None:
                sampling_params.output_type = requested_output_type
            self._clear_cross_attention_kv_cache()
        # Same reasoning for the scheduler's solver history: model_outputs and last_sample
        # are read only inside step() and are re-nulled by the next set_timesteps, so they
        # are dead here, but they pin solver_order + 1 device latents (12 MB at 480p,
        # 28 MB at 720p) across the offload window below.
        self._release_scheduler_history()
        if self._collect_perf:
            parent_forward_seconds = time.perf_counter() - t_parent_start

        run_vae_decode = self.is_vae_rank and not wants_latents

        # --- Offload DiT to CPU to free device memory for VAE (opt-in, VAE rank only) ---
        if self._cpu_offload and run_vae_decode and not is_lite_runtime():
            self._move_transformers_to(torch.device("cpu"))

        # --- VAE Decode ---
        # In patch-parallel mode every rank in the VAE subgroup runs decode in
        # lockstep (the executor's gather/all_reduce/broadcast are collective over
        # that subgroup); only the output rank receives the merged frames
        # (broadcast_result=False leaves other members with a dummy empty tensor).
        # Ranks outside the subgroup skip decode. Otherwise only rank 0 decodes.
        if self._collect_perf:
            t_vae_start = time.perf_counter()
        output = None
        if run_vae_decode:
            decoded = self._decode_latents(result.output)
            if self.is_output_rank:
                output = decoded
        elif wants_latents and self.is_output_rank:
            output = result.output.to("cpu")

        # --- Restore DiT to device after VAE (opt-in, VAE rank only) ---
        if self._cpu_offload and run_vae_decode and not is_lite_runtime():
            self._move_transformers_to(self.device)

        if self._collect_perf:
            vae_decode_seconds = time.perf_counter() - t_vae_start
            e2e_forward_seconds = time.perf_counter() - t_start

            # Denoise time = parent forward (encode + denoise) minus text encoding
            denoise_seconds = (
                (parent_forward_seconds - self._text_encode_seconds)
                if self._text_encode_seconds is not None
                else parent_forward_seconds
            )

            # Only rank 0 has complete timing (including VAE decode).
            # Write metrics file only from rank 0 to avoid race conditions.
            if self.is_output_rank:
                self._last_perf_metrics = self._build_perf_metrics(
                    text_encode_seconds=self._text_encode_seconds,
                    denoise_seconds=denoise_seconds,
                    vae_decode_seconds=vae_decode_seconds,
                    e2e_forward_seconds=e2e_forward_seconds,
                    num_steps=self._num_timesteps,
                    height=req.sampling_params.height,
                    width=req.sampling_params.width,
                    num_frames=req.sampling_params.num_frames,
                )
                self._write_perf_metrics_file()

        # Return raw tensor — the engine's post_process_func handles postprocessing.
        return DiffusionOutput(output=output)

    @staticmethod
    def _build_perf_metrics(
        *,
        text_encode_seconds: float | None,
        denoise_seconds: float | None,
        vae_decode_seconds: float,
        e2e_forward_seconds: float,
        num_steps: int | None,
        height: int | None,
        width: int | None,
        num_frames: int | None,
    ) -> dict:
        """Build the performance metrics dictionary from raw timing values."""
        if not all([num_steps, height, width, num_frames]):
            logger.warning(
                "Pipeline perf metrics: some generation params are unset "
                "(num_steps=%s, height=%s, width=%s, num_frames=%s). "
                "Metrics may be incomplete.",
                num_steps,
                height,
                width,
                num_frames,
            )

        return {
            "text_encode_seconds": text_encode_seconds,
            "denoise_seconds": denoise_seconds,
            "vae_decode_seconds": vae_decode_seconds,
            "e2e_forward_seconds": e2e_forward_seconds,
            "frames_per_second": num_frames / e2e_forward_seconds
            if (num_frames and num_frames > 0 and e2e_forward_seconds > 0)
            else 0.0,
            "num_frames": num_frames or 0,
            "height": height or 0,
            "width": width or 0,
            "num_steps": num_steps or 0,
        }

    def _write_perf_metrics_file(self) -> None:
        """Write _last_perf_metrics to a JSON file accessible by the test process."""
        if not self._last_perf_metrics:
            return

        # Use WORKLOAD_OUTPUT_RW if available, else /tmp
        base_dir = os.environ.get("WORKLOAD_OUTPUT_RW", "/tmp")
        metrics_dir = os.path.join(base_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, "pipeline_perf_metrics.json")

        try:
            with open(metrics_path, "w") as f:
                json.dump(self._last_perf_metrics, f, indent=2)
        except Exception as e:
            logger.warning("Failed to write perf metrics to %s: %s", metrics_path, e)

    def compile_text_encoder(self, *args, **kwargs):
        options = {**kwargs.pop("options", {}), "model_name": "wan_text_encoder"}
        options["compiler_args"] = [
            "--model-type=transformer",
            "--auto-cast=none",
            "-O1",
            "--hbm-scratchpad-page-size=2048",
        ]
        kwargs.setdefault("fullgraph", True)
        self.text_encoder.compile(*args, options=options, **kwargs)

    def compile_vae(self, *args, **kwargs):
        if self.is_vae_rank:
            options = {**kwargs.pop("options", {}), "model_name": "wan_vae"}
            options["compiler_args"] = [
                "--model-type=unet-inference",
                "--auto-cast=none",
                "--internal-max-instruction-limit=15000000",
                "-O1",
                "--hbm-scratchpad-page-size=2048",
            ]
            # Spatial tiling runs multiple decode passes over varying tile shapes,
            # which triggers recompiles and hits FailOnRecompileLimitHit under
            # fullgraph. Disable fullgraph for the VAE when tiling is enabled.
            # TODO: Restore fullgraph=True after padding boundary tiles
            # to a fixed decoder input shape.
            kwargs.setdefault("fullgraph", not getattr(self.vae, "use_tiling", False))
            self.vae.compile(*args, options=options, **kwargs)

    def compile_transformer(self, t, *args, **kwargs):
        options = {**kwargs.pop("options", {})}
        options["compiler_args"] = [
            "--model-type=transformer",
            "--auto-cast=none",
            "-O1",
            "--hbm-scratchpad-page-size=2048",
        ]
        kwargs.setdefault("fullgraph", True)
        if is_lite_runtime() and getattr(t, "_lite_cache_dit_enabled", False):
            t.compile_lite_cache_dit(*args, options=options, **kwargs)
            # Cache-DiT already splits the full transformer into compiled phases. Its
            # first probe generates and returns K/V when this bound orchestrator is used.
            self._compiled_cross_attention_kv_generators[id(t)] = (
                t.forward_and_cache_cross_attention_kv
            )
        else:
            # Compile two full DiT graphs at O1: the steady-state graph consumes K/V,
            # while the first-step graph generates K/V inside the same transformer
            # invocation that first consumes it and returns them with the noise prediction.
            t.compile(*args, options={**options, "model_name": "wan_transformer"}, **kwargs)
            self._compiled_cross_attention_kv_generators[id(t)] = torch.compile(
                t.forward_and_cache_cross_attention_kv,
                *args,
                options={**options, "model_name": "wan_transformer_cache_gen"},
                **kwargs,
            )

    def compile(self, *args, **kwargs):
        """Compile text_encoder, vae, & transformer(s) for Neuron."""
        self.compile_text_encoder(*args, **kwargs)
        self.compile_vae(*args, **kwargs)
        for t in (self.transformer, self.transformer_2):
            if t is not None:
                self.compile_transformer(t, *args, **kwargs)
        if is_lite_runtime():
            self._compiled_cfg_combine = _compile_lite_helper(
                self._cfg_combine,
                *args,
                **kwargs,
            )
            self._compiled_latent_cast = _compile_lite_helper(
                self._cast_latents,
                *args,
                **kwargs,
            )
            self.scheduler.compile(*args, **kwargs)
        return self

    def load_weights(self, weights=None):
        """Load weights from model directory."""
        transformers = {
            "transformer": self.transformer,
            "transformer_2": self.transformer_2,
        }
        fp8_components = [
            component
            for component, transformer in transformers.items()
            if transformer is not None
            and getattr(transformer, "_quantization_mode", "bf16") == "fp8_row_mx"
        ]

        model_path = self.od_config.model
        if not os.path.isdir(model_path):
            from huggingface_hub import snapshot_download

            download_kwargs = {}
            if fp8_components:
                download_kwargs["ignore_patterns"] = [
                    "transformer/*.safetensors",
                    "transformer_2/*.safetensors",
                ]
            model_path = snapshot_download(model_path, **download_kwargs)

        comfyui_fp8_model_path = self._comfyui_fp8_model_path
        if fp8_components:
            if comfyui_fp8_model_path is None:
                raise ValueError("comfyui_fp8_model_path is required when loading FP8 transformers")
            if not os.path.isdir(comfyui_fp8_model_path):
                from huggingface_hub import snapshot_download

                comfyui_fp8_model_path = snapshot_download(
                    comfyui_fp8_model_path,
                    allow_patterns=[
                        f"{FP8_CHECKPOINT_SUBDIR}/{FP8_CHECKPOINT_FILES[component]}"
                        for component in fp8_components
                    ],
                )

        self.text_encoder.load_weights(os.path.join(model_path, "text_encoder"))

        for component, transformer in transformers.items():
            if transformer is None:
                continue
            if component in fp8_components:
                checkpoint_path = os.path.join(
                    comfyui_fp8_model_path,
                    FP8_CHECKPOINT_SUBDIR,
                    FP8_CHECKPOINT_FILES[component],
                )
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError(
                        f"FP8 checkpoint for {component} not found: {checkpoint_path}"
                    )
            else:
                checkpoint_path = os.path.join(model_path, component)
            transformer.load_weights(checkpoint_path)
