# SPDX-License-Identifier: Apache-2.0
"""Stage topologies for checkpoints vllm-omni's own pipeline registry misses.

``StageConfigFactory.create_from_model`` matches a checkpoint's
``model_index.json`` ``_class_name`` against the registered
``diffusers_class_name`` values. When nothing matches, config resolution falls
through to the generic diffusion path, which synthesizes its own single-rank
stage list and **never reads the deploy YAML** — the requested TP/CP/CFG sizes
are silently dropped and the whole pipeline is loaded unsharded onto one core.

The registration has to happen before the engine resolves stage configs, which
rules out the ``vllm_omni.general_plugins`` hook (that runs later, in the engine
process). ``vllm_omni_neuron`` therefore registers these at platform-plugin
activation; see ``vllm_omni_neuron.register_pipeline_topologies``.

This module is deliberately data-only: it holds the topology away from the
pipeline module that implements the model, whose imports (torch, transformers,
``vllm_omni.diffusion``, the Neuron mixins) are far too heavy to pull in while
``vllm_omni.platforms`` is still resolving the current platform — anything in
that chain reading ``current_omni_platform`` would re-enter that resolution.
``vllm_omni.config.stage_config`` is the one dependency, and it is already on the
config-resolution path that follows.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# Topology for the Wan 2.2 I2V checkpoint. vllm-omni ships only ``wan2_2_ti2v``,
# whose ``diffusers_class_name`` matches the T2V checkpoint's ``WanPipeline``;
# the I2V checkpoint reports ``WanImageToVideoPipeline`` and so matches nothing.
WAN2_2_I2V_PIPELINE = PipelineConfig(
    model_type="wan2_2_i2v",
    model_arch="Wan22I2VPipeline",
    diffusers_class_name="WanImageToVideoPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="Wan22I2VPipeline",
        ),
    ),
)
