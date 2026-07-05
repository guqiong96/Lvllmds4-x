# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    create_dspark_draft_vllm_config,
)


@dataclass
class _KernelConfig:
    moe_backend: str = "auto"


@dataclass
class _AttentionConfig:
    backend: str | None = None
    use_non_causal: bool = False


@dataclass
class _SpeculativeConfig:
    moe_backend: str | None = None
    attention_backend: str | None = None


@dataclass
class _VllmConfig:
    kernel_config: _KernelConfig = field(default_factory=_KernelConfig)
    attention_config: _AttentionConfig = field(default_factory=_AttentionConfig)
    speculative_config: _SpeculativeConfig = field(
        default_factory=_SpeculativeConfig
    )


def _make_config(
    target_moe_backend: str,
    draft_moe_backend: str | None,
) -> _VllmConfig:
    return _VllmConfig(
        kernel_config=_KernelConfig(moe_backend=target_moe_backend),
        attention_config=_AttentionConfig(),
        speculative_config=_SpeculativeConfig(moe_backend=draft_moe_backend),
    )


def test_dspark_draft_config_moe_backend_override():
    tgt_config = _make_config(
        target_moe_backend="flashinfer_trtllm",
        draft_moe_backend="marlin",
    )

    draft_config = create_dspark_draft_vllm_config(tgt_config)

    assert draft_config.kernel_config.moe_backend == "marlin"
    assert tgt_config.kernel_config.moe_backend == "flashinfer_trtllm"
    assert draft_config.attention_config.use_non_causal is True


def test_dspark_draft_config_moe_backend_inherits_target():
    tgt_config = _make_config(
        target_moe_backend="flashinfer_cutlass",
        draft_moe_backend=None,
    )

    draft_config = create_dspark_draft_vllm_config(tgt_config)

    assert draft_config.kernel_config.moe_backend == "flashinfer_cutlass"
    assert draft_config.attention_config.use_non_causal is True
