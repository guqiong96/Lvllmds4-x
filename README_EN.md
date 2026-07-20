# Lvllmds4-x

A fork of [yhfgyyf/vllm-deepseek-v4-sm89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89) with CPU-GPU hybrid inference support for DeepSeek-V4 on NVIDIA SM80+ (A100, RTX 4090, etc.).

## Origin

This project integrates the CPU-GPU hybrid inference engine **[lk_moe](https://pypi.org/project/lk-moe/)** into a specialized vLLM fork:

- **Base vLLM Fork:** Forked from `yhfgyyf/vllm-deepseek-v4-sm89`, which provides the compatibility modifications needed for DeepSeek-V4 on SM89 architecture, extended here to support SM80+ GPUs.
- **Hybrid Inference:** The lk_moe engine enables MOE layers to leverage both GPU VRAM and CPU system memory for collaborative computation, with NUMA-aware scheduling and expert weight management.

This version is purpose-built to run **DeepSeek V4 on NVIDIA GPUs with SM80+ compute capability** (A100, RTX 4090, A6000, etc.).

## Relationship with LvLLM

Lvllmds4-x is part of the LvLLM ecosystem—a family of parallel projects that integrate the lk_moe hybrid inference engine into different vLLM branches for different model/hardware targets:

| Project | Upstream vLLM Branch | Target |
|---------|---------------------|--------|
| [LvLLM](https://github.com/guqiong96/Lvllm) | Latest vLLM mainline | General MoE models (Qwen3, GLM, MiniMax, Kimi, etc.) |
| [Lvllmds4](https://github.com/guqiong96/Lvllmds4) | `jasl/vllm` (`codex/ds4-sm120-min-enable`) | DeepSeek-V4 (SM120+) |
| Lvllmds4-x (this repo) | `yhfgyyf/vllm-deepseek-v4-sm89` | DeepSeek-V4 (SM80+) |

Similarly, **[Lsglang](https://github.com/guqiong96/Lsglang)** integrates lk_moe into sglang for the same hybrid inference capabilities across frameworks.

## Usage Guide

Pre-built releases and detailed installation/usage instructions for DeepSeek-V4 on SM80+ are available on the **[Releases page](https://github.com/guqiong96/Lvllmds4-x/releases)**.