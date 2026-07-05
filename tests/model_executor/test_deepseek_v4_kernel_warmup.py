# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.warmup import kernel_warmup


def _mtp_runner(query_len: int = 3):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp"),
        num_spec_tokens=query_len - 1,
        uniform_decode_query_len=query_len,
    )


def test_deepseek_v4_mtp_uniform_decode_warmup_covers_c256():
    requests = kernel_warmup._deepseek_v4_mtp_uniform_decode_warmup_requests(
        _mtp_runner(),
        max_tokens=4096,
        max_reqs=256,
    )

    assert requests == (1, 2, 4, 8, 16, 24, 32, 256)


def test_deepseek_v4_mtp_uniform_decode_warmup_still_respects_limits():
    assert kernel_warmup._deepseek_v4_mtp_uniform_decode_warmup_requests(
        _mtp_runner(),
        max_tokens=4096,
        max_reqs=24,
    ) == (1, 2, 4, 8, 16, 24)
    assert kernel_warmup._deepseek_v4_mtp_uniform_decode_warmup_requests(
        _mtp_runner(),
        max_tokens=96,
        max_reqs=256,
    ) == (1, 2, 4, 8, 16, 24, 32)


class _FakeV2BlockTables:
    def __init__(self):
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor:
        self.calls.append(
            (
                idx_mapping.clone(),
                query_start_loc.clone(),
                positions.clone(),
                num_tokens_padded,
            )
        )
        return torch.empty((1, num_tokens_padded), dtype=torch.int64)


def test_deepseek_v4_slot_mapping_warmup_supports_v2_runner_without_input_batch():
    block_tables = _FakeV2BlockTables()
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        max_num_tokens=4,
        input_buffers=SimpleNamespace(
            query_start_loc=torch.zeros(2, dtype=torch.int32),
            positions=torch.full((4,), -1, dtype=torch.int64),
        ),
        block_tables=block_tables,
    )

    kernel_warmup._deepseek_v4_slot_mapping_warmup(runner)

    assert block_tables.calls
    idx_mapping, query_start_loc, positions, num_tokens_padded = next(
        call for call in block_tables.calls if call[3] == 4
    )
    assert idx_mapping.tolist() == [0]
    assert query_start_loc.tolist() == [0, 4]
    assert positions.tolist() == [0, 1, 2, 3]
    assert num_tokens_padded == 4
