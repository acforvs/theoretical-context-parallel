from typing import cast

import numpy as np

from src.cost_model.architecture import QWEN235, AttentionConfig
from src.cost_model.hardware import H100, HardwareConfig
from src.cost_model.strategies.base import CPStrategy


class RingAttention(CPStrategy):
    """Ring attention with sequential KV block passing.

    Each rank holds a contiguous chunk of tokens. KV blocks are passed
    around the ring while Q blocks stay local. Causal masking means
    ranks only compute attention for KV positions <= Q positions.

    At each step, compute and communication overlap - we wait for whichever
    is slower before proceeding to the next step.
    """

    name = "ring"

    def __init__(
        self, cp: int, hw: HardwareConfig = H100, attn: AttentionConfig = QWEN235
    ):
        super().__init__(cp, hw, attn)

    def total_time(self, batch: list[int]) -> float:
        offsets = np.cumsum([0] + batch)
        total_tokens = offsets[-1]
        tokens_per_rank = total_tokens // self.cp

        rank_starts = np.arange(self.cp) * tokens_per_rank
        rank_ends = rank_starts + tokens_per_rank

        s_starts = offsets[:-1]
        s_ends = offsets[1:]

        nh = self.attn.num_heads
        dh = self.attn.head_dim
        nkvh = self.attn.num_kv_heads

        bytes_per_step = tokens_per_rank * nkvh * dh * 2 * 2
        comm_time_per_step = bytes_per_step / self.hw.p2p_bandwidth(self.cp)

        total_time = 0.0

        for step in range(self.cp):
            max_compute_at_step = 0.0

            for i in range(self.cp):
                q_rank = i
                kv_rank = i - step

                #  skip if kv comes from future positions
                if kv_rank < 0:
                    continue

                q_range = (rank_starts[q_rank], rank_ends[q_rank])
                kv_range = (rank_starts[kv_rank], rank_ends[kv_rank])

                # find overlap of each sample with Q and KV ranges
                q_lens = np.maximum(
                    0, np.minimum(q_range[1], s_ends) - np.maximum(q_range[0], s_starts)
                )
                kv_lens = np.maximum(
                    0,
                    np.minimum(kv_range[1], s_ends) - np.maximum(kv_range[0], s_starts),
                )

                active = (q_lens > 0) & (kv_lens > 0)
                if not np.any(active):
                    continue

                # Diagonal block (same rank): triangular attention
                # Off-diagonal block: rectangular attention
                if step == 0:
                    ops = np.sum(q_lens[active] * (q_lens[active] + 1) / 2)
                else:
                    ops = np.sum(q_lens[active] * kv_lens[active])

                compute_time = 4 * nh * dh * ops / self.hw.compute_flops
                max_compute_at_step = max(max_compute_at_step, compute_time)

            # wait for the slower of compute or communication
            total_time += max(max_compute_at_step, comm_time_per_step)

        return cast(float, total_time)
