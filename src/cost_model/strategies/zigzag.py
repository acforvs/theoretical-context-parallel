from typing import cast

import numpy as np

from src.cost_model.architecture import QWEN235, AttentionConfig
from src.cost_model.hardware import H100, HardwareConfig
from src.cost_model.strategies.base import CPStrategy


class ZigZagAttention(CPStrategy):
    """ZigZag ring attention for better load balancing.

    Instead of contiguous chunks, each rank gets two blocks:
    one from the start and one from the end of the sequence.

    At each step, compute and communication overlap - we wait for whichever
    is slower before proceeding to the next step.
    """

    name = "zigzag"

    def __init__(
        self, cp: int, hw: HardwareConfig = H100, attn: AttentionConfig = QWEN235
    ):
        super().__init__(cp, hw, attn)

    def _get_rank_chunks(self, rank: int, block_size: int) -> list[tuple[int, int]]:
        """Get the two token ranges assigned to a rank."""
        start1 = rank * block_size
        end1 = start1 + block_size
        start2 = (2 * self.cp - 1 - rank) * block_size
        end2 = start2 + block_size
        return [(start1, end1), (start2, end2)]

    def total_time(self, batch: list[int]) -> float:
        offsets = np.cumsum([0] + batch)
        total_tokens = offsets[-1]
        block_size = total_tokens // (2 * self.cp)

        s_starts = offsets[:-1]
        s_ends = offsets[1:]

        nh = self.attn.num_heads
        dh = self.attn.head_dim
        nkvh = self.attn.num_kv_heads

        # assumes bf16
        bytes_per_step = 2 * block_size * nkvh * dh * 2 * 2
        comm_time_per_step = bytes_per_step / self.hw.p2p_bandwidth(self.cp)

        total_time = 0.0

        for step in range(self.cp):
            max_compute_at_step = 0.0

            for i in range(self.cp):
                q_rank = i
                kv_rank = (i - step) % self.cp

                if (i - step) < 0:
                    continue

                q_chunks = self._get_rank_chunks(q_rank, block_size)
                kv_chunks = self._get_rank_chunks(kv_rank, block_size)

                rank_ops = 0.0

                for q_range in q_chunks:
                    for kv_range in kv_chunks:
                        if kv_range[0] > q_range[0]:
                            continue

                        q_lens = np.maximum(
                            0,
                            np.minimum(q_range[1], s_ends)
                            - np.maximum(q_range[0], s_starts),
                        )
                        kv_lens = np.maximum(
                            0,
                            np.minimum(kv_range[1], s_ends)
                            - np.maximum(kv_range[0], s_starts),
                        )

                        active = (q_lens > 0) & (kv_lens > 0)
                        if not np.any(active):
                            continue

                        if q_range == kv_range:
                            ops = np.sum(q_lens[active] * (q_lens[active] + 1) / 2)
                        else:
                            ops = np.sum(q_lens[active] * kv_lens[active])

                        rank_ops += ops

                compute_time = 4 * nh * dh * rank_ops / self.hw.compute_flops
                max_compute_at_step = max(max_compute_at_step, compute_time)

            total_time += max(max_compute_at_step, comm_time_per_step)

        return cast(float, total_time)
