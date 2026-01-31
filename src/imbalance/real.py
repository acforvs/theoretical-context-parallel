from typing import cast

import torch
import triton

from src.imbalance.base import BaseSimulator, SimulationConfig


class RealSimulator(BaseSimulator):
    """Benchmarks flash attention on GPU."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self._q, self._k, self._v = self._make_qkv()

    def _make_qkv(self, head_dim: int = 128) -> tuple[torch.Tensor, ...]:
        shape = (
            self.config.effective_batch_seq_len,
            self.config.local_nheads,
            head_dim,
        )
        return tuple(
            torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3)
        )

    def _generate_cu_seqlens(self, batch: list[int]) -> torch.Tensor:
        seqlens = [0] + batch
        return torch.cumsum(torch.tensor(seqlens, device="cuda"), dim=0).to(torch.int32)

    def compute_cost(self, batch: list[int]) -> float:
        import flash_attn_interface  # FA3

        cu_seqlens = self._generate_cu_seqlens(batch)
        max_seqlen = self.config.max_seq_len

        fn = lambda: flash_attn_interface.flash_attn_varlen_func(
            self._q, self._k, self._v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        )
        return cast(float, triton.testing.do_bench(fn))
