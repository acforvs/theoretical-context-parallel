from src.cost_model.architecture import QWEN235, AttentionConfig
from src.cost_model.hardware import H100, HardwareConfig
from src.cost_model.strategies.base import CPStrategy


class UlyssesAttention(CPStrategy):
    """Ulysses attention with all-to-all communication.

    Ulysses splits attention heads across CP ranks and uses all-to-all
    to exchange Q, K, V before local attention, then all-to-all again for output.

    Total time = max(attention_time, comm_time)
    """

    name = "ulysses"

    def __init__(
        self, cp: int, hw: HardwareConfig = H100, attn: AttentionConfig = QWEN235
    ):
        super().__init__(cp, hw, attn)

    def _compute_time(self, batch: list[int]) -> float:
        """Local attention compute time (heads split across CP)."""
        seqlen = sum(batch)
        nh = self.attn.num_heads
        dh = self.attn.head_dim
        nkvh = self.attn.num_kv_heads

        # Memory time: reading Q, K, V, writing O, assumes dtype=bfloat16
        mem_time = (
            (2 * nh * dh + 2 * nkvh * dh)
            * seqlen
            * 2
            / self.cp
            / self.hw.memory_bandwidth
        )

        # Compute time, assumes causal attention
        ops = sum(s * (s + 1) / 2 for s in batch)
        compute_time = 4 * nh * dh * ops / self.cp / self.hw.compute_flops

        return max(mem_time, compute_time)

    def _comm_time(self, total_seq_len: int) -> float:
        """AllToAlls communication for Q, K, V, and O."""
        nh = self.attn.num_heads
        dh = self.attn.head_dim
        nkvh = self.attn.num_kv_heads

        # Send QKVO in bfloat16
        # note: we do not account for the network latency and do not discount by num_hops / num_ranks
        bytes_transferred = total_seq_len * dh * (2 * nh + 2 * nkvh) * 2 / self.cp
        return bytes_transferred / self.hw.p2p_bandwidth(self.cp)

    def total_time(self, batch: list[int]) -> float:
        """Total time = max(compute, comm)."""
        compute = self._compute_time(batch)
        comm = self._comm_time(sum(batch))
        return max(compute, comm)
