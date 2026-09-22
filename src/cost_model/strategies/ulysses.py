from src.cost_model.architecture import QWEN235, AttentionConfig
from src.cost_model.flash import batch_attention_pairs
from src.cost_model.hardware import H100, HardwareConfig
from src.cost_model.strategies.base import CPStrategy


class UlyssesAttention(CPStrategy):
    """Ulysses attention with all-to-all communication.

    Ulysses splits attention heads across CP ranks and uses all-to-all
    to exchange Q, K, V before local attention, then all-to-all again for output.

    Set hidden_size and hw.gemm_flops to include column-parallel QKV and
    row-parallel output projections with sequence parallelism across TP.
    All forward phases are added without communication/compute overlap.
    """

    name = "ulysses"

    def __init__(
        self,
        cp: int,
        hw: HardwareConfig = H100,
        attn: AttentionConfig = QWEN235,
        tile_aware: bool = True,
        *,
        tp: int = 1,
        hidden_size: int | None = None,
    ):
        super().__init__(cp, hw, attn, tile_aware)
        if tp <= 0:
            raise ValueError("TP must be positive")
        if hidden_size is not None and hw.gemm_flops is None:
            raise ValueError("Set hw.gemm_flops to estimate projections")
        if hidden_size is not None and hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.tp = tp
        self.hidden_size = hidden_size
        self.tp_q_heads, self.tp_kv_heads = self.attn.local_heads(tp)
        self.local_q_heads, self.local_kv_heads = self.attn.local_heads(tp * cp)

    def _attn_compute_time(self, batch: list[int]) -> float:
        """Local attention compute time (heads split across TP and CP)."""
        seqlen = sum(batch)
        local_q_heads, local_kv_heads = self.local_q_heads, self.local_kv_heads
        pairs = batch_attention_pairs(
            batch,
            causal=self.attn.causal,
            tile_aware=self.tile_aware,
            block_q=self.attn.flash_block_q,
            block_kv=self.attn.flash_block_kv,
        )
        return self._flash_time(pairs, seqlen, seqlen, local_q_heads, local_kv_heads)

    def _attn_comm_time(self, total_seq_len: int) -> float:
        """QKV all-to-all before attention."""
        return self._attn_all_to_all_time(
            total_seq_len, self.local_q_heads + 2 * self.local_kv_heads
        )

    def _attn_output_comm_time(self, total_seq_len: int) -> float:
        """Output all-to-all after attention."""
        return self._attn_all_to_all_time(total_seq_len, self.local_q_heads)

    def _attn_all_to_all_time(self, total_seq_len: int, heads: int) -> float:
        if self.cp == 1:
            return 0.0
        # Local head counts include replicated KV heads.
        bytes_transferred = (
            total_seq_len * self.attn.head_dim * heads * self.attn.dtype_bytes
        )
        # TP ranks are contiguous; a CP group spans the TP x CP placement.
        return bytes_transferred / self.hw.p2p_bandwidth(self.cp * self.tp)

    def _gemm_compute_time(self, m: int, k: int, n: int) -> float:
        assert self.hw.gemm_flops is not None
        compute_time = 2 * m * k * n / self.hw.gemm_flops
        io_bytes = (m * k + k * n + m * n) * self.attn.dtype_bytes
        memory_time = io_bytes / self.hw.memory_bandwidth
        return max(compute_time, memory_time) + self.hw.gemm_launch_latency

    def _qkv_compute_time(self, total_tokens: int) -> float:
        if self.hidden_size is None:
            return 0.0
        qkv_width = (self.tp_q_heads + 2 * self.tp_kv_heads) * self.attn.head_dim
        return self._gemm_compute_time(
            total_tokens // self.cp, self.hidden_size, qkv_width
        )

    def _output_compute_time(self, total_tokens: int) -> float:
        if self.hidden_size is None:
            return 0.0
        return self._gemm_compute_time(
            total_tokens // self.cp,
            self.tp_q_heads * self.attn.head_dim,
            self.hidden_size,
        )

    def _sp_all_gather_time(self, total_tokens: int) -> float:
        if self.hidden_size is None or self.tp == 1:
            return 0.0
        # Each SP collective transfers (TP - 1) / TP of the gathered tensor.
        wire_bytes = (
            (total_tokens // self.cp) * self.hidden_size * self.attn.dtype_bytes
            * (self.tp - 1) / self.tp
        )
        return wire_bytes / self.hw.p2p_bandwidth(self.tp)

    def _sp_reduce_scatter_time(self, total_tokens: int) -> float:
        return self._sp_all_gather_time(total_tokens)

    def _sp_comm_time(self, total_tokens: int) -> float:
        return (
            self._sp_all_gather_time(total_tokens)
            + self._sp_reduce_scatter_time(total_tokens)
        )

    def _compute_time(self, batch: list[int]) -> float:
        total_tokens = sum(batch)
        return (
            self._qkv_compute_time(total_tokens)
            + self._attn_compute_time(batch)
            + self._output_compute_time(total_tokens)
        )

    def _comm_time(self, total_seq_len: int) -> float:
        return (
            self._attn_comm_time(total_seq_len)
            + self._attn_output_comm_time(total_seq_len)
            + self._sp_comm_time(total_seq_len)
        )

    def _validate_batch(self, batch: list[int]) -> None:
        if not batch or any(length <= 0 for length in batch):
            raise ValueError("batch must contain positive sequence lengths")
        if sum(batch) % (self.cp * self.tp):
            raise ValueError("total tokens must be divisible by CP * TP")

    def total_time(self, batch: list[int]) -> float:
        self._validate_batch(batch)
        compute = self._compute_time(batch)
        comm = self._comm_time(sum(batch))
        return compute + comm
