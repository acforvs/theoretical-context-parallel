# Estimating Context Parallel

Analytical and empirical models for Ulysses, Ring, and ZigZag context
parallelism. The default model is causal, BF16, GQA-aware, and includes the
128-by-128 tile quantization used by FlashAttention-3 for head dimension 128.

The model estimates one attention forward pass. It is not an end-to-end
training model: projections, backward, collective latency, topology details,
and pipeline bubbles are outside its scope.

## Run

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
python -m src.main imbalance theoretical
python -m src.main cost-model
```

Use `--ideal-flops` to reproduce the theorem's smooth quadratic model and
`--no-plot` when only numerical output is needed.

GPU mode additionally requires Hopper-compatible FlashAttention-3, either as
`flash_attn_3.flash_attn_interface` or the legacy top-level
`flash_attn_interface`. It benchmarks causal varlen GQA using the actual maximum
sequence length in each pack.

## Important assumptions

- `dp` is the total device count; the number of independent DP/CP groups is
  `dp // cp`.
- Input sampling models concat-then-split packing. A document crossing a pack
  boundary becomes a new causal sequence.
- KV heads are replicated when Ulysses CP exceeds the GQA KV-head count.
- Ulysses QKV all-to-all, attention, and output all-to-all are added because
  they are data-dependent. Ring communication overlaps per-step computation.
- The tile-aware model counts scheduled matrix-multiply slots, then applies a
  roofline bound. The checked-in H100 calibration (`680 TFLOP/s` and `95 us`
  fixed cost) comes from `benchmarks/results/fa3_h100_results.json`; rerun
  `benchmarks/analyze_fa3_results.py` when changing FA or hardware.

## Ulysses projections with TP and sequence parallelism

The Python API can also estimate attention forward including column-parallel
QKV and row-parallel output projections. Provide `tp` and `hidden_size` to
`UlyssesAttention`, with independently measured `gemm_flops` in its
`HardwareConfig`. Omitting `hidden_size` preserves attention-only estimates. The existing
CLI and strategy comparisons remain attention-only.

```python
from dataclasses import replace

from src.cost_model import GB300, UlyssesAttention

strategy = UlyssesAttention(
    cp=4,
    tp=2,
    hw=replace(GB300, gemm_flops=1e15),  # Illustrative; calibrate separately.
    hidden_size=4096,  # Example model width; use the actual hidden size.
)
seconds = strategy.total_time([8192, 8192, 8192, 8192])
```

The forward sequence follows standard
[Megatron TP/SP projections](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html):

```text
SP all-gather -> QKV GEMM -> CP all-to-all -> attention
             -> CP all-to-all -> output GEMM -> SP reduce-scatter
```

`batch` contains the full sequence pack for a CP group, shared across TP ranks.
For total tokens S, hidden size H, and head dimension D:

- Layer input/output per rank: `[S / (CP * TP), H]`.
- Each projection processes `S / CP` tokens after the SP all-gather.
- QKV projection output width: `(TP-local Q heads + 2 * TP-local KV heads) * D`.
- FlashAttention Q heads per rank: `Q heads / (TP * CP)`; KV heads are
  replicated when the partition count exceeds the KV-head count.
- Output projection input width: `TP-local Q heads * D`; output width: H.
- Each SP collective transfers `(S / CP) * H * dtype_bytes * (TP - 1) / TP`
  bytes per rank. TP=1 has no SP communication.

`_gemm_compute_time` includes arithmetic and HBM traffic, like attention, using
`max(FLOPs / gemm_flops, input_weight_output_bytes / HBM_bandwidth)`
plus `HardwareConfig.gemm_launch_latency` (default zero, uncalibrated).
FlashAttention uses the separate `flash_flops` and `flash_launch_latency`
hardware fields. `gemm_flops` defaults to `None` until calibrated.
`_compute_time` adds attention and both projections; `_comm_time` adds both CP
exchanges and `_sp_comm_time` for SP all-gather/reduce-scatter. Phases are added
sequentially.
Placement assumes contiguous TP ranks: TP bandwidth uses TP group size, while
CP bandwidth uses the TP*CP span. This is a coarse bandwidth estimate without
collective startup latency, contention, or hierarchical collectives.
Normalization, RoPE, bias, dropout, backward, and MLP/MoE remain excluded.
