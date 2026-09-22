# FlashAttention benchmark notes

`bench_fa3_varlen.py` reproduces the imbalance experiment using causal and
non-causal FA3 varlen GQA on one GPU. The checked-in results were measured on:

- NVIDIA H100 80GB HBM3
- CUDA 13.0, driver 580.95.05
- PyTorch 2.9.1+cu130
- flash-attn-3 3.0.0b1+2025121700base12170
- BF16, head dimension 128, 64 Q heads, 8 KV heads

The benchmark was run with:

```bash
python bench_fa3_varlen.py \
  --dp 128 --warmup 25 --rep 100 \
  --output results/fa3_h100_results.json
python analyze_fa3_results.py results/fa3_h100_results.json
python plot_h100_validation.py \
  results/fa3_h100_results.json \
  ../figures/imbalance/h100-fa3-validation.png \
  --detail-dir ../figures/imbalance/real
```

The fixed-maximum control used `--fixed-max-seqlen --cp 1 8 64 --dp 64`.
These are sequential samples of hypothetical DP/CP-group workloads on one
H100; they do not include concurrent distributed communication or system-level
straggler noise.

## FA4 on GB300

`bench_fa4_varlen.py` uses `flash_attn.cute.flash_attn_varlen_func` with the
same BF16, head-dimension-128 workloads. Install FA4, PyTorch, Triton, and NumPy
in the GPU environment, then run from the repository root:

```bash
python benchmarks/bench_fa4_varlen.py --dp 128 --warmup 25 --rep 100 \
  --output benchmarks/results/fa4_gb300_results.json
python benchmarks/analyze_fa3_results.py benchmarks/results/fa4_gb300_results.json
```

The analyzer accepts the same JSON schema. Timing uses median eager invocation
latency after compilation and warmup; `--warmup` and `--rep` are milliseconds.
`--block-q` and `--block-kv` only set analytical tile counts (default 128 each),
not FA4 kernel configuration. Verify those assumptions before using a tiled
fit. Results also include useful/tiled FLOPs and estimated Q/K/V/output bytes
for a roofline fit. The existing analyzer fits a linear compute-only model.

## GB300 projection GEMMs

`bench_gemm.py` benchmarks tier7_gen2 BF16 QKV and output projections using
Transformer Engine (`transformer_engine.pytorch.Linear`, no bias). It fixes
hidden size 7168, 64 Q heads, 8 KV heads, and head dimension 128. The default sweep uses
32768 tokens (microbatch 1), TP and CP each in {1, 2, 4, 8, 16}, and three
measurements per shape. Inputs and contiguous `[out_features, in_features]`
weights are allocated outside timing; output allocation is included. Timing
uses warmed eager calls, not CUDA graphs. Each TE module runs a local GEMM
with TP already reflected in its dimensions;
distributed collectives, normalization, FP8, and backward are excluded.

```bash
python benchmarks/bench_gemm.py --output benchmarks/results/gemm_te_gb300_results.json
python benchmarks/analyze_gemm_results.py benchmarks/results/gemm_te_gb300_results.json
```

Both projections process `32768 / CP` rows after the SP all-gather. TP=16
replicates KV heads. All 25 TP/CP pairs can benchmark GEMMs, while pairs with
TP*CP > 64 are marked unsupported for full Ulysses. Global batch size does not
change per-invocation GEMM shapes.

These are ungated projection shapes; the gated tier7_gen2 projection also
includes an output-gate block and requires a separate calibration.

The analyzer reports unconstrained linear fits and nonnegative roofline fits
with fixed effective HBM bandwidth (default 5.6 TB/s), jointly and separately
by projection and TP. Errors are in-sample. Inspect residuals and timing
variation before adopting one shared throughput/overhead pair.
