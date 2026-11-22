# Preliminary microbenchmark

This is **not** a transcription benchmark and does not measure or establish the
real-time-factor, memory or accuracy targets in PROMPT.md §6.13. Those require
real release weights and end-to-end runs, which have not yet been performed.

On 2026-11-22 in a Linux x86-64 sandbox reporting an Intel Xeon 2.60 GHz CPU,
with Rust 1.100.0, `--release`, and `MATMUL_NUM_THREADS=1`, one run of the
batched linear microbenchmark measured:

| Operation | Time |
| --- | ---: |
| f64 GEMM, including f32-to-f64 weight conversion | 38.2 ms |
| Previous four-lane scalar dot-product loop | 149.5 ms |

The matrix dimensions were 256×768 times the transpose of 768×768; the measured
maximum absolute difference was 1.71e-13. This is a single, non-statistical
observation on a shared machine. Full transformer golden and exact token-stream
tests remain the correctness gate. Reproduce with:

```sh
MATMUL_NUM_THREADS=1 cargo run -p mtab-core --release --example bench_linear
```

The GEMM library also supports up to four threads via `MATMUL_NUM_THREADS`;
single-token decoding still uses the scalar path. GPU acceleration and
end-to-end release-checkpoint performance remain unmeasured.
