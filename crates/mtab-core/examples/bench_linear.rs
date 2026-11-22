//! Reproducible microbenchmark of prefix linear algebra; not end-to-end RTF.
//! Run: MATMUL_NUM_THREADS=1 cargo run -p mtab-core --release --example bench_linear

use std::hint::black_box;
use std::time::Instant;

fn main() {
    let (m, k, n) = (256usize, 768usize, 768usize);
    let x: Vec<f64> = (0..m * k)
        .map(|i| ((i % 101) as f64 - 50.0) / 50.0)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
        .collect();
    let start = Instant::now();
    let actual = mtab_core::nn::linear(black_box(&x), black_box(&w), None, m, k, n);
    let fast = start.elapsed();
    let start = Instant::now();
    let mut reference = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let (mut a0, mut a1, mut a2, mut a3) = (0.0, 0.0, 0.0, 0.0);
            for z in (0..k).step_by(4) {
                a0 += x[i * k + z] * w[j * k + z] as f64;
                a1 += x[i * k + z + 1] * w[j * k + z + 1] as f64;
                a2 += x[i * k + z + 2] * w[j * k + z + 2] as f64;
                a3 += x[i * k + z + 3] * w[j * k + z + 3] as f64;
            }
            reference[i * n + j] = (a0 + a1) + (a2 + a3);
        }
    }
    let scalar = start.elapsed();
    let max_error = actual
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("{m}x{k} times {n}x{k}: GEMM {fast:?}, scalar {scalar:?}, {:.2}x, max error {max_error:.3e}", scalar.as_secs_f64() / fast.as_secs_f64());
    black_box(reference);
}
