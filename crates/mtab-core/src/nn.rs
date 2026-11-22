//! Minimal neural-net ops for the transformer port (PROMPT.md §6.3).
//!
//! Activations compute in f64 with f32 weight storage (weights cast on use,
//! losslessly), matching upstream's float64 numerics to ~1e-12 while keeping
//! checkpoint memory at f32. Torch-matching semantics on purpose: `linear`
//! takes weights in `nn.Linear` layout `[out, in]` and computes `x @ wᵀ`;
//! `layer_norm` uses the biased variance and `eps` of `nn.LayerNorm`; `gelu`
//! is `F.gelu` exact-erf. Position embeddings stay fp32 — upstream computes
//! them in fp32 unconditionally (`transformer.py`).

/// `y = x @ wᵀ + b` with `w: [n, k]` row-major (torch Linear weight) and
/// optional bias `b: [n]`. `x` is `[m, k]`, `y` is `[m, n]`.
pub fn linear(x: &[f64], w: &[f32], b: Option<&[f32]>, m: usize, k: usize, n: usize) -> Vec<f64> {
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), n * k);
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        let xrow = &x[i * k..(i + 1) * k];
        for j in 0..n {
            let wrow = &w[j * k..(j + 1) * k];
            // 4-lane unrolled dot product (f64 accumulation).
            let (mut a0, mut a1, mut a2, mut a3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let mut kk = 0;
            while kk + 4 <= k {
                a0 += xrow[kk] * wrow[kk] as f64;
                a1 += xrow[kk + 1] * wrow[kk + 1] as f64;
                a2 += xrow[kk + 2] * wrow[kk + 2] as f64;
                a3 += xrow[kk + 3] * wrow[kk + 3] as f64;
                kk += 4;
            }
            let mut s = (a0 + a1) + (a2 + a3);
            while kk < k {
                s += xrow[kk] * wrow[kk] as f64;
                kk += 1;
            }
            out[i * n + j] = s + b.map_or(0.0, |bb| bb[j] as f64);
        }
    }
    out
}

/// `nn.LayerNorm` over the last dim: biased variance, `eps` added before the
/// sqrt, then affine `y = (x - μ) / √(σ² + ε) · γ + β`.
pub fn layer_norm(x: &[f64], gamma: &[f32], beta: &[f32], n: usize) -> Vec<f64> {
    assert_eq!(gamma.len(), n);
    assert_eq!(beta.len(), n);
    let mut out = vec![0.0f64; x.len()];
    for (row, o) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let mean = row.iter().sum::<f64>() / n as f64;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for j in 0..n {
            o[j] = (row[j] - mean) * inv * gamma[j] as f64 + beta[j] as f64;
        }
    }
    out
}

/// `F.gelu(x, approximate="none")` — exact erf form.
pub fn gelu(x: &mut [f64]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + libm::erf(*v / std::f64::consts::SQRT_2));
    }
}

/// Row-wise softmax with temperature (`-inf` logits stay at 0 probability).
pub fn softmax_temp(logits: &[f64], temp: f64) -> Vec<f64> {
    assert!(temp > 0.0);
    let scaled: Vec<f64> = logits.iter().map(|v| v / temp).collect();
    let max = scaled
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0f64;
    let mut out = Vec::with_capacity(scaled.len());
    for &v in &scaled {
        let e = if v.is_finite() { (v - max).exp() } else { 0.0 };
        sum += e;
        out.push(e);
    }
    for e in &mut out {
        *e /= sum;
    }
    out
}

/// Row-wise log-softmax (masked `-inf` entries stay `-inf`).
pub fn log_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0f64;
    for &v in logits {
        if v.is_finite() {
            sum += (v - max).exp();
        }
    }
    let lsum = sum.ln() + max;
    logits
        .iter()
        .map(|&v| {
            if v.is_finite() {
                v - lsum
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect()
}

/// One row of `create_sin_embedding` (upstream `transformer.py`): fp32
/// sinusoidal position features `[cos(phase), sin(phase)]`, `phase_i =
/// pos / max_period^(i / (half_dim - 1))`. Computed in fp32 like upstream
/// (fp16 positions collapse above 2048; the comment there pins fp32).
pub fn sin_embedding(pos: f32, dim: usize, out: &mut [f32]) {
    assert!(dim % 2 == 0);
    let half = dim / 2;
    for i in 0..half {
        let phase = pos / 10_000f32.powf(i as f32 / (half - 1) as f32);
        out[i] = phase.cos();
        out[half + i] = phase.sin();
    }
}

/// Deterministic xorshift* RNG for `--seed` reproducibility (PROMPT.md §6.3).
/// Not torch's RNG — sampling goldens are distributional, not bit-exact.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Categorical draw from a discrete distribution (unnormalized ok).
    pub fn sample_categorical(&mut self, probs: &[f64]) -> usize {
        let total: f64 = probs.iter().map(|&p| p.max(0.0)).sum();
        let mut target = self.next_f64() * total;
        for (i, &p) in probs.iter().enumerate() {
            target -= p.max(0.0);
            if target < 0.0 {
                return i;
            }
        }
        probs.len() - 1
    }
}

/// `utils.sampling.sample_top_k`: keep probs ≥ the k-th largest (ties kept),
/// renormalize.
pub fn top_k_filter(probs: &[f64], k: usize) -> Vec<f64> {
    let k = k.min(probs.len()).max(1);
    let mut sorted: Vec<f64> = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let thresh = sorted[k - 1];
    renormalize(
        probs,
        &probs.iter().map(|&p| p >= thresh).collect::<Vec<_>>(),
    )
}

/// `utils.sampling.sample_top_p`: the smallest prefix of the descending sort
/// with cumulative mass before each kept element ≤ `p`; remainder renormalized.
pub fn top_p_filter(probs: &[f64], p: f64) -> Vec<f64> {
    let order = argsort_desc(probs);
    let mut cum = 0.0f64;
    let mut keep = vec![false; probs.len()];
    for &i in &order {
        keep[i] = cum <= p;
        cum += probs[i];
    }
    renormalize(probs, &keep)
}

fn renormalize(probs: &[f64], keep: &[bool]) -> Vec<f64> {
    let mut out: Vec<f64> = probs
        .iter()
        .zip(keep)
        .map(|(&p, &k)| if k { p } else { 0.0 })
        .collect();
    let sum: f64 = out.iter().sum();
    if sum > 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

/// Indices sorted descending by value (stable on ties).
pub fn argsort_desc(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_manual() {
        let w = [1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
        let x = [5.0f64, 6.0]; // [1, 2]
        let y = linear(&x, &w, None, 1, 2, 2);
        assert_eq!(y, vec![5.0 + 12.0, 15.0 + 24.0]);
        let yb = linear(&x, &w, Some(&[0.5, -0.5]), 1, 2, 2);
        assert_eq!(yb, vec![17.5, 38.5]);
    }

    #[test]
    fn layernorm_matches_manual() {
        let x = [1.0f64, 2.0, 3.0, 4.0];
        let y = layer_norm(&x, &[1.0f32; 4], &[0.0f32; 4], 4);
        let mean = 2.5;
        let var: f64 = ((1.5 * 1.5) * 2.0 + (0.5 * 0.5) * 2.0) / 4.0;
        for (i, v) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            let want = (v - mean) / (var + 1e-5).sqrt();
            assert!((y[i] - want).abs() < 1e-9);
        }
    }

    #[test]
    fn gelu_exact_points() {
        let mut x = [0.0f64, 1.0, -1.0, 2.0];
        gelu(&mut x);
        assert!(x[0].abs() < 1e-12);
        assert!((x[1] - 0.8413).abs() < 1e-3);
        assert!((x[2] - -0.1587).abs() < 1e-3);
        assert!((x[3] - 1.9545).abs() < 1e-3);
    }

    #[test]
    fn softmax_inf_and_basic() {
        let p = softmax_temp(&[1.0, 2.0, f64::NEG_INFINITY], 1.0);
        assert_eq!(p[2], 0.0);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!((p[1] / p[0] - std::f64::consts::E).abs() < 1e-9);
    }

    #[test]
    fn top_k_keeps_ties() {
        let probs = vec![0.4, 0.3, 0.3, 0.0];
        let f = top_k_filter(&probs, 2); // 2nd largest is 0.3; both 0.3s kept
        assert_eq!(f[3], 0.0);
        assert!(f[0] > 0.0 && f[1] > 0.0 && f[2] > 0.0);
        assert!((f.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn top_p_keeps_smallest_prefix() {
        let probs = vec![0.5, 0.3, 0.15, 0.05];
        let f = top_p_filter(&probs, 0.8); // cum_before: 0, 0.5, 0.8, 0.95 → keep ≤ 0.8
        assert!(f[0] > 0.0 && f[1] > 0.0 && f[2] > 0.0);
        assert_eq!(f[3], 0.0); // 0.05's cum_before 0.95 > 0.8
        assert!((f.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rng_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn sin_embedding_layout() {
        let mut out = [0.0f32; 4];
        sin_embedding(1.0, 4, &mut out);
        // half_dim=2: phase_0 = 1/10000^0 = 1; phase_1 = 1/10000^1
        assert!((out[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((out[1] - (1.0f32 / 10000.0f32).cos()).abs() < 1e-6);
        assert!((out[2] - 1.0f32.sin()).abs() < 1e-6);
    }
}
