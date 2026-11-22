//! Mel frontend — exact port of the upstream MuScriptor conditioning frontend
//! (MIT, Kyutai & Mirelo; see `tools/upstream/mel_spectrogram_mit.py` for the
//! verbatim source this follows):
//!
//! - `muscriptor/modules/mel_spectrogram.py::melscale_fbanks` — triangular
//!   filterbank, HTK mel scale (`2595*log10(1+f/700)`), `norm=None`,
//!   `f_min=0`, `f_max=sample_rate/2`.
//! - `_MelSpectrogram` — `torch.stft(n_fft, hop, win_length=n_fft,
//!   window=hann (periodic), center=True, pad_mode="reflect",
//!   normalized=False, onesided=True)`, then `|X| ** power` with
//!   **power = 1.0** (magnitude, not energy), then `spec @ fb`.
//! - `MelSpectrogramConditioner._mel_embedding` — final features are
//!   **`log(mel + 1e-6)`**.
//!
//! Model config (`transcription_model.py`): sample_rate 16000, n_fft 2048,
//! frame_rate 100 (→ hop 160), n_mel_bins 512, log_scale true, eps 1e-6,
//! normalize_audio FALSE. Parity goldens: `tools/gen_mel_fixtures.py` +
//! `tests/mel_parity.rs` (PROMPT.md §6.2 — the sacred gate).

use mtab_core::{MtabError, Result};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Upstream `eps` in `torch.log(mel + eps)`.
pub const UPSTREAM_EPS: f32 = 1e-6;

/// Mel frontend configuration. [`MelConfig::upstream`] matches the values the
/// released MuScriptor models were trained with.
#[derive(Debug, Clone, PartialEq)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    /// Frames per second of audio → hop = sample_rate / frame_rate.
    pub frame_rate: u32,
    pub n_mels: usize,
    /// STFT magnitude exponent (upstream uses 1.0 = magnitude).
    pub power: f32,
    /// Apply `log(mel + eps)` at the end (upstream: true).
    pub log_scale: bool,
    pub eps: f32,
}

impl MelConfig {
    /// The exact configuration of the released MuScriptor models.
    pub fn upstream() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 2048,
            frame_rate: 100,
            n_mels: 512,
            power: 1.0,
            log_scale: true,
            eps: UPSTREAM_EPS,
        }
    }

    /// Hop length in samples (160 at 16 kHz / 100 fps).
    pub fn hop_length(&self) -> usize {
        (self.sample_rate / self.frame_rate) as usize
    }
}

impl Default for MelConfig {
    fn default() -> Self {
        Self::upstream()
    }
}

/// HTK Hz→mel (`modules/mel_spectrogram.py::_hz_to_mel_htk`).
pub fn hz_to_mel_htk(freq: f64) -> f64 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

/// HTK mel→Hz (`modules/mel_spectrogram.py::_mel_to_hz_htk`).
pub fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![a];
    }
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

/// Triangular mel filterbank `[n_freqs][n_mels]`, line-for-line port of
/// `modules/mel_spectrogram.py::melscale_fbanks` (HTK, norm=None).
pub fn melscale_fbanks(
    n_freqs: usize,
    f_min: f64,
    f_max: f64,
    n_mels: usize,
    sample_rate: u32,
) -> Vec<Vec<f64>> {
    let all_freqs = linspace(0.0, sample_rate as f64 / 2.0, n_freqs);
    let m_min = hz_to_mel_htk(f_min);
    let m_max = hz_to_mel_htk(f_max);
    let m_pts = linspace(m_min, m_max, n_mels + 2);
    let f_pts: Vec<f64> = m_pts.iter().map(|&m| mel_to_hz_htk(m)).collect();
    let f_diff: Vec<f64> = (0..n_mels + 1).map(|i| f_pts[i + 1] - f_pts[i]).collect();

    let mut fb = vec![vec![0.0f64; n_mels]; n_freqs];
    for (fi, &freq) in all_freqs.iter().enumerate() {
        for j in 0..n_mels {
            let down_slopes = -(f_pts[j] - freq) / f_diff[j];
            let up_slopes = (f_pts[j + 2] - freq) / f_diff[j + 1];
            fb[fi][j] = down_slopes.min(up_slopes).max(0.0);
        }
    }
    fb
}

/// Reusable mel frontend (filterbank + window + FFT plan precomputed).
pub struct MelFrontend {
    cfg: MelConfig,
    fb: Vec<Vec<f64>>,
    window: Vec<f64>,
    fft: std::sync::Arc<dyn rustfft::Fft<f64>>,
}

impl MelFrontend {
    pub fn new(cfg: MelConfig) -> Self {
        let n_freqs = cfg.n_fft / 2 + 1;
        let fb = melscale_fbanks(
            n_freqs,
            0.0,
            cfg.sample_rate as f64 / 2.0,
            cfg.n_mels,
            cfg.sample_rate,
        );
        // torch.hann_window(n_fft) is PERIODIC: w[i] = 0.5*(1-cos(2*pi*i/n_fft)).
        let window: Vec<f64> = (0..cfg.n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / cfg.n_fft as f64).cos()))
            .collect();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(cfg.n_fft);
        Self {
            cfg,
            fb,
            window,
            fft,
        }
    }

    /// The upstream configuration.
    pub fn upstream() -> Self {
        Self::new(MelConfig::upstream())
    }

    /// Build a frontend from a checkpoint's stored `spectrogram.window`
    /// `[n_fft]` and `mel_scale.fb` `[n_freqs, n_mels]` tensors (f32 values as
    /// saved). Upstream constructs these buffers in torch's default f32
    /// dtype, so released checkpoints carry f32-rounded filterbanks — model
    /// parity requires using the stored values, not recomputing in f64.
    pub fn with_bank(cfg: MelConfig, window: &[f32], fb_flat: &[f32]) -> Result<Self> {
        let n_fft = cfg.n_fft;
        let n_freqs = n_fft / 2 + 1;
        if window.len() != n_fft {
            return Err(MtabError::ShapeMismatch {
                expected: format!("`spectrogram.window` [{n_fft}]"),
                found: format!("[{}]", window.len()),
            });
        }
        if fb_flat.len() != n_freqs * cfg.n_mels {
            return Err(MtabError::ShapeMismatch {
                expected: format!("`mel_scale.fb` [{n_freqs}, {}]", cfg.n_mels),
                found: format!("[{}]", fb_flat.len()),
            });
        }
        let fb = fb_flat
            .chunks_exact(cfg.n_mels)
            .map(|row| row.iter().map(|&v| v as f64).collect())
            .collect();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);
        Ok(Self {
            cfg,
            fb,
            window: window.iter().map(|&v| v as f64).collect(),
            fft,
        })
    }

    pub fn config(&self) -> &MelConfig {
        &self.cfg
    }

    /// Frame count for `len` input samples (center=True: `len / hop + 1`).
    pub fn n_frames(&self, len: usize) -> usize {
        len / self.cfg.hop_length() + 1
    }

    /// Pre-log mel magnitudes `[frames][n_mels]` (STFT power per config,
    /// upstream power = 1.0, then `spec @ fb`).
    pub fn compute_mel(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let n_fft = self.cfg.n_fft;
        let p = n_fft / 2;
        let n = samples.len();
        assert!(
            n > p,
            "input too short for reflect padding: {} samples <= n_fft/2 = {p}",
            n
        );

        // reflect-pad p on both sides (torch pad_mode="reflect": mirror about
        // the edge samples WITHOUT repeating them).
        let mut padded = vec![0.0f64; n + 2 * p];
        for i in 0..p {
            padded[i] = samples[p - i] as f64; // left: x[p], x[p-1], ..., x[1]
            padded[p + n + i] = samples[n - 2 - i] as f64; // right: x[n-2], ..., x[n-p-1]
        }
        for (i, &s) in samples.iter().enumerate() {
            padded[p + i] = s as f64;
        }

        let hop = self.cfg.hop_length();
        let n_frames = self.n_frames(n);
        let n_freqs = n_fft / 2 + 1;
        let mut out = Vec::with_capacity(n_frames);
        let mut buf = vec![Complex::new(0.0f64, 0.0f64); n_fft];

        for t in 0..n_frames {
            let start = t * hop;
            for (i, c) in buf.iter_mut().enumerate() {
                *c = Complex::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);

            let mut frame = vec![0.0f64; self.cfg.n_mels];
            for (fi, fb_row) in self.fb.iter().enumerate().take(n_freqs) {
                let mag = (buf[fi].re * buf[fi].re + buf[fi].im * buf[fi].im).sqrt();
                let mag = mag.powf(self.cfg.power as f64);
                if mag == 0.0 {
                    continue;
                }
                for (j, &w) in fb_row.iter().enumerate() {
                    frame[j] += mag * w;
                }
            }
            out.push(frame.into_iter().map(|v| v as f32).collect());
        }
        out
    }

    /// Model-input features: `log(mel + eps)` when [`MelConfig::log_scale`]
    /// (the exact `MelSpectrogramConditioner._mel_embedding` output).
    pub fn compute(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mel = self.compute_mel(samples);
        if !self.cfg.log_scale {
            return mel;
        }
        let eps = self.cfg.eps;
        mel.into_iter()
            .map(|frame| frame.into_iter().map(|v| (v + eps).ln()).collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hz_mel_round_trip() {
        for f in [0.0, 100.0, 440.0, 1000.0, 8000.0] {
            let back = mel_to_hz_htk(hz_to_mel_htk(f));
            assert!((back - f).abs() < 1e-6, "f={f} back={back}");
        }
        // HTK constant: mel(700) = 2595*log10(2).
        assert!((hz_to_mel_htk(700.0) - 2595.0 * 2f64.log10()).abs() < 1e-9);
    }

    #[test]
    fn fbanks_match_upstream_construction() {
        let fb = melscale_fbanks(1025, 0.0, 8000.0, 512, 16_000);
        assert_eq!(fb.len(), 1025);
        assert_eq!(fb[0].len(), 512);
        // Non-negative triangles. Known upstream quirk: with f_min=0 on a
        // 1025-bin grid (7.8 Hz spacing), the first few triangles (centers
        // below ~one bin spacing) can miss every FFT bin and end up all-zero —
        // torchaudio/upstream compute exactly this. All degenerate filters
        // must live at the low end; the vast majority carry mass.
        let mut empty = Vec::new();
        for j in 0..512 {
            for row in &fb {
                assert!(row[j] >= 0.0);
            }
            let mass: f64 = fb.iter().map(|row| row[j]).sum();
            if mass == 0.0 {
                empty.push(j);
            }
        }
        assert!(
            empty.len() < 32 && empty.iter().all(|&j| j < 32),
            "unexpected degenerate filters: {empty:?}"
        );
        assert!(
            fb.iter().all(|row| row[0] == 0.0),
            "filter 0 should be degenerate"
        );
        // First/last bins: DC sees only zero filter; Nyquist sees no filter.
        assert_eq!(fb[0][511], 0.0);
    }

    #[test]
    fn frame_count_matches_center_padding() {
        let fe = MelFrontend::upstream();
        assert_eq!(fe.n_frames(80_000), 501); // 5 s segment: 80000/160+1
        assert_eq!(fe.n_frames(16_000), 101);
        let mel = fe.compute_mel(&vec![0.0f32; 16_000]);
        assert_eq!(mel.len(), 101);
        assert_eq!(mel[0].len(), 512);
    }

    #[test]
    fn sine_peaks_at_expected_mel_bin() {
        let fe = MelFrontend::upstream();
        let sr = 16_000.0f64;
        let n = 16_000usize;
        let freq = 440.0f64;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin() as f32)
            .collect();
        let mel = fe.compute_mel(&samples);
        // Sum over time, find argmax mel bin.
        let mut energy = vec![0.0f64; 512];
        for frame in &mel {
            for (j, v) in frame.iter().enumerate() {
                energy[j] += *v as f64;
            }
        }
        let argmax = energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // Expected bin: triangle center closest to mel(440).
        let m_pts = linspace(0.0, hz_to_mel_htk(8000.0), 514);
        let expected = m_pts
            .iter()
            .skip(1)
            .take(512)
            .enumerate()
            .min_by(|a, b| {
                (a.1 - hz_to_mel_htk(freq))
                    .abs()
                    .partial_cmp(&(b.1 - hz_to_mel_htk(freq)).abs())
                    .unwrap()
            })
            .unwrap()
            .0;
        assert!(
            (argmax as i64 - expected as i64).abs() <= 1,
            "argmax={argmax} expected={expected}"
        );
    }

    #[test]
    fn log_scale_floor_matches_upstream_eps() {
        let fe = MelFrontend::upstream();
        let mel = fe.compute(&vec![0.0f32; 4096]);
        let expected = (1e-6f32).ln();
        for v in mel[0].iter().take(8) {
            assert!((v - expected).abs() < 1e-4);
        }
    }
}
