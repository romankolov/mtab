//! `mtab-audio` — audio ingest and feature extraction (PROMPT.md §6.1–§6.2).
//!
//! Pipeline: decode (symphonia) → mono mixdown → resample to the model
//! sample rate (16 kHz per upstream
//! `muscriptor/transcription_model.py`) → mel spectrogram matching the
//! upstream frontend EXACTLY (`muscriptor/modules/mel_spectrogram.py`, MIT):
//! HTK mel scale, `norm=None`, periodic Hann window, `win_length == n_fft`,
//! reflect-pad centering, power 2.0, one-sided unnormalized STFT.
//!
//! The mel frontend ([`mel`]) is ported EXACTLY from upstream and verified by
//! golden parity tests against the verbatim MIT reference
//! (`tools/upstream/mel_spectrogram_mit.py`, torch float64 goldens generated
//! by `tools/gen_mel_fixtures.py`); measured max relative error ~4e-12.

pub mod bend_analysis;
pub mod decode;
pub mod dsp;
pub mod dynamics;
pub mod mel;
pub mod pitch_histogram;
pub mod pitch_track;
pub mod resample;
pub mod transcribe;

#[cfg(test)]
pub mod test_util;

/// A decoded audio buffer (interleaved frames × channels).
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    /// Interleaved samples, `frames × channels` long.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count (1 after mono mixdown).
    pub channels: u16,
}

impl AudioBuffer {
    /// Number of frames per channel.
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }

    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }
}

/// Model input sample rate: upstream `_SAMPLE_RATE = 16000`.
pub const MODEL_SAMPLE_RATE: u32 = 16_000;

/// Full ingest pipeline, matching upstream `utils/audio.py::load_audio`
/// semantics EXACTLY (§2: port, don't guess): decode → **mono mean** →
/// resample to 16 kHz. Upstream does NOT normalize or remove DC — the
/// released models were trained with `normalize_audio=False`, so the default
/// path must not normalize either (parity gate, §6.2). [`dsp::peak_normalize`]
/// and [`dsp::remove_dc`] remain available as optional preprocessing.
pub fn load_model_input(path: &std::path::Path) -> mtab_core::Result<AudioBuffer> {
    let buf = decode::load(path)?;
    let mono = dsp::to_mono(&buf);
    resample::resample(&mono, MODEL_SAMPLE_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_and_duration() {
        let buf = AudioBuffer {
            samples: vec![0.0; 32_000],
            sample_rate: 16_000,
            channels: 1,
        };
        assert_eq!(buf.frames(), 32_000);
        assert!((buf.duration_secs() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn model_sample_rate_matches_upstream() {
        assert_eq!(MODEL_SAMPLE_RATE, 16_000);
    }

    #[test]
    fn end_to_end_load_model_input() {
        let path = test_util::write_wav(&vec![0.25f32; 4_800], 48_000);
        let buf = load_model_input(&path).unwrap();
        assert_eq!(buf.sample_rate, 16_000);
        assert_eq!(buf.channels, 1);
        assert!(
            (buf.frames() as i64 - 1_600).abs() <= 16,
            "frames={}",
            buf.frames()
        );
        std::fs::remove_file(&path).ok();
    }
}
