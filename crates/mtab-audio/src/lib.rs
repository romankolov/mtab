//! `mtab-audio` — audio ingest and feature extraction (PROMPT.md §6.1–§6.2).
//!
//! Pipeline: decode (symphonia) → mono mixdown → DC removal → peak normalize
//! (−1 dBFS) → resample to the model sample rate (16 kHz per upstream
//! `muscriptor/transcription_model.py`) → mel spectrogram matching the
//! upstream frontend EXACTLY (`muscriptor/modules/mel_spectrogram.py`, MIT):
//! HTK mel scale, `norm=None`, periodic Hann window, `win_length == n_fft`,
//! reflect-pad centering, power 2.0, one-sided unnormalized STFT.
//!
//! The mel frontend lands in M2b; parity goldens against upstream are gated
//! on `tests/fixtures/` (generate with `tools/gen_mel_fixtures.py`).

pub mod decode;
pub mod dsp;
pub mod resample;

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

/// Full ingest pipeline (§6.1): file → mono 16 kHz f32 ready for features.
pub fn load_model_input(path: &std::path::Path) -> mtab_core::Result<AudioBuffer> {
    let buf = decode::load(path)?;
    let mono = dsp::to_mono(&buf);
    let mut mono = mono;
    dsp::remove_dc(&mut mono);
    dsp::peak_normalize(&mut mono, -1.0);
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
