//! Experimental spectral-flux onset envelope, a building block for rhythm work.
//!
//! This is not a beat tracker or a calibrated onset detector. In particular,
//! note attacks, drums, vibrato and mixed instruments can all produce peaks.
//! The positive-difference spectral-flux definition is described in the
//! [librosa onset-strength documentation](https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_strength.html).
//! This implementation uses linear-frequency log-magnitude bins instead of
//! librosa's default log-power mel features; no librosa parity is implied.

use crate::{AudioBuffer, MODEL_SAMPLE_RATE};
use mtab_core::{MtabError, Result};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

/// FFT window: 64 ms at 16 kHz.
pub const ONSET_WINDOW: usize = 1024;
/// Frame hop: 10 ms at 16 kHz.
pub const ONSET_HOP: usize = 160;
/// Maximum number of analysis frames (about five minutes).
pub const MAX_ONSET_FRAMES: usize = 30_000;

/// A uniform, unnormalized, nonnegative onset-strength envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetEnvelope {
    /// Mean positive log-magnitude difference from the preceding frame.
    /// The first frame is always zero because no previous frame is available.
    pub strength: Vec<f64>,
}

impl OnsetEnvelope {
    /// Time of a frame's window center, in seconds; `None` for invalid indices.
    pub fn frame_time(&self, index: usize) -> Option<f64> {
        (index < self.strength.len())
            .then(|| (index * ONSET_HOP + ONSET_WINDOW / 2) as f64 / f64::from(MODEL_SAMPLE_RATE))
    }
}

/// Compute positive spectral flux for finite mono 16 kHz audio.
///
/// Frames are non-padded, start at samples `i * 160`, and use a periodic Hann
/// window of 1024 samples. For each real FFT bin `k=0..=512`, compute
/// `log1p(1000 * |FFT[k]| / 1024)`. Frame strength is the arithmetic mean of
/// `max(0, current[k] - previous[k])`; frame zero is defined as zero. Values are
/// not probabilities and change with gain. No adaptive threshold, peak picking,
/// vibrato suppression, meter inference or beat tracking is performed. The
/// complete window is required, so clips shorter than 64 ms return no frames.
/// A transient near either boundary may not be represented fully. Analysis is
/// limited to 30,000 frames and rejects larger inputs instead of silently
/// truncating. This restriction bounds the result, not the input audio buffer.
pub fn spectral_flux(audio: &AudioBuffer) -> Result<OnsetEnvelope> {
    if audio.channels != 1 || audio.sample_rate != MODEL_SAMPLE_RATE {
        return Err(MtabError::UnsupportedAudio(
            "onset analysis requires mono audio at 16000 Hz".into(),
        ));
    }
    let count = if audio.samples.len() < ONSET_WINDOW {
        0
    } else {
        1 + (audio.samples.len() - ONSET_WINDOW) / ONSET_HOP
    };
    if count > MAX_ONSET_FRAMES {
        return Err(MtabError::UnsupportedAudio(format!(
            "onset analysis supports at most {MAX_ONSET_FRAMES} frames; analyze a shorter excerpt"
        )));
    }
    if audio.samples.iter().any(|x| !x.is_finite()) {
        return Err(MtabError::UnsupportedAudio(
            "onset analysis requires finite samples".into(),
        ));
    }
    let mut strength = Vec::with_capacity(count);
    if count == 0 {
        return Ok(OnsetEnvelope { strength });
    }
    let window: Vec<f64> = (0..ONSET_WINDOW)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / ONSET_WINDOW as f64).cos())
        .collect();
    let mut fft = FftPlanner::<f64>::new();
    let forward = fft.plan_fft_forward(ONSET_WINDOW);
    let mut fft_buf = vec![Complex::new(0.0, 0.0); ONSET_WINDOW];
    let mut scratch = vec![Complex::new(0.0, 0.0); forward.get_inplace_scratch_len()];
    let mut previous = vec![0.0; ONSET_WINDOW / 2 + 1];
    for frame in 0..count {
        let start = frame * ONSET_HOP;
        for (i, value) in fft_buf.iter_mut().enumerate() {
            *value = Complex::new(f64::from(audio.samples[start + i]) * window[i], 0.0);
        }
        forward.process_with_scratch(&mut fft_buf, &mut scratch);
        let mut sum = 0.0;
        for (last, value) in previous.iter_mut().zip(&fft_buf) {
            let current = (1000.0 * value.norm() / ONSET_WINDOW as f64).ln_1p();
            if frame != 0 {
                sum += (current - *last).max(0.0);
            }
            *last = current;
        }
        strength.push(sum / previous.len() as f64);
    }
    Ok(OnsetEnvelope { strength })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio(samples: Vec<f32>) -> AudioBuffer {
        AudioBuffer {
            samples,
            sample_rate: 16_000,
            channels: 1,
        }
    }

    #[test]
    fn silence_and_short_inputs_have_no_spurious_flux() {
        let short = spectral_flux(&audio(vec![0.0; ONSET_WINDOW - 1])).unwrap();
        assert!(short.strength.is_empty());
        assert_eq!(short.frame_time(0), None);
        let silence = spectral_flux(&audio(vec![0.0; 4000])).unwrap();
        assert!(silence.strength.len() > 1);
        assert!(silence.strength.iter().all(|&value| value == 0.0));
        assert_eq!(silence.frame_time(0), Some(0.032));
        assert_eq!(silence.frame_time(1), Some(0.042));
        assert_eq!(silence.frame_time(silence.strength.len()), None);
    }

    #[test]
    fn bursts_produce_local_increases_and_are_deterministic() {
        let mut samples = vec![0.0; 16_000];
        for start in [3200, 9600] {
            for i in 0..480 {
                samples[start + i] = if i % 2 == 0 { 0.5 } else { -0.5 };
            }
        }
        let input = audio(samples);
        let flux = spectral_flux(&input).unwrap();
        assert_eq!(flux, spectral_flux(&input).unwrap());
        for onset in [0.2_f64, 0.6] {
            assert!(flux.strength.iter().enumerate().any(|(i, value)| {
                (flux.frame_time(i).unwrap() - onset).abs() < 0.05 && *value > 0.01
            }));
        }
        assert_eq!(flux.strength[0], 0.0);
        assert!(flux.strength.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn polarity_invariance_and_extreme_finite_values() {
        let samples: Vec<f32> = (0..4000)
            .map(|i| {
                if i % 7 == 0 {
                    f32::MAX
                } else {
                    -f32::MAX / 2.0
                }
            })
            .collect();
        let original = spectral_flux(&audio(samples.clone())).unwrap();
        let inverted = spectral_flux(&audio(samples.iter().map(|v| -*v).collect())).unwrap();
        assert_eq!(original, inverted);
        assert!(original.strength.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn invalid_and_oversized_inputs_are_rejected() {
        let mut input = audio(vec![0.0; ONSET_WINDOW]);
        input.channels = 2;
        assert!(spectral_flux(&input).is_err());
        input.channels = 1;
        input.sample_rate = 44_100;
        assert!(spectral_flux(&input).is_err());
        input.sample_rate = 16_000;
        input.samples[10] = f32::NAN;
        assert!(spectral_flux(&input).is_err());
        input.samples = vec![0.0; ONSET_WINDOW + ONSET_HOP * MAX_ONSET_FRAMES];
        assert!(spectral_flux(&input).is_err());
    }
}
