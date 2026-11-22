//! Preliminary monophonic pitch evidence for tuning analysis.
//!
//! A bounded set of frames is analysed using an FFT-accelerated YIN-style
//! difference function. This is not a polyphonic transcription or source
//! separator: chords, bass and percussion can create spurious fundamentals.
//! The histogram represents evidence, not guaranteed open-string pitches.

use crate::{
    pitch_track::{YinDetector, WINDOW},
    AudioBuffer, MODEL_SAMPLE_RATE,
};
use mtab_core::{MtabError, Result};

const HOP: usize = 1024;
const MAX_FRAMES: usize = 512;

/// MIDI-pitch evidence and coverage from a monophonic audio analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct PitchHistogram {
    /// Sum of confidence weights per nearest integer MIDI pitch (0–127).
    pub weights: [f64; 128],
    /// Total frames examined, including unvoiced/silent frames.
    pub frames_examined: usize,
    /// Frames whose periodicity passed the threshold and pitch range checks.
    pub voiced_frames: usize,
}

/// Extract up to 512 uniformly spaced mono 16 kHz frames from a recording.
///
/// The window is 128 ms, analysed every 64 ms before uniform subsampling;
/// tested fundamentals range from approximately 38 Hz to 1 kHz. Returns empty
/// evidence for silence or inputs shorter than one analysis window. Frame
/// weights are periodicity scores, not calibrated confidence probabilities.
/// Strongly polyphonic or noisy recordings can give misleading estimates.
pub fn pitch_histogram(audio: &AudioBuffer) -> Result<PitchHistogram> {
    if audio.channels != 1
        || audio.sample_rate != MODEL_SAMPLE_RATE
        || audio.samples.iter().any(|s| !s.is_finite())
    {
        return Err(MtabError::UnsupportedAudio(
            "pitch analysis requires finite mono audio at 16000 Hz".into(),
        ));
    }
    let mut result = PitchHistogram {
        weights: [0.0; 128],
        frames_examined: 0,
        voiced_frames: 0,
    };
    if audio.samples.len() < WINDOW {
        return Ok(result);
    }
    let available = 1 + (audio.samples.len() - WINDOW) / HOP;
    let count = available.min(MAX_FRAMES);
    let mut detector = YinDetector::new();
    for i in 0..count {
        let frame_index = if count == 1 {
            0
        } else {
            ((i as u128 * (available - 1) as u128) / (count - 1) as u128) as usize
        };
        let start = frame_index * HOP;
        result.frames_examined += 1;
        if let Some((frequency, confidence)) =
            detector.analyze(&audio.samples[start..start + WINDOW])
        {
            let midi = (69.0 + 12.0 * (frequency / 440.0).log2()).round();
            if (0.0..=127.0).contains(&midi) {
                result.weights[midi as usize] += confidence;
                result.voiced_frames += 1;
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::{num_complex::Complex, FftPlanner};
    use std::f64::consts::PI;
    const FFT_SIZE: usize = WINDOW * 2;
    fn sine(midi: u8, seconds: f64) -> AudioBuffer {
        let hz = 440.0 * 2.0f64.powf((f64::from(midi) - 69.0) / 12.0);
        let samples = (0..(seconds * 16000.0) as usize)
            .map(|i| (0.4 * (2.0 * PI * hz * i as f64 / 16000.0).sin()) as f32)
            .collect();
        AudioBuffer {
            samples,
            sample_rate: 16000,
            channels: 1,
        }
    }
    #[test]
    fn detects_guitar_range_sine_pitches() {
        for midi in [30u8, 35, 40, 45, 50, 55, 59, 64, 72, 81] {
            let h = pitch_histogram(&sine(midi, 0.5)).unwrap();
            let strongest = h
                .weights
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            assert_eq!(strongest, midi as usize, "pitch {midi}");
            assert_eq!(h.voiced_frames, h.frames_examined);
        }
    }
    #[test]
    fn silence_short_invalid_and_constant_dc() {
        let mut a = AudioBuffer {
            samples: vec![0.0; 10000],
            sample_rate: 16000,
            channels: 1,
        };
        let h = pitch_histogram(&a).unwrap();
        assert!(h.frames_examined > 0);
        assert_eq!(h.voiced_frames, 0);
        a.samples.fill(0.4);
        assert_eq!(pitch_histogram(&a).unwrap().voiced_frames, 0);
        a.samples.truncate(100);
        assert_eq!(pitch_histogram(&a).unwrap().frames_examined, 0);
        a.channels = 2;
        assert!(pitch_histogram(&a).is_err());
        a.channels = 1;
        a.samples[0] = f32::NAN;
        assert!(pitch_histogram(&a).is_err());
    }
    #[test]
    fn sampled_frame_count_is_bounded_and_reproducible() {
        let a = sine(40, 40.0);
        let h = pitch_histogram(&a).unwrap();
        assert_eq!(h.frames_examined, MAX_FRAMES);
        assert_eq!(h, pitch_histogram(&a).unwrap());
    }
    #[test]
    fn harmonic_signal_and_aperiodic_noise() {
        let mut harmonic = sine(40, 0.6);
        let overtone = sine(52, 0.6);
        for (x, y) in harmonic.samples.iter_mut().zip(overtone.samples) {
            *x += 0.6 * y;
        }
        let h = pitch_histogram(&harmonic).unwrap();
        assert_eq!(
            h.weights
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
            40
        );
        let mut seed = 3u32;
        for sample in &mut harmonic.samples {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *sample = (seed as f64 / f64::from(u32::MAX) * 2.0 - 1.0) as f32;
        }
        assert_eq!(pitch_histogram(&harmonic).unwrap().voiced_frames, 0);
    }

    #[test]
    fn fft_difference_matches_naive_sum() {
        let mut planner = FftPlanner::<f64>::new();
        let mut x: Vec<f64> = (0..WINDOW)
            .map(|i| ((i as f64 * 0.031).sin() + (i as f64 * 0.071).cos()) * 0.3)
            .collect();
        let mean = x.iter().sum::<f64>() / WINDOW as f64;
        for v in &mut x {
            *v -= mean;
        }
        let mut spectrum = vec![Complex::new(0.0, 0.0); FFT_SIZE];
        for (s, &v) in spectrum.iter_mut().zip(&x) {
            s.re = v;
        }
        planner.plan_fft_forward(FFT_SIZE).process(&mut spectrum);
        for s in &mut spectrum {
            *s = Complex::new(s.norm_sqr(), 0.0);
        }
        planner.plan_fft_inverse(FFT_SIZE).process(&mut spectrum);
        for tau in [1usize, 16, 100, 421] {
            let naive: f64 = (0..WINDOW - tau).map(|i| (x[i] - x[i + tau]).powi(2)).sum();
            let energy: f64 = x[..WINDOW - tau]
                .iter()
                .chain(&x[tau..])
                .map(|v| v * v)
                .sum();
            let fft_diff = energy - 2.0 * spectrum[tau].re / FFT_SIZE as f64;
            assert!((fft_diff - naive).abs() < 1e-8, "lag {tau}");
        }
    }
}
