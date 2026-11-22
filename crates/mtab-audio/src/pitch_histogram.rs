//! Preliminary monophonic pitch evidence for tuning analysis.
//!
//! A bounded set of frames is analysed using an FFT-accelerated YIN-style
//! difference function. This is not a polyphonic transcription or source
//! separator: chords, bass and percussion can create spurious fundamentals.
//! The histogram represents evidence, not guaranteed open-string pitches.

use mtab_core::{MtabError, Result};
use rustfft::{num_complex::Complex, FftPlanner};

use crate::{AudioBuffer, MODEL_SAMPLE_RATE};

const WINDOW: usize = 2048;
const FFT_SIZE: usize = WINDOW * 2;
const HOP: usize = 1024;
const MAX_FRAMES: usize = 512;
const THRESHOLD: f64 = 0.15;

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
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let inverse = planner.plan_fft_inverse(FFT_SIZE);
    let mut spectrum = vec![Complex::new(0.0, 0.0); FFT_SIZE];
    let mut cumulative = vec![0.0; WINDOW + 1];
    let mut cmnd = vec![1.0; WINDOW + 1];
    let min_tau = (MODEL_SAMPLE_RATE / 1000) as usize;
    let max_tau = (MODEL_SAMPLE_RATE / 38) as usize;
    for i in 0..count {
        let frame_index = if count == 1 {
            0
        } else {
            ((i as u128 * (available - 1) as u128) / (count - 1) as u128) as usize
        };
        let start = frame_index * HOP;
        let frame = &audio.samples[start..start + WINDOW];
        let mean = frame.iter().map(|&x| f64::from(x)).sum::<f64>() / WINDOW as f64;
        cumulative[0] = 0.0;
        for (j, &value) in frame.iter().enumerate() {
            let x = f64::from(value) - mean;
            spectrum[j] = Complex::new(x, 0.0);
            cumulative[j + 1] = cumulative[j] + x * x;
        }
        spectrum[WINDOW..].fill(Complex::new(0.0, 0.0));
        result.frames_examined += 1;
        if cumulative[WINDOW] / (WINDOW as f64) < 1e-8 {
            continue;
        }
        fft.process(&mut spectrum);
        for value in &mut spectrum {
            *value = Complex::new(value.norm_sqr(), 0.0);
        }
        inverse.process(&mut spectrum);
        cmnd[0] = 1.0;
        let mut running = 0.0;
        for tau in 1..=max_tau + 1 {
            let autocorrelation = spectrum[tau].re / FFT_SIZE as f64;
            let diff = (cumulative[WINDOW - tau] + cumulative[WINDOW]
                - cumulative[tau]
                - 2.0 * autocorrelation)
                .max(0.0);
            running += diff;
            cmnd[tau] = if running > 0.0 {
                diff * tau as f64 / running
            } else {
                1.0
            };
        }
        let mut tau = min_tau;
        while tau <= max_tau && cmnd[tau] >= THRESHOLD {
            tau += 1;
        }
        if tau > max_tau {
            continue;
        }
        while tau < max_tau && cmnd[tau + 1] < cmnd[tau] {
            tau += 1;
        }
        // Parabolic sub-sample interpolation of the difference minimum.
        let left = cmnd[tau - 1];
        let middle = cmnd[tau];
        let right = cmnd[tau + 1];
        let denominator = left - 2.0 * middle + right;
        let offset = if denominator.abs() > 1e-12 {
            (0.5 * (left - right) / denominator).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let frequency = f64::from(MODEL_SAMPLE_RATE) / (tau as f64 + offset);
        let midi = (69.0 + 12.0 * (frequency / 440.0).log2()).round();
        if !(0.0..=127.0).contains(&midi) {
            continue;
        }
        result.weights[midi as usize] += (1.0 - middle).clamp(0.0, 1.0);
        result.voiced_frames += 1;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
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
