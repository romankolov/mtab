//! Experimental monophonic F0 tracking for later technique analysis.
//!
//! FFT-accelerated YIN-style periodicity, not a polyphonic pitch separator.
//! Only fully covered 128 ms frames are analyzed; times refer to frame centers.
//! It can confidently follow the wrong note in mixtures or octave-ambiguous
//! audio. No bends or other techniques are inferred by this module.
use crate::{AudioBuffer, MODEL_SAMPLE_RATE};
use mtab_core::{MtabError, Result};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

pub(crate) const WINDOW: usize = 2048;
const FFT_SIZE: usize = WINDOW * 2;
const THRESHOLD: f64 = 0.15;
/// Hop between F0 frame centers at 16 kHz (16 ms).
pub const HOP: usize = 256;
const MAX_FRAMES: usize = 10_000;

/// One fully contained analysis frame. Periodicity is not a calibrated probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchFrame {
    /// Center time in seconds, relative to the input audio.
    pub time: f64,
    /// Fundamental in Hz when periodicity passes the voicing threshold.
    pub frequency_hz: Option<f64>,
    /// Estimated normalized periodicity, zero for unvoiced frames.
    pub periodicity: f64,
}

pub(crate) struct YinDetector {
    fft: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    spectrum: Vec<Complex<f64>>,
    cumulative: Vec<f64>,
    cmnd: Vec<f64>,
}
impl YinDetector {
    pub(crate) fn new() -> Self {
        let mut planner = FftPlanner::<f64>::new();
        Self {
            fft: planner.plan_fft_forward(FFT_SIZE),
            inverse: planner.plan_fft_inverse(FFT_SIZE),
            spectrum: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            cumulative: vec![0.0; WINDOW + 1],
            cmnd: vec![1.0; WINDOW + 1],
        }
    }
    pub(crate) fn analyze(&mut self, frame: &[f32]) -> Option<(f64, f64)> {
        debug_assert_eq!(frame.len(), WINDOW);
        let mean = frame.iter().map(|&x| f64::from(x)).sum::<f64>() / WINDOW as f64;
        self.cumulative[0] = 0.0;
        for (i, &value) in frame.iter().enumerate() {
            let x = f64::from(value) - mean;
            self.spectrum[i] = Complex::new(x, 0.0);
            self.cumulative[i + 1] = self.cumulative[i] + x * x;
        }
        self.spectrum[WINDOW..].fill(Complex::new(0.0, 0.0));
        if self.cumulative[WINDOW] / (WINDOW as f64) < 1e-8 {
            return None;
        }
        self.fft.process(&mut self.spectrum);
        for value in &mut self.spectrum {
            *value = Complex::new(value.norm_sqr(), 0.0);
        }
        self.inverse.process(&mut self.spectrum);
        let min_tau = (MODEL_SAMPLE_RATE / 1000) as usize;
        let max_tau = (MODEL_SAMPLE_RATE / 38) as usize;
        let mut running = 0.0;
        self.cmnd[0] = 1.0;
        for tau in 1..=max_tau + 1 {
            let autocorrelation = self.spectrum[tau].re / FFT_SIZE as f64;
            let diff = (self.cumulative[WINDOW - tau] + self.cumulative[WINDOW]
                - self.cumulative[tau]
                - 2.0 * autocorrelation)
                .max(0.0);
            running += diff;
            self.cmnd[tau] = if running > 0.0 {
                diff * tau as f64 / running
            } else {
                1.0
            };
        }
        let mut tau = min_tau;
        while tau <= max_tau && self.cmnd[tau] >= THRESHOLD {
            tau += 1;
        }
        if tau > max_tau {
            return None;
        }
        while tau < max_tau && self.cmnd[tau + 1] < self.cmnd[tau] {
            tau += 1;
        }
        let (left, mid, right) = (self.cmnd[tau - 1], self.cmnd[tau], self.cmnd[tau + 1]);
        let denominator = left - 2.0 * mid + right;
        let offset = if denominator.abs() > 1e-12 {
            (0.5 * (left - right) / denominator).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        Some((
            f64::from(MODEL_SAMPLE_RATE) / (tau as f64 + offset),
            (1.0 - mid).clamp(0.0, 1.0),
        ))
    }
}

/// Estimate F0 in frames fully contained in `[start, end]` seconds.
///
/// Input must be finite mono 16 kHz audio, with ordered bounds inside the audio.
/// At most 10,000 frames may be requested at a time. Short regions return an
/// empty vector; silent/noisy frames have no frequency. Approximately 38 Hz to
/// 1 kHz is searched. The 128 ms window smooths fast bends and attacks, and
/// there is no ability to isolate one note from polyphonic audio.
pub fn pitch_track(audio: &AudioBuffer, start: f64, end: f64) -> Result<Vec<PitchFrame>> {
    PitchTracker::new(audio)?.track(start, end)
}

/// Reusable tracker for many note regions in one immutable audio buffer.
/// It validates the audio once and reuses FFT plans and analysis arrays.
pub struct PitchTracker<'a> {
    audio: &'a AudioBuffer,
    detector: YinDetector,
}
impl<'a> PitchTracker<'a> {
    /// Validate and prepare a finite, mono 16 kHz buffer.
    pub fn new(audio: &'a AudioBuffer) -> Result<Self> {
        if audio.channels != 1
            || audio.sample_rate != MODEL_SAMPLE_RATE
            || audio.samples.iter().any(|s| !s.is_finite())
        {
            return Err(MtabError::UnsupportedAudio(
                "pitch tracking requires finite mono audio at 16000 Hz".into(),
            ));
        }
        Ok(Self {
            audio,
            detector: YinDetector::new(),
        })
    }
    /// Track a region. Bounds and coverage are identical to [`pitch_track`].
    pub fn track(&mut self, start: f64, end: f64) -> Result<Vec<PitchFrame>> {
        let audio = self.audio;
        let duration = audio.samples.len() as f64 / f64::from(MODEL_SAMPLE_RATE);
        if !start.is_finite() || !end.is_finite() || start < 0.0 || end < start || end > duration {
            return Err(MtabError::UnsupportedAudio(
                "pitch-tracking bounds must be finite, ordered, and inside the audio".into(),
            ));
        }
        let first =
            ((start * f64::from(MODEL_SAMPLE_RATE)).ceil() as usize).min(audio.samples.len());
        let last = ((end * f64::from(MODEL_SAMPLE_RATE)).floor() as usize).min(audio.samples.len());
        let count = if last.saturating_sub(first) < WINDOW {
            0
        } else {
            1 + (last - first - WINDOW) / HOP
        };
        if count > MAX_FRAMES {
            return Err(MtabError::UnsupportedAudio(
                "pitch-tracking region exceeds 10000 frames; split it into smaller regions".into(),
            ));
        }
        let mut frames = Vec::with_capacity(count);
        for i in 0..count {
            let offset = first + i * HOP;
            let result = self
                .detector
                .analyze(&audio.samples[offset..offset + WINDOW]);
            frames.push(PitchFrame {
                time: (offset + WINDOW / 2) as f64 / f64::from(MODEL_SAMPLE_RATE),
                frequency_hz: result.map(|v| v.0),
                periodicity: result.map_or(0.0, |v| v.1),
            });
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    fn swept_sine(start_hz: f64, cents_per_second: f64, seconds: f64) -> AudioBuffer {
        let mut phase = 0.0;
        let samples = (0..(seconds * 16000.0) as usize)
            .map(|i| {
                let t = i as f64 / 16000.0;
                let hz = start_hz * 2.0f64.powf(cents_per_second * t / 1200.0);
                phase += 2.0 * PI * hz / 16000.0;
                (phase.sin() * 0.4) as f32
            })
            .collect();
        AudioBuffer {
            samples,
            sample_rate: 16000,
            channels: 1,
        }
    }
    fn cents(actual: f64, expected: f64) -> f64 {
        1200.0 * (actual / expected).log2()
    }
    #[test]
    fn sine_pitch_accuracy_and_frame_center_bounds() {
        for hz in [55.0, 82.406889, 110.0, 220.0, 440.0, 880.0] {
            let audio = swept_sine(hz, 0.0, 0.8);
            let frames = pitch_track(&audio, 0.1, 0.7).unwrap();
            assert!(!frames.is_empty());
            for frame in &frames {
                assert!(frame.time - 0.064 >= 0.1 - 1e-12);
                assert!(frame.time + 0.064 <= 0.7 + 1e-12);
                assert!(
                    cents(frame.frequency_hz.unwrap(), hz).abs() < 8.0,
                    "{hz} {frame:?}"
                );
                assert!(frame.periodicity > 0.9);
            }
            assert_eq!(frames, pitch_track(&audio, 0.1, 0.7).unwrap());
            let mut tracker = PitchTracker::new(&audio).unwrap();
            assert_eq!(frames, tracker.track(0.1, 0.7).unwrap());
            assert_eq!(frames, tracker.track(0.1, 0.7).unwrap());
        }
    }
    #[test]
    fn smoothly_tracks_synthetic_bend_without_using_outside_samples() {
        let audio = swept_sine(110.0, 200.0, 1.0);
        let frames = pitch_track(&audio, 0.25, 0.75).unwrap();
        assert!(frames.len() > 10);
        for frame in &frames {
            let target = 110.0 * 2.0f64.powf(200.0 * frame.time / 1200.0);
            assert!(
                cents(frame.frequency_hz.unwrap(), target).abs() < 12.0,
                "{frame:?}"
            );
        }
        assert!(frames.last().unwrap().frequency_hz.unwrap() > frames[0].frequency_hz.unwrap());
        let mut altered = audio.clone();
        altered.samples[..4000].fill(0.9);
        altered.samples[12000..].fill(-0.9);
        assert_eq!(frames, pitch_track(&altered, 0.25, 0.75).unwrap());
    }
    #[test]
    fn short_silence_noise_and_invalid_inputs() {
        let mut audio = AudioBuffer {
            samples: vec![0.0; 16000],
            sample_rate: 16000,
            channels: 1,
        };
        assert!(pitch_track(&audio, 0.0, 0.1).unwrap().is_empty());
        assert!(pitch_track(&audio, 0.0, 1.0)
            .unwrap()
            .iter()
            .all(|f| f.frequency_hz.is_none() && f.periodicity == 0.0));
        let mut seed = 5u32;
        for sample in &mut audio.samples {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *sample = (seed as f64 / f64::from(u32::MAX) * 2.0 - 1.0) as f32;
        }
        assert!(pitch_track(&audio, 0.0, 1.0)
            .unwrap()
            .iter()
            .all(|f| f.frequency_hz.is_none()));
        for (start, end) in [(f64::NAN, 0.4), (0.5, 0.4), (-0.1, 0.4), (0.0, 1.1)] {
            assert!(pitch_track(&audio, start, end).is_err());
        }
        audio.channels = 2;
        assert!(pitch_track(&audio, 0.0, 0.5).is_err());
        audio.channels = 1;
        audio.samples[1] = f32::NAN;
        assert!(pitch_track(&audio, 0.0, 0.5).is_err());
    }
    #[test]
    fn oversized_request_is_rejected_before_allocating_result() {
        let audio = AudioBuffer {
            samples: vec![0.0; WINDOW + HOP * MAX_FRAMES],
            sample_rate: 16000,
            channels: 1,
        };
        assert!(pitch_track(&audio, 0.0, audio.duration_secs()).is_err());
    }
}
