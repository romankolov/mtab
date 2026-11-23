//! Experimental dead-note and palm-mute *suggestions* from mono note attacks.
//!
//! A dead note (a muted, unpitched "chuck") has an attack with energy but no
//! stable harmonic structure and dies away quickly. A palm mute keeps the pitch
//! but darkens the spectrum and shortens the decay. This module measures two
//! simple, gain-invariant descriptors over one short window at each supplied
//! attack — attack-to-tail energy loss and high-band energy share — and reports
//! a tentative class per note.
//!
//! It is a fixed-threshold heuristic, not a trained classifier, and it is not
//! calibrated: on mixes, drums, bass and other instruments contribute to the
//! same windows, overlapping notes confuse the tail measurement, and an
//! alternate picking style, capo or tone change can move both descriptors.
//! Treat output as review suggestions. Nothing here mutates transcription
//! output or claims confidence.

use crate::{AudioBuffer, MODEL_SAMPLE_RATE};
use mtab_core::{MtabError, NoteEvent, Result};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

/// Attack-analysis window: 32 ms at 16 kHz (covers a pick transient).
pub const ATTACK_WINDOW: usize = 512;
/// Tail-analysis window: 32 ms, starting [`TAIL_DELAY`] after the attack.
pub const TAIL_WINDOW: usize = 512;
/// Silence inserted between the attack and tail windows.
pub const TAIL_DELAY: usize = 256;
/// Largest spectral centroid, in Hz, treated as still pitch-like.
const DEAD_MAX_CENTROID_HZ: f64 = 900.0;
/// Smallest attack-to-tail loss (dB) treated as a dead note.
const DEAD_MIN_DECAY_DB: f64 = 18.0;
/// Palm-mute decay range (dB) between a dead note and an open note.
const PALM_MUTE_MIN_DECAY_DB: f64 = 6.0;
/// A palm mute must still be pitch-like: its centroid stays below this.
const PALM_MUTE_MAX_CENTROID_HZ: f64 = 1200.0;

/// The tentative articulation of one supplied note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Articulation {
    /// Attack has energy, dies quickly, and has no stable pitch: likely muted.
    DeadNote,
    /// Pitch survives but the attack is dark and short: likely palm muted.
    PalmMute,
    /// Neither pattern was clear; open (or unclassifiable) note.
    None,
    /// Note too short to cover both windows, or partly outside the audio.
    TooShort,
    /// Attack window has no measurable energy above the silence floor.
    Silent,
}

/// Gain-invariant measurements behind one [`Articulation`]; mostly diagnostic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArticulationEvidence {
    /// Attack window RMS, linear.
    pub attack_rms: f64,
    /// Tail window RMS, linear.
    pub tail_rms: f64,
    /// `20 * log10(attack_rms / tail_rms)`; positive means the sound decayed.
    /// Infinite when the tail is digital silence.
    pub decay_db: f64,
    /// Energy-weighted mean frequency of the attack window, in Hz.
    pub centroid_hz: f64,
}

/// One assessment per input note, in input order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArticulationAssessment {
    /// Tentative class.
    pub articulation: Articulation,
    /// Measurements; zeroed when no window was analyzed.
    pub evidence: ArticulationEvidence,
}

fn invalid() -> MtabError {
    MtabError::UnsupportedAudio(
        "articulation analysis requires finite mono audio at 16000 Hz and ordered notes".into(),
    )
}

struct Window {
    hann: Vec<f64>,
    bin_hz: Vec<f64>,
}

impl Window {
    fn new() -> Self {
        Self {
            hann: (0..ATTACK_WINDOW)
                .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / ATTACK_WINDOW as f64).cos())
                .collect(),
            bin_hz: (0..ATTACK_WINDOW / 2 + 1)
                .map(|k| k as f64 * f64::from(MODEL_SAMPLE_RATE) / ATTACK_WINDOW as f64)
                .collect(),
        }
    }
}

struct Analyzer {
    window: Window,
    fft: Arc<dyn Fft<f64>>,
    scratch: Vec<Complex<f64>>,
    buf: Vec<Complex<f64>>,
    power: Vec<f64>,
}

impl Analyzer {
    fn new() -> Self {
        let window = Window::new();
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(ATTACK_WINDOW);
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        Self {
            window,
            fft,
            scratch,
            buf: vec![Complex::new(0.0, 0.0); ATTACK_WINDOW],
            power: vec![0.0; ATTACK_WINDOW / 2 + 1],
        }
    }

    /// Fill `power[k]` with the Hann-windowed power spectrum of one frame.
    fn spectrum(&mut self, frame: &[f32]) -> &[f64] {
        for (i, value) in self.buf.iter_mut().enumerate() {
            *value = Complex::new(f64::from(frame[i]) * self.window.hann[i], 0.0);
        }
        self.fft
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        for (power, value) in self.power.iter_mut().zip(&self.buf) {
            *power = value.norm_sqr();
        }
        &self.power
    }

    /// Energy-weighted mean frequency of a power spectrum.
    fn centroid(&self) -> f64 {
        let total: f64 = self.power.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        self.power
            .iter()
            .zip(&self.window.bin_hz)
            .map(|(p, hz)| p * hz)
            .sum::<f64>()
            / total
    }
}

fn rms(samples: &[f32]) -> f64 {
    (samples
        .iter()
        .map(|&s| f64::from(s) * f64::from(s))
        .sum::<f64>()
        / samples.len().max(1) as f64)
        .sqrt()
}

/// Suggest dead-note / palm-mute markings for monophonic-guitar audio.
///
/// Input must be finite mono 16 kHz audio. Each note needs an onset inside the
/// audio plus `ATTACK_WINDOW + TAIL_DELAY + TAIL_WINDOW` samples after it; notes
/// that do not fit are reported as [`Articulation::TooShort`] rather than
/// truncated. Windows are taken from the supplied note start, not from a
/// detected onset, so transcription timing error shows up as window smear.
/// Returns one assessment per input note and never changes the notes.
pub fn suggest_articulations(
    audio: &AudioBuffer,
    notes: &[NoteEvent],
) -> Result<Vec<ArticulationAssessment>> {
    if audio.channels != 1
        || audio.sample_rate != MODEL_SAMPLE_RATE
        || audio.samples.iter().any(|s| !s.is_finite())
    {
        return Err(MtabError::UnsupportedAudio(
            "articulation analysis requires finite mono audio at 16000 Hz".into(),
        ));
    }
    if notes
        .iter()
        .any(|n| !n.start.is_finite() || n.start < 0.0 || n.end < n.start || n.midi > 127)
    {
        return Err(invalid());
    }
    let span = ATTACK_WINDOW + TAIL_DELAY + TAIL_WINDOW;
    let mut analyzer = Analyzer::new();
    let mut out = Vec::with_capacity(notes.len());
    let empty = ArticulationEvidence {
        attack_rms: 0.0,
        tail_rms: 0.0,
        decay_db: 0.0,
        centroid_hz: 0.0,
    };
    for note in notes {
        let first = (note.start * f64::from(MODEL_SAMPLE_RATE)).round() as usize;
        if first.saturating_add(span) > audio.samples.len() {
            out.push(ArticulationAssessment {
                articulation: Articulation::TooShort,
                evidence: empty,
            });
            continue;
        }
        let attack = &audio.samples[first..first + ATTACK_WINDOW];
        let tail_start = first + ATTACK_WINDOW + TAIL_DELAY;
        let tail = &audio.samples[tail_start..tail_start + TAIL_WINDOW];
        let attack_rms = rms(attack);
        let tail_rms = rms(tail);
        // Below this the attack is effectively silence and no ratio is meaningful.
        if attack_rms < 1e-4 {
            out.push(ArticulationAssessment {
                articulation: Articulation::Silent,
                evidence: ArticulationEvidence {
                    attack_rms,
                    tail_rms,
                    decay_db: 0.0,
                    centroid_hz: 0.0,
                },
            });
            continue;
        }
        let centroid_hz = {
            analyzer.spectrum(attack);
            analyzer.centroid()
        };
        let decay_db = 20.0 * (attack_rms / tail_rms.max(1e-12)).log10();
        let articulation = if decay_db >= DEAD_MIN_DECAY_DB && centroid_hz < DEAD_MAX_CENTROID_HZ {
            Articulation::DeadNote
        } else if decay_db >= PALM_MUTE_MIN_DECAY_DB && centroid_hz < PALM_MUTE_MAX_CENTROID_HZ {
            Articulation::PalmMute
        } else {
            Articulation::None
        };
        out.push(ArticulationAssessment {
            articulation,
            evidence: ArticulationEvidence {
                attack_rms,
                tail_rms,
                decay_db,
                centroid_hz,
            },
        });
    }
    Ok(out)
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
    fn note(start: f64, end: f64, midi: u8) -> NoteEvent {
        NoteEvent {
            start,
            end,
            midi,
            velocity: 80,
            instrument: 24,
        }
    }
    /// A decaying harmonic tone; `partials` sets how bright it starts.
    fn tone(hz: f64, seconds: f64, decay: f64, partials: usize) -> Vec<f32> {
        let n = (seconds * 16_000.0) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                let envelope = (-decay * t).exp();
                let mut sum = 0.0;
                for p in 1..=partials {
                    sum += (2.0 * PI * hz * p as f64 * t).sin() / p as f64;
                }
                (sum * envelope * 0.3) as f32
            })
            .collect()
    }

    #[test]
    fn rejects_invalid_audio_and_notes() {
        let stereo = AudioBuffer {
            samples: vec![0.0; 8000],
            sample_rate: 16_000,
            channels: 2,
        };
        assert!(suggest_articulations(&stereo, &[]).is_err());
        let a = audio(vec![0.0; 8000]);
        assert!(suggest_articulations(&a, &[note(f64::NAN, 1.0, 40)]).is_err());
        assert!(suggest_articulations(&a, &[note(0.0, 1.0, 200)]).is_err());
    }

    #[test]
    fn distinguishes_sustained_open_note_from_fast_decay() {
        // A long, bright, slowly decaying tone is an open note; a fast-decaying
        // dark one is a dead note; a moderate dark decay is a palm mute.
        let mut samples = tone(196.0, 0.5, 1.0, 10);
        let open = suggest_articulations(&audio(samples.clone()), &[note(0.0, 0.4, 55)])
            .unwrap()
            .remove(0);
        assert_eq!(open.articulation, Articulation::None, "{open:?}");
        samples = tone(196.0, 0.5, 60.0, 3);
        let dead = suggest_articulations(&audio(samples), &[note(0.0, 0.4, 55)])
            .unwrap()
            .remove(0);
        assert_eq!(dead.articulation, Articulation::DeadNote, "{dead:?}");
        assert!(dead.evidence.decay_db >= DEAD_MIN_DECAY_DB);
    }

    #[test]
    fn reports_short_silent_and_out_of_range_notes() {
        let a = audio(tone(196.0, 0.05, 1.0, 6));
        let out = suggest_articulations(&a, &[note(0.04, 0.05, 55), note(5.0, 6.0, 55)]).unwrap();
        assert_eq!(out[0].articulation, Articulation::TooShort);
        assert_eq!(out[1].articulation, Articulation::TooShort);
        let silence = audio(vec![0.0; 8000]);
        let quiet = suggest_articulations(&silence, &[note(0.0, 0.4, 55)]).unwrap();
        assert_eq!(quiet[0].articulation, Articulation::Silent);
    }
}
