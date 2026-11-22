//! Preliminary note velocity estimation from a mono audio envelope.
//!
//! The first 50 ms of each note is measured relative to a two-second local
//! RMS window. This is a simple mix-level heuristic: it does not separate
//! instruments or isolate pitches, so overlapping instruments/notes can
//! bias estimates. Silence or notes outside the audio keep their input velocity.
//! Upstream-compatible model/token decoding is not modified.

use mtab_core::{MtabError, NoteEvent, Result};

use crate::AudioBuffer;

/// Number of notes assigned an envelope estimate and left unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicsReport {
    /// Notes whose velocity was updated using the audio envelope.
    pub estimated: usize,
    /// Notes with insufficient audio energy or coverage; original velocity kept.
    pub skipped: usize,
}

/// Estimate note velocities in the range 30–127 from a mono audio signal.
///
/// Each note's first 50 ms RMS is compared to the RMS of up to one second on
/// each side of its onset. A 1 dB increase in the ratio adds four MIDI units
/// around an arbitrary reference velocity of 90. This relative mapping is not
/// calibrated to a MIDI performance and measures mixture energy, not isolated
/// per-note amplitude; concurrent sounds and noisy input can bias it. Scaling
/// the whole input by a common gain leaves the result approximately unchanged.
/// Notes with no covered non-silent attack retain their incoming velocity.
/// Returns an error before mutating notes for invalid audio or note metadata.
pub fn estimate_velocities(audio: &AudioBuffer, notes: &mut [NoteEvent]) -> Result<DynamicsReport> {
    if audio.channels != 1 || audio.sample_rate == 0 || audio.samples.iter().any(|s| !s.is_finite())
    {
        return Err(MtabError::UnsupportedAudio(
            "dynamics estimation requires finite mono audio with a nonzero sample rate".into(),
        ));
    }
    if notes.iter().any(|n| {
        !n.start.is_finite()
            || !n.end.is_finite()
            || n.start < 0.0
            || n.end <= n.start
            || !(1..=127).contains(&n.velocity)
    }) {
        return Err(MtabError::UnsupportedAudio("dynamics estimation requires notes with nonnegative ordered finite times and velocities 1–127".into()));
    }
    // Blockwise squared-energy prefix sums bound overhead to roughly one f64
    // per 10 ms instead of one per sample, while edge samples are exact.
    let block = (audio.sample_rate as usize / 100).max(1);
    let mut prefix = vec![0.0f64];
    for chunk in audio.samples.chunks(block) {
        let energy: f64 = chunk.iter().map(|&x| f64::from(x).powi(2)).sum();
        prefix.push(prefix.last().copied().unwrap() + energy);
    }
    let rms = |start: usize, end: usize| -> f64 {
        if start >= end {
            return 0.0;
        }
        let first_full = start.saturating_add(block - 1) / block;
        let last_full = end / block;
        let energy = if first_full >= last_full {
            audio.samples[start..end]
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum()
        } else {
            let left = first_full * block;
            let right = last_full * block;
            let edge: f64 = audio.samples[start..left]
                .iter()
                .chain(&audio.samples[right..end])
                .map(|&v| f64::from(v).powi(2))
                .sum();
            edge + prefix[last_full] - prefix[first_full]
        };
        (energy.max(0.0) / (end - start) as f64).sqrt()
    };
    let rate = audio.sample_rate as f64;
    let len = audio.samples.len();
    let mut report = DynamicsReport {
        estimated: 0,
        skipped: 0,
    };
    for note in notes {
        let start = (note.start * rate).floor() as usize;
        let end = (note.end.min(note.start + 0.050) * rate).ceil() as usize;
        if start >= len || end.min(len) <= start {
            report.skipped += 1;
            continue;
        }
        let local_start = start.saturating_sub(audio.sample_rate as usize);
        let local_end = start.saturating_add(audio.sample_rate as usize).min(len);
        let attack = rms(start, end.min(len));
        let local = rms(local_start, local_end);
        if attack < 1e-8 || local < 1e-8 {
            report.skipped += 1;
            continue;
        }
        let db = 20.0 * (attack / local).log10();
        note.velocity = (90.0 + 4.0 * db).round().clamp(30.0, 127.0) as u8;
        report.estimated += 1;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(start: f64) -> NoteEvent {
        NoteEvent {
            start,
            end: start + 0.2,
            midi: 64,
            velocity: 100,
            instrument: 24,
        }
    }

    #[test]
    fn louder_attacks_have_higher_velocity_and_gain_invariance() {
        let mut samples = vec![0.0f32; 4000];
        for v in &mut samples[500..550] {
            *v = 0.1;
        }
        for v in &mut samples[1000..1050] {
            *v = 0.4;
        }
        let mut audio = AudioBuffer {
            samples,
            sample_rate: 1000,
            channels: 1,
        };
        let mut notes = [note(0.5), note(1.0)];
        let report = estimate_velocities(&audio, &mut notes).unwrap();
        assert_eq!(
            report,
            DynamicsReport {
                estimated: 2,
                skipped: 0
            }
        );
        assert!(notes[1].velocity > notes[0].velocity);
        assert!(notes.iter().all(|n| (30..=127).contains(&n.velocity)));
        let result = notes;
        for v in &mut audio.samples {
            *v *= 0.25;
        }
        let mut scaled = [note(0.5), note(1.0)];
        estimate_velocities(&audio, &mut scaled).unwrap();
        assert_eq!(result, scaled);
    }

    #[test]
    fn silence_and_out_of_range_preserve_original() {
        let audio = AudioBuffer {
            samples: vec![0.0; 1000],
            sample_rate: 1000,
            channels: 1,
        };
        let mut notes = [note(0.2), note(2.0)];
        assert_eq!(
            estimate_velocities(&audio, &mut notes).unwrap(),
            DynamicsReport {
                estimated: 0,
                skipped: 2
            }
        );
        assert_eq!([notes[0].velocity, notes[1].velocity], [100, 100]);
    }

    #[test]
    fn invalid_input_does_not_partially_mutate() {
        let audio = AudioBuffer {
            samples: vec![0.5; 1000],
            sample_rate: 1000,
            channels: 1,
        };
        let mut notes = [note(0.2), note(0.3)];
        notes[1].end = f64::NAN;
        assert!(estimate_velocities(&audio, &mut notes).is_err());
        assert_eq!(notes[0].velocity, 100);
        let mut valid = [note(0.2)];
        let mut invalid = audio;
        invalid.samples[0] = f32::NAN;
        assert!(estimate_velocities(&invalid, &mut valid).is_err());
        assert_eq!(valid[0].velocity, 100);
    }
    #[test]
    fn constant_amplitude_is_reference_velocity_even_at_edges() {
        let audio = AudioBuffer {
            samples: vec![0.3; 127],
            sample_rate: 1033,
            channels: 1,
        };
        let mut notes = [note(0.0), note(0.009), note(0.05), note(0.12)];
        let report = estimate_velocities(&audio, &mut notes).unwrap();
        assert_eq!(
            report,
            DynamicsReport {
                estimated: 4,
                skipped: 0
            }
        );
        assert_eq!(notes.map(|n| n.velocity), [90; 4]);
    }
}
