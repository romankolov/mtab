//! Conservative experimental bend *suggestions* from isolated note spans.
//!
//! This does not mutate transcription output, and it is not reliable on mixes:
//! undetected overlapping notes, vibrato, alternate tuning and octave errors
//! can all be mistaken for bends. Suggestions require review. The CLI does not
//! currently enable this analysis.
use crate::{pitch_track::PitchTracker, AudioBuffer};
use mtab_core::{MtabError, NoteEvent, Result};

/// The simple contour shape supported by a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BendShape {
    Bend,
    Release,
    BendRelease,
}
/// A tentative positive pitch deviation relative to equal-tempered MIDI pitch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BendCandidate {
    /// Contour shape.
    pub shape: BendShape,
    /// Approximate absolute time of reaching the peak (or starting a release).
    pub peak_time: f64,
    /// Approximate maximum deviation, in cents; not rounded to GP5 values.
    pub depth_cents: f64,
}
/// Why a note was analyzed or skipped. A result is not a calibrated confidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BendAssessment {
    /// A plausible simple contour; verify against isolated audio.
    Candidate(BendCandidate),
    /// Note shorter than the minimum useful analysis span.
    TooShort,
    /// Another supplied note overlaps this note in time, even across instruments.
    OverlappingNotes,
    /// Note end lies beyond the available audio.
    OutsideAudio,
    /// Most frames were not voiced.
    InsufficientVoicing,
    /// Estimated pitch is too far from the supplied note or bends exceed 200 cents.
    OutOfRange,
    /// No sufficiently clear simple contour, including constant pitch or vibrato.
    NoClearBend,
}

/// Suggest simple bends from non-overlapping notes in mono model-rate audio.
///
/// A note needs at least 0.4 seconds, at least 80% voiced frames, and estimated
/// deviations within -35..=210 cents of its MIDI pitch. It must have a coherent
/// bend/release shape of at least 40 cents. The tracker uses a 128 ms window and thus
/// smears timing and cannot resolve rapid gestures. Overlapping *supplied* notes
/// are skipped, but undetected sounds in the mixture are not removed. Returns
/// one assessment per input note in original order; never changes input notes.
pub fn suggest_bends(audio: &AudioBuffer, notes: &[NoteEvent]) -> Result<Vec<BendAssessment>> {
    if notes.iter().any(|n| {
        !n.start.is_finite()
            || !n.end.is_finite()
            || n.start < 0.0
            || n.end <= n.start
            || n.midi > 127
    }) {
        return Err(MtabError::UnsupportedAudio(
            "bend analysis requires finite ordered note times and MIDI pitch 0–127".into(),
        ));
    }
    let mut tracker = PitchTracker::new(audio)?;
    let mut order: Vec<usize> = (0..notes.len()).collect();
    order.sort_by(|&a, &b| notes[a].start.total_cmp(&notes[b].start).then(a.cmp(&b)));
    let mut overlaps = vec![false; notes.len()];
    let mut prior_end = 0.0_f64;
    for (position, &index) in order.iter().enumerate() {
        let note = &notes[index];
        if prior_end > note.start
            || order
                .get(position + 1)
                .is_some_and(|&next| notes[next].start < note.end)
        {
            overlaps[index] = true;
        }
        prior_end = prior_end.max(note.end);
    }
    let mut out = Vec::with_capacity(notes.len());
    for (i, note) in notes.iter().enumerate() {
        let assessment = if overlaps[i] {
            BendAssessment::OverlappingNotes
        } else if note.end > audio.duration_secs() {
            BendAssessment::OutsideAudio
        } else if note.end - note.start < 0.4 {
            BendAssessment::TooShort
        } else {
            assess(&mut tracker, note)?
        };
        out.push(assessment);
    }
    Ok(out)
}

fn median3(mut x: [f64; 3]) -> f64 {
    x.sort_by(f64::total_cmp);
    x[1]
}
fn assess(tracker: &mut PitchTracker<'_>, note: &NoteEvent) -> Result<BendAssessment> {
    let frames = tracker.track(note.start, note.end)?;
    if frames.len() < 9 {
        return Ok(BendAssessment::TooShort);
    }
    let base = 440.0 * 2.0f64.powf((f64::from(note.midi) - 69.0) / 12.0);
    let voiced: Vec<_> = frames
        .iter()
        .filter_map(|f| {
            f.frequency_hz
                .map(|hz| (f.time, 1200.0 * (hz / base).log2()))
        })
        .collect();
    if voiced.len() * 5 < frames.len() * 4 {
        return Ok(BendAssessment::InsufficientVoicing);
    }
    if voiced
        .iter()
        .any(|(_, cents)| !(-35.0..=210.0).contains(cents))
    {
        return Ok(BendAssessment::OutOfRange);
    }
    // Median-of-three suppresses single-frame octave/noise artifacts without
    // claiming to recover dynamics faster than the tracker window.
    let smooth: Vec<_> = voiced
        .windows(3)
        .map(|w| (w[1].0, median3([w[0].1, w[1].1, w[2].1])))
        .collect();
    if smooth.len() < 7 {
        return Ok(BendAssessment::InsufficientVoicing);
    }
    let first = median3([smooth[0].1, smooth[1].1, smooth[2].1]);
    let n = smooth.len();
    let last = median3([smooth[n - 3].1, smooth[n - 2].1, smooth[n - 1].1]);
    let (peak_index, &(_, peak)) = smooth
        .iter()
        .enumerate()
        .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
        .unwrap();
    if peak < 40.0 || peak > 200.0 {
        return Ok(BendAssessment::NoClearBend);
    }
    let mut previous = 0_i8;
    let mut changes = 0;
    for pair in smooth.windows(2) {
        let difference = pair[1].1 - pair[0].1;
        let sign = if difference > 4.0 {
            1
        } else if difference < -4.0 {
            -1
        } else {
            0
        };
        if sign != 0 {
            if previous != 0 && previous != sign {
                changes += 1;
            }
            previous = sign;
        }
    }
    if changes > 2 {
        return Ok(BendAssessment::NoClearBend);
    }
    // Frame centers lie at least half a window inside the note, so a moving
    // contour may still be offset at the first and last measured frames.
    let near_base = |value: f64| value.abs() <= 30.0;
    let shape = if near_base(first) && last >= 40.0 && peak - last <= 25.0 {
        BendShape::Bend
    } else if first >= 40.0 && near_base(last) {
        BendShape::Release
    } else if near_base(first) && near_base(last) && peak_index >= n / 5 && peak_index < n * 4 / 5 {
        BendShape::BendRelease
    } else {
        return Ok(BendAssessment::NoClearBend);
    };
    let peak_time = match shape {
        BendShape::Bend | BendShape::BendRelease => {
            smooth
                .iter()
                .find(|(_, cents)| *cents >= peak - 10.0)
                .unwrap()
                .0
        }
        BendShape::Release => {
            smooth
                .iter()
                .find(|(_, cents)| *cents <= first - 10.0)
                .unwrap_or(&smooth[peak_index])
                .0
        }
    };
    Ok(BendAssessment::Candidate(BendCandidate {
        shape,
        peak_time,
        depth_cents: peak,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    fn note(start: f64, end: f64) -> NoteEvent {
        NoteEvent {
            start,
            end,
            midi: 45,
            velocity: 100,
            instrument: 24,
        }
    }
    fn audio(contour: impl Fn(f64) -> f64) -> AudioBuffer {
        let mut phase = 0.0;
        let samples = (0..16000)
            .map(|i| {
                let time = i as f64 / 16000.0;
                let hz = 110.0 * 2.0f64.powf(contour(time) / 1200.0);
                phase += 2.0 * PI * hz / 16000.0;
                (0.4 * phase.sin()) as f32
            })
            .collect();
        AudioBuffer {
            samples,
            sample_rate: 16000,
            channels: 1,
        }
    }
    #[test]
    fn recognizes_synthetic_bend_release_and_bend_release() {
        type Contour = Box<dyn Fn(f64) -> f64>;
        let cases: [(BendShape, Contour); 3] = [
            (
                BendShape::Bend,
                Box::new(|t| ((t - 0.2) / 0.35).clamp(0.0, 1.0) * 100.0),
            ),
            (
                BendShape::Release,
                Box::new(|t| (1.0 - ((t - 0.25) / 0.4).clamp(0.0, 1.0)) * 100.0),
            ),
            (
                BendShape::BendRelease,
                Box::new(|t| {
                    (((t - 0.15) / 0.3).clamp(0.0, 1.0)).min(((0.85 - t) / 0.3).clamp(0.0, 1.0))
                        * 100.0
                }),
            ),
        ];
        for (expected, contour) in cases {
            let result = suggest_bends(&audio(contour), &[note(0.0, 1.0)]).unwrap();
            let BendAssessment::Candidate(candidate) = result[0] else {
                panic!("{expected:?}: {result:?}")
            };
            assert_eq!(candidate.shape, expected);
            assert!(
                (candidate.depth_cents - 100.0).abs() < 15.0,
                "{candidate:?}"
            );
            let expected_peak = match expected {
                BendShape::Bend => 0.55,
                BendShape::Release => 0.25,
                BendShape::BendRelease => 0.45,
            };
            assert!(
                (candidate.peak_time - expected_peak).abs() < 0.14,
                "{candidate:?}"
            );
        }
    }
    #[test]
    fn bend_release_ramp_at_note_edges_remains_detectable() {
        let result = suggest_bends(&audio(|t| 200.0 * t.min(1.0 - t)), &[note(0.0, 1.0)]).unwrap();
        let BendAssessment::Candidate(candidate) = result[0] else {
            panic!("expected bend/release, got {result:?}")
        };
        assert_eq!(candidate.shape, BendShape::BendRelease);
        assert!((candidate.depth_cents - 100.0).abs() < 15.0);
    }
    #[test]
    fn abstains_on_constant_pitch_vibrato_noise_and_out_of_range() {
        for contour in [
            Box::new(|_| 0.0) as Box<dyn Fn(f64) -> f64>,
            Box::new(|_| 20.0),
            Box::new(|_| 30.0),
            Box::new(|_| 60.0),
            Box::new(|t| 50.0 + 45.0 * (2.0 * PI * 5.0 * t).sin()),
        ] {
            let result = suggest_bends(&audio(contour), &[note(0.0, 1.0)]).unwrap();
            assert!(
                !matches!(result[0], BendAssessment::Candidate(_)),
                "{result:?}"
            );
        }
        assert_eq!(
            suggest_bends(&audio(|_| 300.0), &[note(0.0, 1.0)]).unwrap(),
            vec![BendAssessment::OutOfRange]
        );
        let mut silence = audio(|_| 0.0);
        silence.samples.fill(0.0);
        assert_eq!(
            suggest_bends(&silence, &[note(0.0, 1.0)]).unwrap(),
            vec![BendAssessment::InsufficientVoicing]
        );
        let mut seed = 7u32;
        for sample in &mut silence.samples {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *sample = (2.0 * seed as f64 / f64::from(u32::MAX) - 1.0) as f32;
        }
        assert_eq!(
            suggest_bends(&silence, &[note(0.0, 1.0)]).unwrap(),
            vec![BendAssessment::InsufficientVoicing]
        );
    }
    #[test]
    fn overlap_detection_handles_nested_unsorted_and_touching_notes() {
        let audio = audio(|_| 0.0);
        let notes = [
            note(0.2, 0.3),
            note(0.0, 0.8),
            note(0.8, 1.0),
            note(0.4, 0.6),
        ];
        let result = suggest_bends(&audio, &notes).unwrap();
        assert_eq!(
            result,
            [
                BendAssessment::OverlappingNotes,
                BendAssessment::OverlappingNotes,
                BendAssessment::TooShort,
                BendAssessment::OverlappingNotes
            ]
        );
        assert_eq!(
            suggest_bends(&audio, &[note(0.6, 1.1)]).unwrap(),
            vec![BendAssessment::OutsideAudio]
        );
        assert!(suggest_bends(&audio, &[note(f64::NAN, 1.0)]).is_err());
    }
}
