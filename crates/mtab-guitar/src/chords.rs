//! Experimental exact-pitch-class chord suggestions for review.
//!
//! This symbolic heuristic sees supplied note events, not audio. It assumes
//! the input contains only the target instrument, ignores octave/voicing, and
//! abstains when the sounding pitch-class set is not exactly a supported chord
//! or matches more than one candidate. It cannot identify incomplete chords,
//! embellishments, non-chord tones, inversions or harmonic context. No claim of
//! real-audio chord accuracy is made; suggestions are not wired to the CLI or
//! notation exporters.

use mtab_core::{MtabError, NoteEvent, Result};

use crate::ChordSymbol;

const TEMPLATES: [(&str, &[u8]); 6] = [
    ("", &[0, 4, 7]),
    ("m", &[0, 3, 7]),
    ("dim", &[0, 3, 6]),
    ("7", &[0, 4, 7, 10]),
    ("maj7", &[0, 4, 7, 11]),
    ("m7", &[0, 3, 7, 10]),
];
const ROOTS: [&str; 12] = [
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
];
const MAX_NOTES: usize = 50_000;

fn label(mask: u16) -> Option<String> {
    let mut found = None;
    for (root, name) in ROOTS.iter().enumerate() {
        for (suffix, intervals) in TEMPLATES {
            let template = intervals.iter().fold(0u16, |bits, &interval| {
                bits | 1 << ((root + usize::from(interval)) % 12)
            });
            if mask == template {
                if found.is_some() {
                    return None;
                }
                found = Some(format!("{name}{suffix}"));
            }
        }
    }
    found
}

/// Suggest exact triad/seventh chord labels for sustained symbolic note sets.
///
/// `minimum_duration` is the shortest emitted continuous chord interval, in
/// seconds. Adjacent intervals with the same label are merged before applying
/// that threshold. Half-open intervals `[start, end)` determine sounding notes;
/// identical start/end times are processed together. Results omit silent,
/// incomplete, out-of-vocabulary or multiply matched sets rather than guessing.
/// Labels use a fixed sharp/flat spelling, not key-aware notation. No confidence
/// score or guarantee of harmonic correctness is implied.
///
/// Returns an error on invalid times, pitches, or more than 50,000 notes.
pub fn suggest_chords(notes: &[NoteEvent], minimum_duration: f64) -> Result<Vec<ChordSymbol>> {
    if !minimum_duration.is_finite() || minimum_duration <= 0.0 || notes.len() > MAX_NOTES {
        return Err(MtabError::InvalidOutput(
            "chord suggestions require positive finite minimum duration and at most 50000 notes"
                .into(),
        ));
    }
    let mut events = Vec::with_capacity(notes.len() * 2);
    for note in notes {
        if !note.start.is_finite()
            || !note.end.is_finite()
            || note.start < 0.0
            || note.end <= note.start
            || note.midi > 127
        {
            return Err(MtabError::InvalidOutput("chord suggestions require finite, positive-duration note times and MIDI pitches 0–127".into()));
        }
        let pitch_class = usize::from(note.midi % 12);
        events.push((note.start, pitch_class, true));
        events.push((note.end, pitch_class, false));
    }
    events.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut active = [0usize; 12];
    let mut result: Vec<ChordSymbol> = Vec::new();
    let mut index = 0;
    while index < events.len() {
        let time = events[index].0;
        while index < events.len() && events[index].0 == time {
            let (_, pitch, on) = events[index];
            if on {
                active[pitch] += 1;
            } else {
                active[pitch] -= 1;
            }
            index += 1;
        }
        if index == events.len() {
            break;
        }
        let next = events[index].0;
        let mask = active
            .iter()
            .enumerate()
            .fold(0u16, |m, (i, &count)| m | (u16::from(count > 0) << i));
        if let Some(name) = label(mask) {
            if let Some(previous) = result
                .last_mut()
                .filter(|c| c.end == time && c.label == name)
            {
                previous.end = next;
            } else {
                result.push(ChordSymbol {
                    start: time,
                    end: next,
                    label: name,
                });
            }
        }
    }
    result.retain(|chord| chord.end - chord.start >= minimum_duration);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(midi: u8, start: f64, end: f64) -> NoteEvent {
        NoteEvent {
            start,
            end,
            midi,
            velocity: 100,
            instrument: 24,
        }
    }

    #[test]
    fn recognizes_exact_chords_and_merges_voicing_changes() {
        let notes = [
            note(48, 0.0, 1.0),
            note(52, 0.0, 1.0),
            note(55, 0.0, 1.0),
            note(60, 0.4, 0.6),
            note(57, 1.0, 2.0),
            note(60, 1.0, 2.0),
            note(64, 1.0, 2.0),
        ];
        assert_eq!(
            suggest_chords(&notes, 0.1).unwrap(),
            vec![
                ChordSymbol {
                    start: 0.0,
                    end: 1.0,
                    label: "C".into()
                },
                ChordSymbol {
                    start: 1.0,
                    end: 2.0,
                    label: "Am".into()
                },
            ]
        );
    }

    #[test]
    fn abstains_on_incomplete_or_extra_pitch_classes() {
        let notes = [
            note(60, 0.0, 1.0),
            note(64, 0.0, 1.0),
            note(67, 0.0, 0.3),
            note(62, 0.1, 0.2),
        ];
        assert_eq!(
            suggest_chords(&notes, 0.11).unwrap(),
            Vec::<ChordSymbol>::new()
        );
        let result = suggest_chords(&notes, 0.05).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(
            (
                result[0].start,
                result[0].end,
                result[1].start,
                result[1].end
            ),
            (0.0, 0.1, 0.2, 0.3)
        );
    }

    #[test]
    fn sweep_matches_independent_interval_reconstruction() {
        let mut rng = 0x8a21_57d3_0123_4567u64;
        for _ in 0..500 {
            let mut notes = Vec::new();
            for _ in 0..24 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let start = ((rng >> 32) % 20) as f64;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let end = start + 1.0 + ((rng >> 32) % 5) as f64;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                notes.push(note(48 + ((rng >> 32) % 24) as u8, start, end));
            }
            let mut boundaries: Vec<_> = notes.iter().flat_map(|n| [n.start, n.end]).collect();
            boundaries.sort_by(f64::total_cmp);
            boundaries.dedup();
            let mut expected: Vec<ChordSymbol> = Vec::new();
            for window in boundaries.windows(2) {
                let mid = (window[0] + window[1]) / 2.0;
                let mask = notes
                    .iter()
                    .filter(|n| n.start <= mid && mid < n.end)
                    .fold(0u16, |m, n| m | 1 << (n.midi % 12));
                if let Some(name) = label(mask) {
                    if let Some(prev) = expected
                        .last_mut()
                        .filter(|c| c.end == window[0] && c.label == name)
                    {
                        prev.end = window[1];
                    } else {
                        expected.push(ChordSymbol {
                            start: window[0],
                            end: window[1],
                            label: name,
                        });
                    }
                }
            }
            assert_eq!(suggest_chords(&notes, 0.1).unwrap(), expected);
        }
    }

    #[test]
    fn supports_seventh_chords_and_rejects_bad_metadata() {
        let c7 = [
            note(60, 0.0, 1.0),
            note(64, 0.0, 1.0),
            note(67, 0.0, 1.0),
            note(70, 0.0, 1.0),
        ];
        assert_eq!(suggest_chords(&c7, 0.1).unwrap()[0].label, "C7");
        assert!(suggest_chords(&c7, 0.0).is_err());
        let mut invalid = c7;
        invalid[1].end = f64::NAN;
        assert!(suggest_chords(&invalid, 0.1).is_err());
        invalid[1] = note(128, 0.0, 1.0);
        assert!(suggest_chords(&invalid, 0.1).is_err());
    }
}
