//! Bounded global string/fret search for a single guitar track.
//!
//! The search retains several alternative histories instead of choosing each
//! note independently. It enforces same-string non-overlap and a maximum span
//! between simultaneously sounding fretted notes, and scores position shifts,
//! string jumps, high frets and open-string preference. Beam pruning makes this
//! an approximation: it cannot guarantee the global optimum or even preserve a
//! feasible solution when the configured beam is too small. Chord dictionaries,
//! advanced playability and full end-to-end GuitarSet evaluation are future work.

use std::collections::HashSet;

use mtab_core::{MtabError, NoteEvent, Result};

use crate::{Capo, TabNote, Tuning};

/// Search constraints and beam width.
#[derive(Debug, Clone, Copy)]
pub struct FingeringOptions {
    /// Physical fretboard length; capo plus relative fret cannot exceed this.
    pub max_fret: u8,
    /// Maximum simultaneous fretted-note span; open strings do not count.
    pub max_span: u8,
    /// Number of alternative partial solutions retained after each onset.
    pub beam_width: usize,
}

impl Default for FingeringOptions {
    fn default() -> Self {
        Self {
            max_fret: 22,
            max_span: 4,
            beam_width: 512,
        }
    }
}

#[derive(Clone)]
struct Active {
    string: u8,
    fret: u8,
    end: f64,
}

struct PathNode {
    previous: Option<usize>,
    index: usize,
    string: u8,
    fret: u8,
}

#[derive(Clone)]
struct State {
    active: Vec<Active>,
    position: Option<u8>,
    last_string: Option<u8>,
    score: f64,
    path: Option<usize>,
}

/// Assign raw notes from a **single instrument track** to a tuning and capo.
/// Returns notes in the same order as input; their velocities and timings are
/// preserved, not estimated. All pitches, string collisions, and simultaneous
/// fret spans are checked. Unplayable input returns an error rather than
/// silently dropping notes. No chord-shape dictionary or measured GuitarSet
/// accuracy is implied by this preliminary solver.
pub fn assign_notes(
    notes: &[NoteEvent],
    tuning: &Tuning,
    capo: Capo,
    options: FingeringOptions,
) -> Result<Vec<TabNote>> {
    if !(Tuning::MIN_STRINGS..=Tuning::MAX_STRINGS).contains(&tuning.strings.len())
        || !tuning.strings.windows(2).all(|w| w[0] < w[1])
        || tuning.strings.iter().any(|&p| p > 127)
    {
        return Err(MtabError::InvalidTuning(
            "tuning must contain 6–8 strictly ascending MIDI pitches".into(),
        ));
    }
    if options.max_fret == 0
        || options.max_fret > 36
        || capo.position() > options.max_fret
        || options.max_span == 0
        || options.beam_width == 0
        || options.beam_width > 4096
    {
        return Err(MtabError::InvalidTuning("fingering requires frets 1–36, capo within the fretboard, positive span and beam width 1–4096".into()));
    }
    for (index, note) in notes.iter().enumerate() {
        if !note.start.is_finite()
            || !note.end.is_finite()
            || note.start < 0.0
            || note.end <= note.start
            || note.midi > 127
            || !(1..=127).contains(&note.velocity)
            || note.instrument >= 128
            || notes
                .first()
                .is_some_and(|first| note.instrument != first.instrument)
        {
            return Err(MtabError::UnplayableNote {
                index,
                reason: "notes need finite ordered times, valid MIDI pitch/velocity and one non-drum instrument"
                    .into(),
            });
        }
    }
    let mut order: Vec<usize> = (0..notes.len()).collect();
    order.sort_by(|&a, &b| notes[a].start.total_cmp(&notes[b].start).then(a.cmp(&b)));
    let mut states = vec![State {
        active: Vec::new(),
        position: None,
        last_string: None,
        score: 0.0,
        path: None,
    }];
    // An index arena avoids recursive reference-counted path destruction on
    // long songs; only surviving beam paths receive nodes.
    let mut paths: Vec<PathNode> = Vec::new();
    for &index in &order {
        let note = &notes[index];
        let candidates: Vec<(u8, u8)> = tuning
            .strings
            .iter()
            .enumerate()
            .filter_map(|(string, &open)| {
                let fret = note.midi as i16 - open as i16 - capo.position() as i16;
                (fret >= 0 && fret <= (options.max_fret - capo.position()) as i16)
                    .then_some((string as u8, fret as u8))
            })
            .collect();
        if candidates.is_empty() {
            return Err(MtabError::UnplayableNote {
                index,
                reason: format!("pitch {} is outside this tuning/capo/fret range", note.midi),
            });
        }
        let mut next = Vec::new();
        for state in &states {
            let active: Vec<Active> = state
                .active
                .iter()
                .filter(|a| a.end > note.start)
                .cloned()
                .collect();
            for &(string, fret) in &candidates {
                if active.iter().any(|a| a.string == string) {
                    continue;
                }
                let mut fretted = active.iter().filter_map(|a| (a.fret > 0).then_some(a.fret));
                let (mut low, mut high) = (fret, fret);
                if fret == 0 {
                    if let Some(first) = fretted.next() {
                        low = first;
                        high = first;
                    }
                }
                for f in fretted {
                    low = low.min(f);
                    high = high.max(f);
                }
                if low > 0 && high - low > options.max_span {
                    continue;
                }
                let mut score = state.score;
                if fret == 0 {
                    score -= 0.8;
                } else {
                    score += fret as f64 * 0.015;
                    if let Some(pos) = state.position {
                        score += f64::from(fret.abs_diff(pos));
                    }
                }
                if let Some(last) = state.last_string {
                    score += 0.15 * f64::from(string.abs_diff(last));
                }
                let mut active_next = active.clone();
                active_next.push(Active {
                    string,
                    fret,
                    end: note.end,
                });
                next.push((
                    State {
                        active: active_next,
                        position: if fret == 0 {
                            state.position
                        } else {
                            Some(fret)
                        },
                        last_string: Some(string),
                        score,
                        path: state.path,
                    },
                    string,
                    fret,
                ));
            }
        }
        if next.is_empty() {
            return Err(MtabError::UnplayableNote { index, reason: "no feasible string remains under overlap/span constraints (or the beam pruned the feasible path)".into() });
        }
        // Stable sort preserves deterministic generation order for equal costs.
        // Merge equivalent future-relevant states, keeping their cheapest path,
        // so redundant histories do not crowd useful alternatives out of the beam.
        next.sort_by(|a, b| a.0.score.total_cmp(&b.0.score));
        let mut seen = HashSet::new();
        next.retain(|(state, _, _)| {
            let mut active: Vec<(u8, u8, u64)> = state
                .active
                .iter()
                .map(|a| (a.string, a.fret, a.end.to_bits()))
                .collect();
            active.sort_unstable();
            seen.insert((state.position, state.last_string, active))
        });
        next.truncate(options.beam_width);
        states = next
            .into_iter()
            .map(|(mut state, string, fret)| {
                paths.push(PathNode {
                    previous: state.path,
                    index,
                    string,
                    fret,
                });
                state.path = Some(paths.len() - 1);
                state
            })
            .collect();
    }
    let mut out: Vec<TabNote> = notes
        .iter()
        .map(|n| TabNote {
            start: n.start,
            end: n.end,
            midi: n.midi,
            string: 0,
            fret: 0,
            velocity: n.velocity,
            bend: None,
            technique: Vec::new(),
            tie_from_prev: false,
        })
        .collect();
    let mut path = states[0].path;
    while let Some(index) = path {
        let node = &paths[index];
        out[node.index].string = node.string;
        out[node.index].fret = node.fret;
        path = node.previous;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn note(midi: u8, start: f64, end: f64) -> NoteEvent {
        NoteEvent {
            start,
            end,
            midi,
            velocity: 91,
            instrument: 24,
        }
    }

    #[test]
    fn chooses_playable_chord_and_preserves_input_order() {
        let notes = [
            note(64, 0.0, 1.0),
            note(40, 0.0, 1.0),
            note(55, 0.0, 1.0),
            note(59, 0.0, 1.0),
        ];
        let assigned = assign_notes(
            &notes,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default(),
        )
        .unwrap();
        assert_eq!(
            assigned
                .iter()
                .map(|n| (n.string, n.fret))
                .collect::<Vec<_>>(),
            vec![(5, 0), (0, 0), (3, 0), (4, 0)]
        );
        for (before, after) in notes.iter().zip(&assigned) {
            assert_eq!(
                (before.midi, before.start, before.end, before.velocity),
                (after.midi, after.start, after.end, after.velocity)
            );
        }
    }

    #[test]
    fn beam_keeps_alternative_needed_for_future_overlap() {
        // E4 can use open top E or B string fret 5. A4 at fret 21 can only
        // use the top E within the 22-fret board. Greedy chooses open E first.
        let notes = [note(64, 0.0, 2.0), note(85, 1.0, 2.0)];
        let opts = FingeringOptions {
            max_span: 22,
            beam_width: 1,
            ..FingeringOptions::default()
        };
        assert!(assign_notes(&notes, &Tuning::standard(), Capo::NONE, opts).is_err());
        let result = assign_notes(
            &notes,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions {
                beam_width: 64,
                ..opts
            },
        )
        .unwrap();
        assert_ne!(result[0].string, result[1].string);
        assert_eq!((result[1].string, result[1].fret), (5, 21));
    }

    #[test]
    fn capo_and_extended_tuning_affect_candidates() {
        let capo = Capo::new(2).unwrap();
        let a = assign_notes(
            &[note(42, 0.0, 1.0)],
            &Tuning::standard(),
            capo,
            FingeringOptions::default(),
        )
        .unwrap();
        assert_eq!((a[0].string, a[0].fret), (0, 0));
        assert!(assign_notes(
            &[note(40, 0.0, 1.0)],
            &Tuning::standard(),
            capo,
            FingeringOptions::default()
        )
        .is_err());
        let b = assign_notes(
            &[note(35, 0.0, 1.0)],
            &Tuning::seven_string(),
            Capo::NONE,
            FingeringOptions::default(),
        )
        .unwrap();
        assert_eq!((b[0].string, b[0].fret), (0, 0));
    }

    #[test]
    fn handles_empty_invalid_and_string_collisions() {
        assert!(assign_notes(
            &[],
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default()
        )
        .unwrap()
        .is_empty());
        assert!(assign_notes(
            &[note(40, 0.0, 1.0), note(40, 0.2, 1.0)],
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default()
        )
        .is_err());
        assert!(assign_notes(
            &[note(40, f64::NAN, 1.0)],
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default()
        )
        .is_err());
        assert!(assign_notes(
            &[note(40, 0.0, 1.0)],
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions {
                beam_width: 0,
                ..FingeringOptions::default()
            }
        )
        .is_err());
    }

    #[test]
    fn independently_assigned_tracks_can_share_strings() {
        let notes = [note(40, 0.0, 1.0)];
        let a = assign_notes(
            &notes,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default(),
        )
        .unwrap();
        let b = assign_notes(
            &notes,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default(),
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a[0].string, 0);
    }
    #[test]
    fn rejects_unplayable_fret_span_and_accepts_boundary_release() {
        let impossible = [note(41, 0.0, 2.0), note(85, 1.0, 2.0)];
        assert!(assign_notes(
            &impossible,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default()
        )
        .is_err());
        let sequential = [note(41, 0.0, 1.0), note(85, 1.0, 2.0)];
        let a = assign_notes(
            &sequential,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default(),
        )
        .unwrap();
        let b = assign_notes(
            &sequential,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default(),
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_drums_and_mixed_instrument_tracks() {
        let a = note(40, 0.0, 1.0);
        let mut b = note(45, 1.0, 2.0);
        b.instrument = 26;
        assert!(assign_notes(
            &[a, b],
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default()
        )
        .is_err());
        b.instrument = 128;
        assert!(assign_notes(
            &[b],
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default()
        )
        .is_err());
    }
}
