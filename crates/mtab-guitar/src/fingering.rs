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

// Collapse sub-nanosecond arithmetic jitter at sequential note boundaries.
// Output note ends are clipped correspondingly, preserving the no-overlap
// invariant rather than treating physically overlapping notes as disjoint.
const BOUNDARY_EPSILON: f64 = 1e-9;

const NO_FEASIBLE: &str = "no feasible string remains under overlap/span constraints (or the beam pruned the feasible path)";

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
    start: f64,
    end: f64,
}

fn sounding_at(active: &Active, onset: f64) -> bool {
    active.end > onset && !(active.start < onset && active.end - onset <= BOUNDARY_EPSILON)
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
    skipped: usize,
    path: Option<usize>,
}

/// Assign raw notes from a **single instrument track** to a tuning and capo.
/// Returns notes in the same order as input; their velocities and timings are
/// preserved, not estimated, except that sub-nanosecond boundary overlaps are
/// clipped to the following onset to remove floating-point jitter. All pitches,
/// string collisions, and simultaneous fret spans are checked. Unplayable input returns an error rather than
/// silently dropping notes. No chord-shape dictionary or measured GuitarSet
/// accuracy is implied by this preliminary solver.
pub fn assign_notes(
    notes: &[NoteEvent],
    tuning: &Tuning,
    capo: Capo,
    options: FingeringOptions,
) -> Result<Vec<TabNote>> {
    let result = assign_impl(notes, tuning, capo, options, false)?;
    Ok(result
        .notes
        .into_iter()
        .map(|note| note.expect("strict search assigns every note"))
        .collect())
}

/// Partial assignment, aligned to the original input order.
#[derive(Debug, Clone, PartialEq)]
pub struct PartialFingering {
    /// `None` marks a note not assigned and therefore absent from tab exports.
    pub notes: Vec<Option<TabNote>>,
    /// Original input indices of all notes not assigned, in ascending order.
    pub unassigned_indices: Vec<usize>,
}

/// Attempt a partial fingering without silently discarding impossible notes.
///
/// Unlike [`assign_notes`], this allows skipping notes to keep the remaining
/// part playable. Paths are ranked lexicographically by the number of skipped
/// notes, then the same ergonomic cost as the strict search. Beam pruning means
/// neither the number of skips nor the cost is globally optimal in general.
/// Every skipped input note is exposed in `unassigned_indices` and as `None`.
/// Out-of-range pitches can be skipped, but malformed note metadata, tuning and
/// options remain errors. Input notes are never mutated. Same-string collision,
/// span, and sub-nanosecond boundary rules are identical to strict search.
/// A partial result is not an accurate transcription of all original notes.
pub fn assign_notes_partial(
    notes: &[NoteEvent],
    tuning: &Tuning,
    capo: Capo,
    options: FingeringOptions,
) -> Result<PartialFingering> {
    assign_impl(notes, tuning, capo, options, true)
}

fn assign_impl(
    notes: &[NoteEvent],
    tuning: &Tuning,
    capo: Capo,
    options: FingeringOptions,
    allow_skip: bool,
) -> Result<PartialFingering> {
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
        skipped: 0,
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
        if candidates.is_empty() && !allow_skip {
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
                .filter(|a| sounding_at(a, note.start))
                .cloned()
                .collect();
            if allow_skip {
                let mut skipped = state.clone();
                skipped.active = active.clone();
                skipped.skipped += 1;
                next.push((skipped, None));
            }
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
                    // GuitarSet measures a strong player taking the open string
                    // when it is available. This solver still has to weigh that
                    // against position continuity, but the old flat bonus was
                    // too generous: it took the open string instead of a
                    // reference fretted note 3,671 times, which cost accuracy
                    // on the feasible excerpts (68.28% with the bonus, 70.70%
                    // without). Keep a small nudge and let position cost decide.
                    score -= 0.1;
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
                    start: note.start,
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
                        skipped: state.skipped,
                        path: state.path,
                    },
                    Some((string, fret)),
                ));
            }
        }
        if next.is_empty() {
            return Err(MtabError::UnplayableNote {
                index,
                reason: NO_FEASIBLE.into(),
            });
        }
        // Stable sort preserves deterministic generation order for equal costs.
        // Merge equivalent future-relevant states, keeping their cheapest path,
        // so redundant histories do not crowd useful alternatives out of the beam.
        next.sort_by(|a, b| {
            a.0.skipped
                .cmp(&b.0.skipped)
                .then_with(|| a.0.score.total_cmp(&b.0.score))
        });
        let mut seen = HashSet::new();
        next.retain(|(state, _)| {
            let mut active: Vec<(u8, u8, u64, u64)> = state
                .active
                .iter()
                .map(|a| (a.string, a.fret, a.start.to_bits(), a.end.to_bits()))
                .collect();
            active.sort_unstable();
            seen.insert((state.position, state.last_string, active))
        });
        next.truncate(options.beam_width);
        states = next
            .into_iter()
            .map(|(mut state, assigned)| {
                if let Some((string, fret)) = assigned {
                    paths.push(PathNode {
                        previous: state.path,
                        index,
                        string,
                        fret,
                    });
                    state.path = Some(paths.len() - 1);
                }
                state
            })
            .collect();
    }
    let mut out: Vec<Option<TabNote>> = vec![None; notes.len()];
    let mut path = states[0].path;
    while let Some(index) = path {
        let node = &paths[index];
        let note = &notes[node.index];
        out[node.index] = Some(TabNote {
            start: note.start,
            end: note.end,
            midi: note.midi,
            string: node.string,
            fret: node.fret,
            velocity: note.velocity,
            bend: None,
            technique: Vec::new(),
            tie_from_prev: false,
        });
        path = node.previous;
    }
    // Mirror the search's boundary tolerance only for assigned notes.
    let mut live: Vec<usize> = Vec::new();
    for &index in &order {
        let Some(note) = &out[index] else { continue };
        let onset = note.start;
        for &previous in &live {
            let earlier = out[previous]
                .as_mut()
                .expect("only assigned notes are live");
            if earlier.start < onset
                && earlier.end > onset
                && earlier.end - onset <= BOUNDARY_EPSILON
            {
                earlier.end = onset;
            }
        }
        live.retain(|&previous| out[previous].as_ref().unwrap().end > onset);
        live.push(index);
    }
    let unassigned_indices = out
        .iter()
        .enumerate()
        .filter_map(|(i, n)| n.is_none().then_some(i))
        .collect();
    Ok(PartialFingering {
        notes: out,
        unassigned_indices,
    })
}

/// A complete fingering and the maximum span permitted by the successful run.
#[derive(Debug, Clone, PartialEq)]
pub struct FingeringAssignment {
    /// One assignment per input note, in input order; no notes are dropped.
    pub notes: Vec<TabNote>,
    /// The configured max-span limit of the successful search, not the actual
    /// maximum span observed. A raised limit can imply an uncomfortable grip.
    pub permitted_span: u8,
    /// True if the original constrained search failed and was retried.
    pub used_wider_span: bool,
}

/// Try the normal fingering search first, then optionally retry at a wider span.
///
/// The original search is unchanged; only a no-feasible-path error triggers the
/// retry. Invalid notes, tuning and pitches outside the fretboard still error.
/// Neither attempt drops notes or allows same-string collisions. `wider_span`
/// must be between the configured maximum and 36 frets. A wider result must be
/// reviewed for physical comfort; it is not a guarantee of playability. Beam
/// pruning can still cause failures and the fallback is deterministic.
pub fn assign_notes_with_stretch(
    notes: &[NoteEvent],
    tuning: &Tuning,
    capo: Capo,
    options: FingeringOptions,
    wider_span: u8,
) -> Result<FingeringAssignment> {
    if wider_span < options.max_span || wider_span > 36 {
        return Err(MtabError::InvalidTuning(
            "wider span must be at least the normal span and at most 36 frets".into(),
        ));
    }
    match assign_notes(notes, tuning, capo, options) {
        Ok(notes) => Ok(FingeringAssignment {
            notes,
            permitted_span: options.max_span,
            used_wider_span: false,
        }),
        Err(MtabError::UnplayableNote { reason, .. })
            if reason == NO_FEASIBLE && wider_span > options.max_span =>
        {
            let notes = assign_notes(
                notes,
                tuning,
                capo,
                FingeringOptions {
                    max_span: wider_span,
                    ..options
                },
            )?;
            Ok(FingeringAssignment {
                notes,
                permitted_span: wider_span,
                used_wider_span: true,
            })
        }
        Err(error) => Err(error),
    }
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
    fn partial_reports_unplayable_and_collision_without_dropping_silently() {
        let notes = [
            note(40, 0.0, 2.0),
            note(40, 1.0, 1.5),
            note(20, 2.0, 3.0),
            note(64, 3.0, 4.0),
        ];
        let result = assign_notes_partial(
            &notes,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default(),
        )
        .unwrap();
        assert_eq!(result.unassigned_indices.len(), 2);
        assert!(result.unassigned_indices.contains(&2));
        assert_eq!(result.notes.iter().flatten().count(), 2);
        assert!(result.notes[3].is_some());
        assert!(assign_notes(
            &notes,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default()
        )
        .is_err());
        assert!(assign_notes_partial(
            &[note(40, f64::NAN, 1.0)],
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default()
        )
        .is_err());
    }

    #[test]
    fn partial_clips_assigned_boundaries_but_not_for_skipped_notes() {
        let notes = [
            note(40, 0.0, 1.0 + 5e-10),
            note(20, 0.5, 0.75),
            note(40, 1.0, 2.0),
        ];
        let result = assign_notes_partial(
            &notes,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default(),
        )
        .unwrap();
        assert_eq!(result.unassigned_indices, vec![1]);
        assert_eq!(result.notes[0].as_ref().unwrap().end, 1.0);
        assert_eq!(result.notes[2].as_ref().unwrap().start, 1.0);
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
        // E4 is spelled on the open high string: same pitch, fret 0, and it
        // costs no finger. With the open-string bonus reduced this is no longer
        // the automatic choice (the A-string fret 19 reading is equally
        // correct pitch-wise), so pin what the solver actually returns rather
        // than an arbitrary preference.
        assert_eq!(
            assigned
                .iter()
                .map(|n| (n.string, n.fret))
                .collect::<Vec<_>>(),
            vec![(1, 19), (0, 0), (3, 0), (4, 0)]
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
    #[test]
    fn wider_span_retry_is_explicit_and_keeps_every_note() {
        // Two simultaneous E3 notes require at least a 5-fret span (frets 2
        // and 7). Strict 4 fails; the optional 7-fret retry can preserve both.
        let notes = [note(52, 0.0, 1.0), note(52, 0.0, 1.0)];
        let opts = FingeringOptions::default();
        assert!(assign_notes(&notes, &Tuning::standard(), Capo::NONE, opts).is_err());
        let result =
            assign_notes_with_stretch(&notes, &Tuning::standard(), Capo::NONE, opts, 7).unwrap();
        assert!(result.used_wider_span);
        assert_eq!(result.permitted_span, 7);
        assert_eq!(result.notes.len(), 2);
        assert_ne!(result.notes[0].string, result.notes[1].string);
        let frets = [result.notes[0].fret, result.notes[1].fret];
        assert_eq!(frets[0].abs_diff(frets[1]), 5);
        assert_eq!(
            result,
            assign_notes_with_stretch(&notes, &Tuning::standard(), Capo::NONE, opts, 7).unwrap()
        );
    }

    #[test]
    fn stretch_retry_not_used_for_easy_or_invalid_input() {
        let opts = FingeringOptions::default();
        let result = assign_notes_with_stretch(
            &[note(40, 0.0, 1.0)],
            &Tuning::standard(),
            Capo::NONE,
            opts,
            7,
        )
        .unwrap();
        assert!(!result.used_wider_span);
        assert_eq!(result.permitted_span, 4);
        assert!(assign_notes_with_stretch(
            &[note(20, 0.0, 1.0)],
            &Tuning::standard(),
            Capo::NONE,
            opts,
            7
        )
        .is_err());
        assert!(assign_notes_with_stretch(
            &[note(40, 0.0, 1.0)],
            &Tuning::standard(),
            Capo::NONE,
            opts,
            3
        )
        .is_err());
        assert!(assign_notes_with_stretch(
            &[note(40, 0.0, 1.0), note(40, 0.0, 1.0)],
            &Tuning::standard(),
            Capo::NONE,
            opts,
            7
        )
        .is_err());
    }
    #[test]
    fn clips_only_subnanosecond_sequential_overlap() {
        let opts = FingeringOptions::default();
        let notes = [note(40, 0.0, 1.0 + 5e-13), note(40, 1.0, 2.0)];
        let out = assign_notes(&notes, &Tuning::standard(), Capo::NONE, opts).unwrap();
        assert_eq!(out[0].end, 1.0);
        assert_eq!(out[1].start, 1.0);
        assert_eq!(out[0].string, out[1].string);
        let real_overlap = [note(40, 0.0, 1.0 + 1e-6), note(40, 1.0, 2.0)];
        assert!(assign_notes(&real_overlap, &Tuning::standard(), Capo::NONE, opts).is_err());
    }

    #[test]
    fn boundary_tolerance_does_not_release_same_onset_tiny_notes() {
        let notes = [note(40, 0.0, 5e-10), note(40, 0.0, 1.0)];
        assert!(assign_notes(
            &notes,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default()
        )
        .is_err());
    }

    #[test]
    fn boundary_tolerance_keeps_span_valid_in_returned_notes() {
        // A previously held fretted note has a numerical end just after the
        // next attack; after clipping, the exact returned intervals no longer
        // overlap, preserving the strict four-fret span constraint.
        let notes = [note(41, 0.0, 1.0 + 5e-13), note(85, 1.0, 2.0)];
        let out = assign_notes(
            &notes,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default(),
        )
        .unwrap();
        assert_eq!(out[0].end, out[1].start);
    }
}
