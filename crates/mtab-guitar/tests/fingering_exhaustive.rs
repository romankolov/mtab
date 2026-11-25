//! Small exhaustive reference search for the bounded fingering beam.
//!
//! This separately enumerates all assignments for short cases; it checks the
//! published solver objective, not musical quality or corpus accuracy.
use mtab_core::NoteEvent;
use mtab_guitar::{assign_notes, Capo, FingeringOptions, TabNote, Tuning};

const OPEN: [u8; 6] = [40, 45, 50, 55, 59, 64];

/// Must match `mtab_guitar::fingering`'s start window; a mismatch makes this
/// oracle check a different objective than the solver.
const START_POSITION: u8 = 7;

// Calculate constraints and cost from a complete path rather than maintaining
// the beam's active-state representation. Notes are in chronological order.
fn path_cost(notes: &[NoteEvent], path: &[(u8, u8)], span: u8) -> Option<f64> {
    let mut score = 0.0;
    // Mirrors `State::reach` / `State::shifted` in the solver: a window of
    // frets the hand occupies, moved one fret at a time, starting from the same
    // seeded window so both are minimising the same objective.
    let mut position: Option<(u8, u8)> = Some((START_POSITION - 2, START_POSITION + 2));
    let mut last: Option<u8> = None;
    for (i, &(string, fret)) in path.iter().enumerate() {
        let mut sounding_frets = Vec::new();
        for j in 0..i {
            if notes[j].end > notes[i].start {
                if path[j].0 == string {
                    return None;
                }
                if path[j].1 != 0 {
                    sounding_frets.push(path[j].1);
                }
            }
        }
        if fret != 0 {
            sounding_frets.push(fret);
        }
        if let (Some(lo), Some(hi)) = (sounding_frets.iter().min(), sounding_frets.iter().max()) {
            if hi - lo > span {
                return None;
            }
        }
        if fret == 0 {
            // Must mirror `fingering.rs`; a mismatch turns this test into a
            // check of two different objectives.
            score += 0.4;
        } else {
            score += f64::from(fret) * 0.015;
            let shifted = match position {
                None => {
                    let half = span / 2;
                    (fret.saturating_sub(half), fret.saturating_add(span - half))
                }
                Some((low, high)) => {
                    if fret >= low && fret <= high {
                        (low, high)
                    } else if fret < low {
                        let drop = low - fret;
                        (low - drop, high.saturating_sub(drop))
                    } else {
                        let rise = fret - high;
                        (low.saturating_add(rise), high.saturating_add(rise))
                    }
                }
            };
            score += match position {
                None => 0.0,
                Some((low, high)) if fret >= low && fret <= high => 0.0,
                Some((low, _)) if fret < low => f64::from(low - fret),
                Some((_, high)) => f64::from(fret - high),
            };
            position = Some(shifted);
        }
        if let Some(old) = last {
            score += 0.15 * f64::from(string.abs_diff(old));
        }
        last = Some(string);
    }
    Some(score)
}

fn enumerate(notes: &[NoteEvent], span: u8, capo: u8) -> Option<f64> {
    fn visit(
        notes: &[NoteEvent],
        span: u8,
        capo: u8,
        path: &mut Vec<(u8, u8)>,
        best: &mut Option<f64>,
    ) {
        if path.len() == notes.len() {
            if let Some(score) = path_cost(notes, path, span) {
                *best = Some(best.map_or(score, |old| old.min(score)));
            }
            return;
        }
        let pitch = notes[path.len()].midi;
        for (string, open) in OPEN.iter().enumerate() {
            let fret = i16::from(pitch) - i16::from(*open) - i16::from(capo);
            if (0..=i16::from(22 - capo)).contains(&fret) {
                path.push((string as u8, fret as u8));
                if path_cost(notes, path, span).is_some() {
                    visit(notes, span, capo, path, best);
                }
                path.pop();
            }
        }
    }
    let mut best = None;
    visit(notes, span, capo, &mut Vec::new(), &mut best);
    best
}

fn next(seed: &mut u64, limit: u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 32) % limit
}

fn score_result(notes: &[NoteEvent], result: &[TabNote], capo: u8, span: u8) -> f64 {
    let mut order: Vec<_> = (0..notes.len()).collect();
    order.sort_by(|&a, &b| notes[a].start.total_cmp(&notes[b].start).then(a.cmp(&b)));
    let mut sorted = Vec::new();
    let mut path = Vec::new();
    for i in order {
        let note = &result[i];
        assert_eq!(
            (note.start, note.end, note.midi, note.velocity),
            (
                notes[i].start,
                notes[i].end,
                notes[i].midi,
                notes[i].velocity
            )
        );
        assert_eq!(OPEN[usize::from(note.string)] + capo + note.fret, note.midi);
        assert!(note.fret <= 22 - capo);
        sorted.push(notes[i]);
        path.push((note.string, note.fret));
    }
    path_cost(&sorted, &path, span).expect("solver returned an invalid path")
}

#[test]
fn wide_beam_agrees_with_exhaustive_short_cases() {
    let mut seed = 0x83017a9u64;
    let mut feasible = 0;
    let mut infeasible = 0;
    for case in 0..400 {
        let count = 1 + next(&mut seed, 6) as usize;
        let capo = next(&mut seed, 4) as u8;
        let span = 1 + next(&mut seed, 8) as u8;
        let mut notes = Vec::new();
        for _ in 0..count {
            let start = next(&mut seed, 8) as f64 * 0.5;
            notes.push(NoteEvent {
                start,
                end: start + (1 + next(&mut seed, 4)) as f64 * 0.5,
                midi: 40 + capo + next(&mut seed, u64::from(47 - capo)) as u8,
                velocity: 91,
                instrument: 24,
            });
        }
        let mut sorted = notes.clone();
        sorted.sort_by(|a, b| a.start.total_cmp(&b.start));
        let best = enumerate(&sorted, span, capo);
        let actual = assign_notes(
            &notes,
            &Tuning::standard(),
            Capo::new(capo).unwrap(),
            FingeringOptions {
                max_fret: 22,
                max_span: span,
                beam_width: 4096,
            },
        );
        match (best, actual) {
            (None, Err(_)) => infeasible += 1,
            (Some(expected), Ok(result)) => {
                feasible += 1;
                let score = score_result(&notes, &result, capo, span);
                assert!(
                    (score - expected).abs() < 1e-9,
                    "case {case}: got {score}, expected {expected}, notes={notes:?}"
                );
            }
            (expected, result) => {
                panic!("case {case}: exhaustive={expected:?}, beam={result:?}, notes={notes:?}")
            }
        }
    }
    eprintln!("exhaustive fingering cases: {feasible} feasible, {infeasible} infeasible");
    assert!(
        feasible > 50 && infeasible > 50,
        "insufficient coverage: {feasible}, {infeasible}"
    );
}

// Independent exhaustive enumeration of assignments *and* omissions. The
// score evaluator receives only selected notes; it does not model beam states.
fn enumerate_partial(notes: &[NoteEvent], span: u8, capo: u8) -> (usize, f64) {
    fn visit(
        notes: &[NoteEvent],
        i: usize,
        span: u8,
        capo: u8,
        selected: &mut Vec<NoteEvent>,
        path: &mut Vec<(u8, u8)>,
        best: &mut (usize, f64),
    ) {
        if i == notes.len() {
            let skipped = notes.len() - selected.len();
            let score = path_cost(selected, path, span).unwrap();
            if skipped < best.0 || (skipped == best.0 && score < best.1) {
                *best = (skipped, score);
            }
            return;
        }
        visit(notes, i + 1, span, capo, selected, path, best);
        for (string, open) in OPEN.iter().enumerate() {
            let fret = i16::from(notes[i].midi) - i16::from(*open) - i16::from(capo);
            if (0..=i16::from(22 - capo)).contains(&fret) {
                selected.push(notes[i]);
                path.push((string as u8, fret as u8));
                if path_cost(selected, path, span).is_some() {
                    visit(notes, i + 1, span, capo, selected, path, best);
                }
                selected.pop();
                path.pop();
            }
        }
    }
    let mut best = (usize::MAX, f64::INFINITY);
    visit(
        notes,
        0,
        span,
        capo,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut best,
    );
    best
}

#[test]
fn partial_wide_beam_agrees_with_exhaustive_skip_search() {
    let mut seed = 0x194db51u64;
    let mut with_skips = 0;
    let mut without_skips = 0;
    for case in 0..400 {
        let count = 1 + next(&mut seed, 6) as usize;
        let capo = next(&mut seed, 4) as u8;
        let span = 1 + next(&mut seed, 8) as u8;
        let mut notes = Vec::new();
        for _ in 0..count {
            let start = next(&mut seed, 6) as f64 * 0.5;
            notes.push(NoteEvent {
                start,
                end: start + (1 + next(&mut seed, 4)) as f64 * 0.5,
                midi: 36 + next(&mut seed, 56) as u8,
                velocity: 91,
                instrument: 24,
            });
        }
        let mut sorted = notes.clone();
        sorted.sort_by(|a, b| a.start.total_cmp(&b.start));
        let expected = enumerate_partial(&sorted, span, capo);
        let result = mtab_guitar::assign_notes_partial(
            &notes,
            &Tuning::standard(),
            Capo::new(capo).unwrap(),
            FingeringOptions {
                max_fret: 22,
                max_span: span,
                beam_width: 4096,
            },
        )
        .unwrap();
        let actual_indices: Vec<_> = result
            .notes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.is_none().then_some(i))
            .collect();
        assert_eq!(result.unassigned_indices, actual_indices);
        assert_eq!(actual_indices.len(), expected.0, "case {case}: {notes:?}");
        let mut selected_notes = Vec::new();
        let mut selected_output = Vec::new();
        for (original, output) in notes.iter().zip(result.notes) {
            if let Some(output) = output {
                selected_notes.push(*original);
                selected_output.push(output);
            }
        }
        let score = score_result(&selected_notes, &selected_output, capo, span);
        assert!(
            (score - expected.1).abs() < 1e-9,
            "case {case}: got {score}, expected {}, notes={notes:?}",
            expected.1
        );
        if expected.0 == 0 {
            without_skips += 1;
        } else {
            with_skips += 1;
        }
    }
    eprintln!("partial exhaustive cases: {without_skips} without skips, {with_skips} with skips");
    assert!(with_skips > 50 && without_skips > 50);
}
