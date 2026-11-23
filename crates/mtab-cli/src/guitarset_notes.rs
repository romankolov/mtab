//! Note-event evaluation for one GuitarSet recording and one JSONL prediction.
//!
//! Note onsets are matched one-to-one at the same rounded MIDI pitch with a
//! 50 ms tolerance. A second metric also requires offsets to fall within the
//! larger of 50 ms and 20% of the reference duration. Both use maximum-cardinality
//! bipartite matching, not greedy nearest-neighbour matching. These are note
//! event metrics, not framewise multi-pitch, tablature or TabCNN/TDR metrics.

use std::io::{BufRead, BufReader};
use std::path::Path;

use mtab_core::{MtabError, NoteEvent, Result};

use crate::guitarset::parse_jams;

const ONSET_TOLERANCE: f64 = 0.05;
const OFFSET_MIN_TOLERANCE: f64 = 0.05;
const OFFSET_DURATION_FACTOR: f64 = 0.2;
const MAX_NOTES: usize = 4096;

fn failure(reason: impl Into<String>) -> MtabError {
    MtabError::InvalidModelSource(reason.into())
}

fn load_predictions(path: &Path) -> Result<Vec<NoteEvent>> {
    let file = std::fs::File::open(path)?;
    let mut notes = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let note: NoteEvent = serde_json::from_str(&line).map_err(|e| {
            failure(format!(
                "invalid JSONL note at {} line {}: {e}",
                path.display(),
                index + 1
            ))
        })?;
        if !note.start.is_finite()
            || !note.end.is_finite()
            || note.start < 0.0
            || note.end <= note.start
            || note.midi > 127
            || !(1..=127).contains(&note.velocity)
        {
            return Err(failure(format!(
                "invalid note times, pitch or velocity at {} line {}",
                path.display(),
                index + 1
            )));
        }
        notes.push(note);
        if notes.len() > MAX_NOTES {
            return Err(failure(format!(
                "prediction exceeds {MAX_NOTES} notes; split the recording before evaluation"
            )));
        }
    }
    Ok(notes)
}

pub(super) fn clip(notes: &mut Vec<NoteEvent>, duration: Option<f64>) {
    if let Some(end) = duration {
        notes.retain(|n| n.start < end);
        for note in notes {
            note.end = note.end.min(end);
        }
    }
}

/// Shared event-compatibility rule; note matching ignores instrument labels.
pub(super) fn compatible(reference: &NoteEvent, predicted: &NoteEvent, offsets: bool) -> bool {
    let offset_limit =
        OFFSET_MIN_TOLERANCE.max(OFFSET_DURATION_FACTOR * (reference.end - reference.start));
    reference.midi == predicted.midi
        && (reference.start - predicted.start).abs() <= ONSET_TOLERANCE + 1e-12
        && (!offsets || (reference.end - predicted.end).abs() <= offset_limit + 1e-12)
}

/// Maximum-cardinality one-to-one matching via augmenting paths.
pub(super) fn matching(reference: &[NoteEvent], predicted: &[NoteEvent], offsets: bool) -> usize {
    let edges: Vec<Vec<usize>> = reference
        .iter()
        .map(|r| {
            predicted
                .iter()
                .enumerate()
                .filter_map(|(i, p)| compatible(r, p, offsets).then_some(i))
                .collect()
        })
        .collect();
    maximum_matching(&edges, predicted.len())
}

// Hopcroft–Karp: unlike naive augmenting-path search, dense simultaneous
// pitches do not incur a cubic traversal. The note-count limit bounds the
// adjacency allocation and recursion depth.
pub(super) fn maximum_matching(edges: &[Vec<usize>], right_count: usize) -> usize {
    use std::collections::VecDeque;
    let mut left = vec![None; edges.len()];
    let mut right = vec![None; right_count];
    let mut distance = vec![usize::MAX; edges.len()];
    let mut matched = 0;
    loop {
        let mut queue = VecDeque::new();
        for (i, pair) in left.iter().enumerate() {
            if pair.is_none() {
                distance[i] = 0;
                queue.push_back(i);
            } else {
                distance[i] = usize::MAX;
            }
        }
        let mut reachable = false;
        while let Some(i) = queue.pop_front() {
            for &p in &edges[i] {
                match right[p] {
                    None => reachable = true,
                    Some(other) if distance[other] == usize::MAX => {
                        distance[other] = distance[i] + 1;
                        queue.push_back(other);
                    }
                    _ => {}
                }
            }
        }
        if !reachable {
            break;
        }
        fn augment(
            i: usize,
            edges: &[Vec<usize>],
            left: &mut [Option<usize>],
            right: &mut [Option<usize>],
            distance: &mut [usize],
        ) -> bool {
            for &p in &edges[i] {
                let eligible = match right[p] {
                    None => true,
                    Some(other) if distance[other] == distance[i] + 1 => {
                        augment(other, edges, left, right, distance)
                    }
                    _ => false,
                };
                if eligible {
                    left[i] = Some(p);
                    right[p] = Some(i);
                    return true;
                }
            }
            distance[i] = usize::MAX;
            false
        }
        for i in 0..edges.len() {
            if left[i].is_none() && augment(i, edges, &mut left, &mut right, &mut distance) {
                matched += 1;
            }
        }
    }
    matched
}

fn percent(n: usize, d: usize) -> String {
    if d == 0 {
        "n/a".into()
    } else {
        format!("{:.2}%", n as f64 * 100.0 / d as f64)
    }
}

/// Evaluate a single predicted JSONL file against its matching JAMS annotation.
pub fn evaluate(
    annotation: &Path,
    predictions: &Path,
    duration: Option<f64>,
    report: Option<&Path>,
) -> Result<()> {
    if !annotation.is_file() {
        return Err(failure(
            "note evaluation requires one .jams annotation file",
        ));
    }
    if duration.is_some_and(|d| !d.is_finite() || d <= 0.0) {
        return Err(failure(
            "--clip-duration must be a finite positive number of seconds",
        ));
    }
    let guard = crate::ReportGuard::new(report)?;
    guard.protect(annotation)?;
    guard.protect(predictions)?;
    let mut reference: Vec<NoteEvent> = parse_jams(annotation)?
        .into_iter()
        .map(|r| r.note)
        .collect();
    let mut predicted = load_predictions(predictions)?;
    clip(&mut reference, duration);
    clip(&mut predicted, duration);
    if reference.len() > MAX_NOTES {
        return Err(failure(format!(
            "reference exceeds {MAX_NOTES} notes; split the recording before evaluation"
        )));
    }
    let onset = matching(&reference, &predicted, false);
    let onset_offset = matching(&reference, &predicted, true);
    let row = |name: &str, tp: usize| -> String {
        let fp = predicted.len() - tp;
        let missing = reference.len() - tp;
        format!(
            "| {name} | {tp} | {fp} | {missing} | {} | {} | {} |",
            percent(tp, predicted.len()),
            percent(tp, reference.len()),
            percent(2 * tp, 2 * tp + fp + missing)
        )
    };
    let window = duration.map_or_else(
        || "full annotation".into(),
        |d| format!("first {d:.6} seconds; note offsets clipped at the boundary"),
    );
    let result = format!("# GuitarSet note-event evaluation\n\n\
One user-supplied JSONL prediction is compared with one user-supplied JAMS file; the user must ensure that they describe the same recording. Reference pitches are rounded to the nearest MIDI semitone. Instrument labels and strings are not evaluated.\n\n\
- Evaluation window: {window}\n- Reference notes: {}\n- Predicted notes: {}\n\n\
| Metric | True positives | False positives | False negatives | Precision | Recall | F1 |\n\
| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n{}\n{}\n\n\
Matching requires the same pitch and onset within 50 ms; the second metric additionally requires offset within max(50 ms, 20% of the reference note duration). Maximum-cardinality one-to-one matching is used independently for each row. This is not framewise multi-pitch F1, string/fret accuracy, TDR, or a tab-labeler-v3 comparison; one excerpt is not a corpus benchmark.\n\n\
GuitarSet (Xi et al., 2018), Zenodo 3371780, version 1.1.0, CC BY 4.0; known annotation errors may affect results.\n", reference.len(), predicted.len(), row("Pitch + onset", onset), row("Pitch + onset + offset", onset_offset));
    if let Some(path) = report {
        std::fs::write(path, result)?;
    } else {
        print!("{result}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn note(start: f64, end: f64, midi: u8) -> NoteEvent {
        NoteEvent {
            start,
            end,
            midi,
            velocity: 100,
            instrument: 24,
        }
    }
    #[test]
    fn maximum_matching_beats_greedy_and_respects_pitch() {
        let refs = [note(0.05, 0.5, 60), note(0.11, 0.6, 60)];
        let preds = [
            note(0.09, 0.5, 60),
            note(0.01, 0.5, 60),
            note(0.11, 0.6, 61),
        ];
        assert_eq!(matching(&refs, &preds, false), 2);
    }
    #[test]
    fn offsets_and_duplicate_notes_are_one_to_one() {
        let refs = [note(0.0, 1.0, 60), note(0.0, 1.0, 60)];
        let preds = [note(0.0, 1.19, 60), note(0.0, 1.3, 60)];
        assert_eq!(matching(&refs, &preds, false), 2);
        assert_eq!(matching(&refs, &preds, true), 1);
    }
    #[test]
    fn clips_and_drops_outside_window() {
        let mut notes = vec![note(4.9, 5.2, 60), note(5.0, 6.0, 61)];
        clip(&mut notes, Some(5.0));
        assert_eq!(notes, vec![note(4.9, 5.0, 60)]);
    }
    #[test]
    fn matching_agrees_with_exhaustive_small_graphs() {
        fn brute(edges: &[Vec<usize>], left: usize, used: u8) -> usize {
            if left == edges.len() {
                return 0;
            }
            let mut best = brute(edges, left + 1, used);
            for &right in &edges[left] {
                if used & (1 << right) == 0 {
                    best = best.max(1 + brute(edges, left + 1, used | (1 << right)));
                }
            }
            best
        }
        for mask in 0u16..512 {
            let edges: Vec<Vec<usize>> = (0..3)
                .map(|i| (0..3).filter(|j| mask & (1 << (i * 3 + j)) != 0).collect())
                .collect();
            assert_eq!(
                maximum_matching(&edges, 3),
                brute(&edges, 0, 0),
                "graph mask {mask}"
            );
        }
    }
}
