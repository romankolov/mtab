//! Oracle-note string-assignment evaluation on local GuitarSet JAMS files.
//!
//! This is deliberately narrower than end-to-end transcription: reference
//! pitches and timings are given to the fingering solver. It does not measure
//! note recognition, TabCNN frame metrics, or the tab-labeler-v3 baseline.

use std::path::{Path, PathBuf};

use mtab_core::{MtabError, NoteEvent, Result};
use mtab_guitar::{assign_notes, assign_notes_with_stretch, Capo, FingeringOptions, Tuning};
use serde_json::Value;

#[derive(Debug, Clone)]
pub(super) struct Reference {
    pub(super) note: NoteEvent,
    string: u8,
}

#[derive(Debug, Default)]
struct Counts {
    files: usize,
    notes: usize,
    assigned: usize,
    correct: usize,
    failures: usize,
    adaptive_assigned: usize,
    adaptive_correct: usize,
    adaptive_recovered: usize,
    adaptive_wide_recovered: usize,
    adaptive_max_span: u8,
    adaptive_failures: usize,
    greedy_assigned: usize,
    greedy_correct: usize,
    greedy_collisions: usize,
    reference_span_over_four: usize,
    reference_string_overlaps: usize,
    reference_other: usize,
}

#[derive(Debug)]
struct Failure {
    index: Option<usize>,
    reason: String,
    adaptive_error: Option<String>,
}

fn invalid(path: &Path, reason: &str) -> MtabError {
    MtabError::InvalidModelSource(format!(
        "invalid GuitarSet annotation {}: {reason}",
        path.display()
    ))
}

pub(super) fn parse_jams(path: &Path) -> Result<Vec<Reference>> {
    let json: Value = serde_json::from_reader(std::fs::File::open(path)?)?;
    let annotations = json
        .get("annotations")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(path, "missing annotations array"))?;
    let tuning = Tuning::standard();
    let mut refs = Vec::new();
    let mut seen = [false; 6];
    for annotation in annotations {
        if annotation.get("namespace").and_then(Value::as_str) != Some("note_midi") {
            continue;
        }
        let string = annotation
            .pointer("/annotation_metadata/data_source")
            .and_then(Value::as_str)
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&i| i < 6)
            .ok_or_else(|| invalid(path, "note_midi needs string data_source 0–5"))?;
        if seen[string] {
            return Err(invalid(path, "duplicate note_midi string"));
        }
        seen[string] = true;
        let values = annotation
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid(path, "missing note data"))?;
        for value in values {
            let start = value
                .get("time")
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid(path, "missing note time"))?;
            let duration = value
                .get("duration")
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid(path, "missing note duration"))?;
            let pitch = value
                .get("value")
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid(path, "missing note pitch"))?;
            if !start.is_finite()
                || start < 0.0
                || !duration.is_finite()
                || duration <= 0.0
                || !(start + duration).is_finite()
                || !pitch.is_finite()
                || !(0.0..=127.0).contains(&pitch.round())
            {
                return Err(invalid(
                    path,
                    "note needs nonnegative finite time, positive duration and valid MIDI pitch",
                ));
            }
            let midi = pitch.round() as u8;
            if midi < tuning.strings[string] || midi > tuning.strings[string] + 22 {
                return Err(invalid(
                    path,
                    "reference note outside the standard-tuning 0–22 fret range",
                ));
            }
            refs.push(Reference {
                note: NoteEvent {
                    start,
                    end: start + duration,
                    midi,
                    velocity: 100,
                    instrument: 24,
                },
                string: string as u8,
            });
        }
    }
    if seen.iter().any(|&present| !present) {
        return Err(invalid(path, "expected six note_midi string annotations"));
    }
    // Stable tie handling: string-ascending input order for equal onsets.
    refs.sort_by(|a, b| {
        a.note
            .start
            .total_cmp(&b.note.start)
            .then(a.string.cmp(&b.string))
    });
    Ok(refs)
}

fn score_file(refs: &[Reference], counts: &mut Counts) -> Option<Failure> {
    let tuning = Tuning::standard();
    let notes: Vec<NoteEvent> = refs.iter().map(|r| r.note).collect();
    counts.files += 1;
    counts.notes += refs.len();
    let mut failure = None;
    match assign_notes(&notes, &tuning, Capo::NONE, FingeringOptions::default()) {
        Ok(assigned) => {
            counts.assigned += assigned.len();
            let correct = refs
                .iter()
                .zip(&assigned)
                .filter(|(reference, assigned)| reference.string == assigned.string)
                .count();
            counts.correct += correct;
            counts.adaptive_assigned += assigned.len();
            counts.adaptive_correct += correct;
        }
        Err(error) => {
            counts.failures += 1;
            let mut adaptive_error = None;
            match assign_notes_with_stretch(
                &notes,
                &tuning,
                Capo::NONE,
                FingeringOptions::default(),
                7,
            ) {
                Ok(result) => {
                    counts.adaptive_recovered += 1;
                    let actual_span = max_fretted_span(&result.notes);
                    counts.adaptive_wide_recovered += usize::from(actual_span > 4);
                    counts.adaptive_max_span = counts.adaptive_max_span.max(actual_span);
                    counts.adaptive_assigned += result.notes.len();
                    counts.adaptive_correct += refs
                        .iter()
                        .zip(&result.notes)
                        .filter(|(reference, assigned)| reference.string == assigned.string)
                        .count();
                }
                Err(error) => {
                    counts.adaptive_failures += 1;
                    adaptive_error = Some(match error {
                        MtabError::UnplayableNote { index, reason } => {
                            format!("note {index}: {reason}")
                        }
                        other => other.to_string(),
                    });
                }
            }
            failure = Some(match error {
                MtabError::UnplayableNote { index, reason } => Failure {
                    index: Some(index),
                    reason,
                    adaptive_error,
                },
                other => Failure {
                    index: None,
                    reason: other.to_string(),
                    adaptive_error,
                },
            });
        }
    }
    // Deliberately simple per-note baseline: smallest fret, then lower string.
    // It does not enforce chord or sustain collisions and is not v3.
    let mut active_until = [0.0f64; 6];
    for r in refs {
        let candidate = tuning
            .strings
            .iter()
            .enumerate()
            .filter_map(|(i, &open)| {
                r.note
                    .midi
                    .checked_sub(open)
                    .filter(|&fret| fret <= 22)
                    .map(|fret| (fret, i as u8))
            })
            .min();
        if let Some((_, string)) = candidate {
            counts.greedy_assigned += 1;
            if active_until[usize::from(string)] > r.note.start {
                counts.greedy_collisions += 1;
            }
            active_until[usize::from(string)] = active_until[usize::from(string)].max(r.note.end);
            if string == r.string {
                counts.greedy_correct += 1;
            }
        }
    }
    failure
}

// Actual overlapping fretted-note span of an assigned track, ignoring open
// strings. This is not a substitute for finger or ergonomic modelling.
fn max_fretted_span(notes: &[mtab_guitar::TabNote]) -> u8 {
    let mut order: Vec<_> = notes.iter().collect();
    order.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut active: Vec<(f64, u8)> = Vec::new();
    let mut maximum = 0;
    for note in order {
        active.retain(|&(end, _)| end > note.start);
        if note.fret > 0 {
            active.push((note.end, note.fret));
        }
        if let (Some(min), Some(max)) = (
            active.iter().map(|a| a.1).min(),
            active.iter().map(|a| a.1).max(),
        ) {
            maximum = maximum.max(max - min);
        }
    }
    maximum
}

// Diagnostic of the reference annotations at the first failing onset. These
// are properties of the annotation, not proof of the algorithm's failure cause.
fn reference_conflicts(refs: &[Reference], index: usize) -> Option<(bool, bool)> {
    let now = refs.get(index)?.note.start;
    let mut occupied = [false; 6];
    let mut collision = false;
    let mut frets = Vec::new();
    let open = [40u8, 45, 50, 55, 59, 64];
    for reference in refs.iter().take(index + 1).filter(|r| r.note.end > now) {
        let string = usize::from(reference.string);
        if occupied[string] {
            collision = true;
        }
        occupied[string] = true;
        let fret = reference.note.midi - open[string];
        if fret > 0 {
            frets.push(fret);
        }
    }
    let span = frets
        .iter()
        .max()
        .zip(frets.iter().min())
        .map_or(0, |(max, min)| max - min);
    Some((span > 4, collision))
}

fn percent(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "n/a".into()
    } else {
        format!("{:.2}%", numerator as f64 * 100.0 / denominator as f64)
    }
}

/// Evaluate a local JAMS file or a directory of JAMS files; optional report is
/// Markdown. All solver failures count as completely unassigned excerpts so
/// a high failure rate cannot disappear from aggregate accuracy.
pub fn evaluate(path: &Path, report: Option<&Path>) -> Result<()> {
    let mut files: Vec<PathBuf> = if path.is_dir() {
        std::fs::read_dir(path)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .is_some_and(|x| x.eq_ignore_ascii_case("jams"))
            })
            .collect()
    } else if path.is_file() {
        vec![path.to_owned()]
    } else {
        return Err(MtabError::InvalidModelSource(format!(
            "GuitarSet annotation path not found: {}",
            path.display()
        )));
    };
    files.sort();
    if files.is_empty() {
        return Err(MtabError::InvalidModelSource(
            "no .jams files found in annotation directory".into(),
        ));
    }
    if let Some(out) = report {
        if let Ok(dest) = std::fs::canonicalize(out) {
            if files
                .iter()
                .any(|file| std::fs::canonicalize(file).ok().as_ref() == Some(&dest))
            {
                return Err(MtabError::InvalidModelSource(
                    "report must not overwrite a source annotation".into(),
                ));
            }
        }
    }
    let mut counts = Counts::default();
    let mut failures = Vec::new();
    for file in &files {
        let refs = parse_jams(file)?;
        if let Some(failure) = score_file(&refs, &mut counts) {
            let name = file.file_name().unwrap_or_default().to_string_lossy();
            let conflicts = failure
                .index
                .and_then(|index| reference_conflicts(&refs, index));
            if let Some((wide, collision)) = conflicts {
                counts.reference_span_over_four += usize::from(wide);
                counts.reference_string_overlaps += usize::from(collision);
                counts.reference_other += usize::from(!wide && !collision);
            }
            failures.push((name.into_owned(), failure, conflicts));
        }
    }
    let mut result = format!("# GuitarSet oracle-note string assignment\n\n\
Reference pitches and onset/offsets, not model predictions, are provided to all methods.\n\
String accuracy is correct assignments divided by **all** reference notes; unassigned notes count as incorrect.\n\
Wrong-string rate is incorrect assigned notes divided by assigned notes. Beam-search failures mark the entire excerpt unassigned.\n\
Adaptive search retries failed excerpts with a seven-fret span, which may be uncomfortable and needs review.\n\
Fingering clips sequential note-boundary overlaps of at most 1 ns; reference times otherwise remain unchanged.\n\
The greedy baseline chooses the lowest fret per note independently and does not enforce overlaps.\n\
This is not end-to-end accuracy, a TabCNN/TDR metric, or a tab-labeler-v3 comparison.\n\n\
- Annotation files: {}\n- Reference notes: {}\n- Beam-search failed excerpts: {}\n- Adaptive retry recovered excerpts: {}\n- Recovered excerpts with actual span above four: {}\n- Largest actual span in recovered excerpts: {} frets\n- Adaptive retry still failed excerpts: {}\n- Greedy notes overlapping another note on the same string: {}\n\
- At failure, reference fretted span exceeds four: {} excerpts\n\
- At failure, reference has a same-string overlap: {} excerpts\n\
- At failure, reference has neither condition: {} excerpts\n\
\nThese reference diagnostics may overlap and are not proof of failure cause.\n\n\
| Method | Assigned | Unassigned | Correct strings | String accuracy (all notes) | Wrong-string rate (assigned) |\n\
| --- | ---: | ---: | ---: | ---: | ---: |\n\
| Bounded beam | {} | {} | {} | {} | {} |\n\
| Adaptive span 4→7 | {} | {} | {} | {} | {} |\n\
| Per-note lowest-fret greedy | {} | {} | {} | {} | {} |\n\n\
Settings: standard EADGBE, capo 0, frets 0–22, normal span 4, adaptive retry span 7, beam width 512.\n\
GuitarSet (Xi et al., 2018), Zenodo 3371780, version 1.1.0, CC BY 4.0; annotations are user-provided and known dataset errors may affect results.\n",
        counts.files, counts.notes, counts.failures, counts.adaptive_recovered, counts.adaptive_wide_recovered, counts.adaptive_max_span, counts.adaptive_failures, counts.greedy_collisions, counts.reference_span_over_four, counts.reference_string_overlaps, counts.reference_other,
        counts.assigned, counts.notes - counts.assigned, counts.correct, percent(counts.correct, counts.notes), percent(counts.assigned - counts.correct, counts.assigned),
        counts.adaptive_assigned, counts.notes - counts.adaptive_assigned, counts.adaptive_correct, percent(counts.adaptive_correct, counts.notes), percent(counts.adaptive_assigned - counts.adaptive_correct, counts.adaptive_assigned),
        counts.greedy_assigned, counts.notes - counts.greedy_assigned, counts.greedy_correct, percent(counts.greedy_correct, counts.notes), percent(counts.greedy_assigned - counts.greedy_correct, counts.greedy_assigned));
    if !failures.is_empty() {
        result.push_str("\n## Strict-search failures and adaptive retry\n\n");
        for (file, failure, conflicts) in failures {
            let labels = match conflicts {
                Some((true, true)) => "reference span >4 and same-string overlap",
                Some((true, false)) => "reference span >4",
                Some((false, true)) => "reference same-string overlap",
                Some((false, false)) => "neither reference conflict observed",
                None => "reference diagnostic unavailable",
            };
            let index = failure
                .index
                .map_or_else(|| "unknown note".to_string(), |i| format!("note {i}"));
            let retry = failure.adaptive_error.map_or_else(
                || "recovered with wider span".to_string(),
                |error| format!("retry failed: {}", error.replace(['\n', '\r'], " ")),
            );
            result.push_str(&format!(
                "- `{}`: {}: {} ({}); {}\n",
                file.replace('`', "'"),
                index,
                failure.reason.replace(['\n', '\r'], " "),
                labels,
                retry
            ));
        }
    }
    if let Some(path) = report {
        std::fs::write(path, &result)?;
    } else {
        print!("{result}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counts_unplayable_excerpt_as_unassigned() {
        let refs = vec![
            Reference {
                note: NoteEvent {
                    start: 0.0,
                    end: 1.0,
                    midi: 40,
                    velocity: 100,
                    instrument: 24
                },
                string: 0
            };
            2
        ];
        let mut counts = Counts::default();
        assert!(score_file(&refs, &mut counts).is_some());
        assert_eq!(
            (
                counts.notes,
                counts.failures,
                counts.assigned,
                counts.correct
            ),
            (2, 1, 0, 0)
        );
        assert_eq!((counts.greedy_assigned, counts.greedy_correct), (2, 2));
        assert_eq!(counts.greedy_collisions, 1);
        assert_eq!(counts.adaptive_failures, 1);
    }
    #[test]
    fn correct_and_wrong_string_are_counted() {
        let refs = vec![Reference {
            note: NoteEvent {
                start: 0.0,
                end: 1.0,
                midi: 64,
                velocity: 100,
                instrument: 24,
            },
            string: 4,
        }];
        let mut counts = Counts::default();
        assert!(score_file(&refs, &mut counts).is_none());
        assert_eq!((counts.notes, counts.assigned, counts.correct), (1, 1, 0));
        assert_eq!((counts.greedy_assigned, counts.greedy_correct), (1, 0));
    }
    #[test]
    fn distinguishes_reference_span_and_collision() {
        let note = |start, end, midi, string| Reference {
            note: NoteEvent {
                start,
                end,
                midi,
                velocity: 100,
                instrument: 24,
            },
            string,
        };
        let wide = [note(0.0, 1.0, 52, 1), note(0.5, 1.0, 52, 2)];
        assert_eq!(reference_conflicts(&wide, 1), Some((true, false)));
        let same_string = [note(0.0, 1.0, 40, 0), note(0.5, 1.0, 42, 0)];
        assert_eq!(reference_conflicts(&same_string, 1), Some((false, true)));
        assert_eq!(reference_conflicts(&same_string, 2), None);
        let sequential = [note(0.0, 0.5, 40, 0), note(0.5, 1.0, 42, 0)];
        assert_eq!(reference_conflicts(&sequential, 1), Some((false, false)));
    }
    #[test]
    fn adaptive_evaluation_recovers_wider_span_without_hiding_strict_failure() {
        let make = |string| Reference {
            note: NoteEvent {
                start: 0.0,
                end: 1.0,
                midi: 52,
                velocity: 100,
                instrument: 24,
            },
            string,
        };
        let refs = [make(1), make(2)];
        let mut counts = Counts::default();
        assert!(score_file(&refs, &mut counts).is_some());
        assert_eq!((counts.failures, counts.assigned), (1, 0));
        assert_eq!(
            (
                counts.adaptive_recovered,
                counts.adaptive_failures,
                counts.adaptive_assigned
            ),
            (1, 0, 2)
        );
        assert_eq!(
            (counts.adaptive_wide_recovered, counts.adaptive_max_span),
            (1, 5)
        );
    }
}
