//! Oracle-note string-assignment evaluation on local GuitarSet JAMS files.
//!
//! This is deliberately narrower than end-to-end transcription: reference
//! pitches and timings are given to the fingering solver. It does not measure
//! note recognition, TabCNN frame metrics, or the tab-labeler-v3 baseline.

use std::path::{Path, PathBuf};

use mtab_core::{MtabError, NoteEvent, Result};
use mtab_guitar::{assign_notes, Capo, FingeringOptions, Tuning};
use serde_json::Value;

#[derive(Debug, Clone)]
struct Reference {
    note: NoteEvent,
    string: u8,
}

#[derive(Debug, Default)]
struct Counts {
    files: usize,
    notes: usize,
    assigned: usize,
    correct: usize,
    failures: usize,
    greedy_assigned: usize,
    greedy_correct: usize,
    greedy_collisions: usize,
}

fn invalid(path: &Path, reason: &str) -> MtabError {
    MtabError::InvalidModelSource(format!(
        "invalid GuitarSet annotation {}: {reason}",
        path.display()
    ))
}

fn parse_jams(path: &Path) -> Result<Vec<Reference>> {
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

fn score_file(refs: &[Reference], counts: &mut Counts) -> Option<String> {
    let tuning = Tuning::standard();
    let notes: Vec<NoteEvent> = refs.iter().map(|r| r.note).collect();
    counts.files += 1;
    counts.notes += refs.len();
    let mut failure = None;
    match assign_notes(&notes, &tuning, Capo::NONE, FingeringOptions::default()) {
        Ok(assigned) => {
            counts.assigned += assigned.len();
            counts.correct += refs
                .iter()
                .zip(assigned)
                .filter(|(reference, assigned)| reference.string == assigned.string)
                .count();
        }
        Err(error) => {
            counts.failures += 1;
            failure = Some(match error {
                MtabError::UnplayableNote { index, reason } => format!("note {index}: {reason}"),
                other => other.to_string(),
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
        if let Some(reason) = score_file(&parse_jams(file)?, &mut counts) {
            let name = file.file_name().unwrap_or_default().to_string_lossy();
            failures.push((name.into_owned(), reason));
        }
    }
    let mut result = format!("# GuitarSet oracle-note string assignment\n\n\
Reference pitches and onset/offsets, not model predictions, are provided to both methods.\n\
String accuracy is correct assignments divided by **all** reference notes; unassigned notes count as incorrect.\n\
Wrong-string rate is incorrect assigned notes divided by assigned notes. Beam-search failures mark the entire excerpt unassigned.\n\
The greedy baseline chooses the lowest fret per note independently and does not enforce overlaps.\n\
This is not end-to-end accuracy, a TabCNN/TDR metric, or a tab-labeler-v3 comparison.\n\n\
- Annotation files: {}\n- Reference notes: {}\n- Beam-search failed excerpts: {}\n- Greedy notes overlapping another note on the same string: {}\n\n\
| Method | Assigned | Unassigned | Correct strings | String accuracy (all notes) | Wrong-string rate (assigned) |\n\
| --- | ---: | ---: | ---: | ---: | ---: |\n\
| Bounded beam | {} | {} | {} | {} | {} |\n\
| Per-note lowest-fret greedy | {} | {} | {} | {} | {} |\n\n\
Settings: standard EADGBE, capo 0, frets 0–22, maximum concurrent fretted span 4, beam width 512.\n\
GuitarSet (Xi et al., 2018), Zenodo 3371780, version 1.1.0, CC BY 4.0; annotations are user-provided and known dataset errors may affect results.\n",
        counts.files, counts.notes, counts.failures, counts.greedy_collisions,
        counts.assigned, counts.notes - counts.assigned, counts.correct, percent(counts.correct, counts.notes), percent(counts.assigned - counts.correct, counts.assigned),
        counts.greedy_assigned, counts.notes - counts.greedy_assigned, counts.greedy_correct, percent(counts.greedy_correct, counts.notes), percent(counts.greedy_assigned - counts.greedy_correct, counts.greedy_assigned));
    if !failures.is_empty() {
        result.push_str("\n## Unassigned excerpts\n\n");
        for (file, reason) in failures {
            result.push_str(&format!(
                "- `{}`: {}\n",
                file.replace('`', "'"),
                reason.replace(['\n', '\r'], " ")
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
}
