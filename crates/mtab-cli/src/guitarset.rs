//! Oracle-note string-assignment evaluation on local GuitarSet JAMS files.
//!
//! This is deliberately narrower than end-to-end transcription: reference
//! pitches and timings are given to the fingering solver. It does not measure
//! note recognition, TabCNN frame metrics, or the tab-labeler-v3 baseline.

use std::path::{Path, PathBuf};

use mtab_core::{MtabError, NoteEvent, Result};
use mtab_guitar::{
    assign_notes, assign_notes_partial, assign_notes_with_stretch, Capo, FingeringOptions, Tuning,
};
use serde_json::Value;

#[derive(Debug, Clone)]
pub(super) struct Reference {
    pub(super) note: NoteEvent,
    pub(super) string: u8,
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
    partial_assigned: usize,
    partial_correct: usize,
    partial_errors: usize,
    partial_recovered: usize,
    greedy_assigned: usize,
    greedy_correct: usize,
    greedy_collisions: usize,
    reference_span_over_four: usize,
    reference_string_overlaps: usize,
    reference_other: usize,
    // Excerpts whose annotations no single hand could play, split out so the
    // solver is not graded on reference data that has no feasible answer.
    infeasible_excerpts: usize,
    infeasible_span_only: usize,
    infeasible_overlap_only: usize,
    infeasible_both: usize,
    feasible_notes: usize,
    feasible_strict_correct: usize,
    feasible_partial_correct: usize,
    feasible_partial_assigned: usize,
    feasible_greedy_correct: usize,
    feasible_anatomy: Anatomy,
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

/// Error anatomy: which notes the solver gets wrong and how.
///
/// The point is to tell chord-shape work from single-line work. A solver that
/// is wrong on isolated melody notes has a different problem from one that is
/// wrong inside chords, and the fix for the two is not the same.
#[derive(Debug, Default, Clone, Copy)]
struct Anatomy {
    /// Notes whose onset has no other note sounding — single-line playing.
    melodic_notes: usize,
    melodic_correct: usize,
    /// Notes sharing an onset with at least one other note.
    chord_notes: usize,
    chord_correct: usize,
    /// Reference string is open (fret 0).
    open_reference_notes: usize,
    open_reference_correct: usize,
    /// We picked an open string where the reference frets.
    chose_open_over_fretted: usize,
    /// Errors where we chose a higher-pitched string than the reference.
    err_higher_string: usize,
    /// Errors where we chose a lower-pitched string than the reference.
    err_lower_string: usize,
    /// Sum of |our fret - reference fret| over errors, for a mean miss.
    err_fret_distance: usize,
    errors: usize,
    /// Totals for the systematic-bias check: are we consistently too high or
    /// too low on the neck?
    reference_fret_sum: usize,
    chosen_fret_sum: usize,
    /// Errors where we picked the open string but the reference frets.
    open_instead_of_fret: usize,
}

/// Per-excerpt outcome, kept so the feasibility subset can be accumulated
/// without re-running the solver.
#[derive(Debug, Default, Clone, Copy)]
struct ExcerptScore {
    notes: usize,
    strict_correct: usize,
    partial_assigned: usize,
    partial_correct: usize,
    greedy_correct: usize,
    anatomy: Anatomy,
}

fn score_file(refs: &[Reference], counts: &mut Counts) -> (Option<Failure>, ExcerptScore) {
    let tuning = Tuning::standard();
    let notes: Vec<NoteEvent> = refs.iter().map(|r| r.note).collect();
    counts.files += 1;
    counts.notes += refs.len();
    let mut failure = None;
    let mut excerpt = ExcerptScore {
        notes: refs.len(),
        ..ExcerptScore::default()
    };
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
            counts.partial_assigned += assigned.len();
            counts.partial_correct += correct;
            excerpt.strict_correct = correct;
            excerpt.partial_assigned = assigned.len();
            excerpt.partial_correct = correct;
            excerpt.anatomy = error_anatomy(refs, &assigned);
        }
        Err(error) => {
            counts.failures += 1;
            match assign_notes_partial(&notes, &tuning, Capo::NONE, FingeringOptions::default()) {
                Ok(result) => {
                    counts.partial_recovered +=
                        usize::from(result.notes.iter().any(Option::is_some));
                    for (reference, assigned) in refs.iter().zip(&result.notes) {
                        if let Some(assigned) = assigned {
                            counts.partial_assigned += 1;
                            let right = usize::from(reference.string == assigned.string);
                            counts.partial_correct += right;
                            excerpt.partial_assigned += 1;
                            excerpt.partial_correct += right;
                        }
                    }
                }
                Err(_) => counts.partial_errors += 1,
            }
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
                excerpt.greedy_correct += 1;
            }
        }
    }
    (failure, excerpt)
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

/// Attribute the strict solver's errors to melodic vs chordal contexts and to
/// the direction of the string miss.
///
/// "Melodic" here means the note is the only one sounding at its onset, which
/// is what a single-shape template cannot help with. Everything sharing an
/// onset is "chordal".
fn error_anatomy(refs: &[Reference], assigned: &[mtab_guitar::TabNote]) -> Anatomy {
    const OPEN: [u8; 6] = [40, 45, 50, 55, 59, 64];
    let mut anatomy = Anatomy::default();
    // `refs` is onset-sorted, so simultaneous notes are adjacent.
    let mut index = 0;
    while index < refs.len() {
        let onset = refs[index].note.start;
        let mut end = index + 1;
        while end < refs.len() && (refs[end].note.start - onset).abs() < 1e-9 {
            end += 1;
        }
        let simultaneous = end - index;
        for position in index..end {
            let reference = &refs[position];
            let ours = &assigned[position];
            let correct = reference.string == ours.string;
            if simultaneous > 1 {
                anatomy.chord_notes += 1;
                anatomy.chord_correct += usize::from(correct);
            } else {
                anatomy.melodic_notes += 1;
                anatomy.melodic_correct += usize::from(correct);
            }
            let reference_fret = reference
                .note
                .midi
                .saturating_sub(OPEN[usize::from(reference.string.min(5))]);
            if reference.note.midi >= OPEN[usize::from(reference.string.min(5))]
                && reference_fret == 0
            {
                anatomy.open_reference_notes += 1;
                anatomy.open_reference_correct += usize::from(correct);
            }
            let our_fret = usize::from(ours.fret);
            anatomy.reference_fret_sum += usize::from(reference_fret);
            anatomy.chosen_fret_sum += our_fret;
            if !correct {
                anatomy.errors += 1;
                if our_fret == 0 && reference_fret > 0 {
                    anatomy.open_instead_of_fret += 1;
                }
                if ours.string > reference.string {
                    anatomy.err_higher_string += 1;
                } else {
                    anatomy.err_lower_string += 1;
                }
                anatomy.err_fret_distance += usize::from(ours.fret.abs_diff(reference_fret));
            }
        }
        index = end;
    }
    anatomy
}

/// Whether a single left hand could hold everything the annotations ask for at
/// any one instant.
///
/// Two conditions make an excerpt unplayable as written:
///
/// - **Span over four frets** among simultaneously sounding *fretted* notes.
///   Open strings are excluded: a ringing open string costs no finger.
/// - **Two notes overlapping in time on the same string.** One string cannot
///   sound two pitches at once.
///
/// GuitarSet is recorded with a hexaphonic pickup, so a note on one string can
/// appear on a neighbouring string's annotation as sympathetic resonance. The
/// duplicate-pitch, two-string case that dominates the span failures is almost
/// certainly that bleed rather than a real double stop, and there is no
/// fingering a physical hand could give it.
///
/// This uses the same span limit as the default search so the split matches
/// exactly what the solver is allowed to produce.
fn reference_feasibility(refs: &[Reference]) -> (bool, bool) {
    const OPEN: [u8; 6] = [40, 45, 50, 55, 59, 64];
    let mut wide = false;
    let mut overlap = false;
    for (index, current) in refs.iter().enumerate() {
        let now = current.note.start;
        let mut frets: Vec<u8> = Vec::new();
        let mut per_string = [0u8; 6];
        // `refs` is sorted by onset, so everything before `index` starts no
        // later than `now`; only the ones still sounding matter.
        for reference in refs.iter().take(index + 1) {
            if reference.note.end <= now {
                continue;
            }
            let string = usize::from(reference.string.min(5));
            per_string[string] += 1;
            if per_string[string] > 1 {
                overlap = true;
            }
            let fret = reference.note.midi.saturating_sub(OPEN[string]);
            if fret > 0 && reference.note.midi >= OPEN[string] {
                frets.push(fret);
            }
        }
        // Check the fretted span of everything sounding at this instant.
        if let (Some(&min), Some(&max)) = (frets.iter().min(), frets.iter().max()) {
            if max - min > 4 {
                wide = true;
            }
        }
    }
    (wide, overlap)
}

fn accumulate_anatomy(total: &mut Anatomy, part: &Anatomy) {
    total.melodic_notes += part.melodic_notes;
    total.melodic_correct += part.melodic_correct;
    total.chord_notes += part.chord_notes;
    total.chord_correct += part.chord_correct;
    total.open_reference_notes += part.open_reference_notes;
    total.open_reference_correct += part.open_reference_correct;
    total.chose_open_over_fretted += part.chose_open_over_fretted;
    total.err_higher_string += part.err_higher_string;
    total.err_lower_string += part.err_lower_string;
    total.err_fret_distance += part.err_fret_distance;
    total.errors += part.errors;
    total.reference_fret_sum += part.reference_fret_sum;
    total.chosen_fret_sum += part.chosen_fret_sum;
    total.open_instead_of_fret += part.open_instead_of_fret;
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
    let guard = crate::ReportGuard::new(report)?;
    for file in &files {
        guard.protect(file)?;
    }
    let mut counts = Counts::default();
    let mut failures = Vec::new();
    for file in &files {
        let refs = parse_jams(file)?;
        let (wide, overlap) = reference_feasibility(&refs);
        let (failure, excerpt) = score_file(&refs, &mut counts);
        if wide || overlap {
            counts.infeasible_excerpts += 1;
            counts.infeasible_span_only += usize::from(wide && !overlap);
            counts.infeasible_overlap_only += usize::from(overlap && !wide);
            counts.infeasible_both += usize::from(wide && overlap);
        } else {
            counts.feasible_notes += excerpt.notes;
            counts.feasible_strict_correct += excerpt.strict_correct;
            counts.feasible_partial_assigned += excerpt.partial_assigned;
            counts.feasible_partial_correct += excerpt.partial_correct;
            counts.feasible_greedy_correct += excerpt.greedy_correct;
            accumulate_anatomy(&mut counts.feasible_anatomy, &excerpt.anatomy);
        }
        if let Some(failure) = failure {
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
Partial search retries strict failures at span four, explicitly counts skipped notes as incorrect, and has no optimality guarantee.\n\
Fingering clips sequential note-boundary overlaps of at most 1 ns; reference times otherwise remain unchanged.\n\
The greedy baseline chooses the lowest fret per note independently and does not enforce overlaps.\n\
This is not end-to-end accuracy, a TabCNN/TDR metric, or a tab-labeler-v3 comparison.\n\n\
- Annotation files: {}\n- Reference notes: {}\n- Beam-search failed excerpts: {}\n- Adaptive retry recovered excerpts: {}\n- Recovered excerpts with actual span above four: {}\n- Largest actual span in recovered excerpts: {} frets\n- Adaptive retry still failed excerpts: {}\n- Partial retry recovered at least one note: {} excerpts\n- Partial retry errors: {} excerpts\n- Greedy notes overlapping another note on the same string: {}\n\
- At failure, reference fretted span exceeds four: {} excerpts\n\
- At failure, reference has a same-string overlap: {} excerpts\n\
- At failure, reference has neither condition: {} excerpts\n\
\nThese reference diagnostics may overlap and are not proof of failure cause.\n\n\
| Method | Assigned | Unassigned | Correct strings | String accuracy (all notes) | Wrong-string rate (assigned) |\n\
| --- | ---: | ---: | ---: | ---: | ---: |\n\
| Bounded beam | {} | {} | {} | {} | {} |\n\
| Adaptive span 4→7 | {} | {} | {} | {} | {} |\n\
| Partial retry span 4 | {} | {} | {} | {} | {} |\n\
| Per-note lowest-fret greedy | {} | {} | {} | {} | {} |\n\n\
Settings: standard EADGBE, capo 0, frets 0–22, normal span 4, adaptive retry span 7, beam width 512.\n\
GuitarSet (Xi et al., 2018), Zenodo 3371780, version 1.1.0, CC BY 4.0; annotations are user-provided and known dataset errors may affect results.\n",
        counts.files, counts.notes, counts.failures, counts.adaptive_recovered, counts.adaptive_wide_recovered, counts.adaptive_max_span, counts.adaptive_failures, counts.partial_recovered, counts.partial_errors, counts.greedy_collisions, counts.reference_span_over_four, counts.reference_string_overlaps, counts.reference_other,
        counts.assigned, counts.notes - counts.assigned, counts.correct, percent(counts.correct, counts.notes), percent(counts.assigned - counts.correct, counts.assigned),
        counts.adaptive_assigned, counts.notes - counts.adaptive_assigned, counts.adaptive_correct, percent(counts.adaptive_correct, counts.notes), percent(counts.adaptive_assigned - counts.adaptive_correct, counts.adaptive_assigned),
        counts.partial_assigned, counts.notes - counts.partial_assigned, counts.partial_correct, percent(counts.partial_correct, counts.notes), percent(counts.partial_assigned - counts.partial_correct, counts.partial_assigned),
        counts.greedy_assigned, counts.notes - counts.greedy_assigned, counts.greedy_correct, percent(counts.greedy_correct, counts.notes), percent(counts.greedy_assigned - counts.greedy_correct, counts.greedy_assigned));
    result.push_str(&format!(
        "\n## Feasibility decomposition\n\n\
An excerpt is **infeasible** when its annotations ask for something no left hand could \
produce at some instant: a fretted span over four frets between simultaneously \
sounding notes (open strings excluded — a ringing open string costs no finger), or \
two notes overlapping in time on the same string. GuitarSet was recorded with a \
hexaphonic pickup, so a note on one string can also appear on a neighbouring string's \
annotation; the duplicate-pitch two-string case that dominates the span failures is \
most likely that bleed, and no fingering could satisfy it.\n\n\
- Infeasible excerpts: {} of {} ({})\n\
- ...span over four only: {}\n\
- ...same-string overlap only: {}\n\
- ...both: {}\n\
- Feasible excerpts: {}\n\
- Reference notes in feasible excerpts: {}\n\n\
The same three methods, scored only on the feasible excerpts:\n\n\
| Method | Feasible notes | Correct strings | String accuracy | Wrong-string rate (assigned) |\n\
| --- | ---: | ---: | ---: | ---: |\n\
| Strict beam (span 4) | {} | {} | {} | {} |\n\
| Partial retry (span 4) | {} | {} | {} | {} |\n\
| Per-note lowest-fret greedy | {} | {} | {} | {} |\n\n\
Comparing the two tables shows how much of the headline gap is the reference data \
and how much is the solver.\n",
        counts.infeasible_excerpts,
        counts.files,
        percent(counts.infeasible_excerpts, counts.files),
        counts.infeasible_span_only,
        counts.infeasible_overlap_only,
        counts.infeasible_both,
        counts.files - counts.infeasible_excerpts,
        counts.feasible_notes,
        counts.feasible_notes,
        counts.feasible_strict_correct,
        percent(counts.feasible_strict_correct, counts.feasible_notes),
        percent(
            counts.feasible_notes - counts.feasible_strict_correct,
            counts.feasible_notes
        ),
        counts.feasible_notes,
        counts.feasible_partial_correct,
        percent(counts.feasible_partial_correct, counts.feasible_notes),
        percent(
            counts.feasible_partial_assigned - counts.feasible_partial_correct,
            counts.feasible_partial_assigned
        ),
        counts.feasible_notes,
        counts.feasible_greedy_correct,
        percent(counts.feasible_greedy_correct, counts.feasible_notes),
        percent(
            counts.feasible_notes - counts.feasible_greedy_correct,
            counts.feasible_notes
        ),
    ));
    let a = &counts.feasible_anatomy;
    result.push_str(&format!(
        "\n## Where the errors are (feasible excerpts only)\n\n\
Splitting the strict solver's mistakes by context says which fix is worth \
building. A note is *melodic* when nothing else sounds at its onset and \
*chordal* when it shares that onset with another note.\n\n\
| Context | Notes | Correct | Accuracy |\n\
| --- | ---: | ---: | ---: |\n\
| Melodic (alone at onset) | {} | {} | {} |\n\
| Chordal (shares onset) | {} | {} | {} |\n\
| Open-string reference | {} | {} | {} |\n\n\
- Wrong-string direction: {} chose a higher-pitched string, {} a lower one\n\
- Mean fret distance on errors: {:.2} frets\n\
- Mean fret: reference {:.2}, ours {:.2}\n\
- Errors where we took the open string but the reference frets: {}\n\n\
If melodic accuracy is already high, chord-shape templates are not the next \
lever; the loss is in single-line choices.\n",
        a.melodic_notes,
        a.melodic_correct,
        percent(a.melodic_correct, a.melodic_notes),
        a.chord_notes,
        a.chord_correct,
        percent(a.chord_correct, a.chord_notes),
        a.open_reference_notes,
        a.open_reference_correct,
        percent(a.open_reference_correct, a.open_reference_notes),
        a.err_higher_string,
        a.err_lower_string,
        if a.errors == 0 {
            0.0
        } else {
            a.err_fret_distance as f64 / a.errors as f64
        },
        if a.melodic_notes + 0 == 0 {
            0.0
        } else {
            a.reference_fret_sum as f64 / (a.melodic_notes + 0) as f64
        },
        if a.melodic_notes == 0 {
            0.0
        } else {
            a.chosen_fret_sum as f64 / a.melodic_notes as f64
        },
        a.open_instead_of_fret,
    ));
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
        assert!(score_file(&refs, &mut counts).0.is_some());
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
        assert_eq!(
            (
                counts.partial_recovered,
                counts.partial_assigned,
                counts.partial_correct,
                counts.partial_errors
            ),
            (1, 1, 1, 0)
        );
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
        assert!(score_file(&refs, &mut counts).0.is_none());
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
        assert!(score_file(&refs, &mut counts).0.is_some());
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

#[cfg(test)]
mod feasibility_tests {
    use super::*;

    fn note(start: f64, end: f64, midi: u8, string: u8) -> Reference {
        Reference {
            note: NoteEvent {
                start,
                end,
                midi,
                velocity: 100,
                instrument: 24,
            },
            string,
        }
    }

    #[test]
    fn a_plain_chord_is_feasible() {
        // Open-position A major: frets 2/2/2, well inside one hand.
        let refs = vec![
            note(0.0, 1.0, 57, 3),
            note(0.0, 1.0, 61, 4),
            note(0.0, 1.0, 64, 5),
        ];
        assert_eq!(reference_feasibility(&refs), (false, false));
    }

    #[test]
    fn a_wide_simultaneous_grip_is_infeasible() {
        // Fret 2 on string 1 and fret 8 on string 3 sound together: span 6.
        let refs = vec![note(0.0, 1.0, 47, 1), note(0.0, 1.0, 58, 2)];
        assert_eq!(reference_feasibility(&refs), (true, false));
    }

    #[test]
    fn a_ringing_open_string_does_not_count_toward_the_span() {
        // Open low E plus a fret-5 note is a five-fret numeric span, but the
        // open string needs no finger, so this is playable and feasible.
        let refs = vec![note(0.0, 2.0, 40, 0), note(0.5, 1.0, 45, 1)];
        assert_eq!(reference_feasibility(&refs), (false, false));
    }

    #[test]
    fn the_same_pitch_on_two_strings_is_infeasible() {
        // The GuitarSet hexaphonic-bleed case: midi 52 on the A string at fret 7
        // and on the D string at fret 2, sounding together.
        let refs = vec![note(0.497, 0.921, 52, 1), note(0.499, 0.847, 52, 2)];
        let (wide, overlap) = reference_feasibility(&refs);
        assert!(wide, "a 5-fret gap between simultaneous fretted notes");
        assert!(!overlap, "different strings, so not a same-string overlap");
    }

    #[test]
    fn one_string_cannot_sound_two_pitches() {
        let refs = vec![note(0.0, 1.0, 45, 1), note(0.5, 1.5, 47, 1)];
        assert_eq!(reference_feasibility(&refs), (false, true));
    }

    #[test]
    fn sequential_notes_on_one_string_are_fine() {
        let refs = vec![note(0.0, 0.5, 45, 1), note(0.5, 1.0, 47, 1)];
        assert_eq!(reference_feasibility(&refs), (false, false));
    }
}
