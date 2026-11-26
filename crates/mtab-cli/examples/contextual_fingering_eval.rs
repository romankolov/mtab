//! Evaluate optional learned string costs through the production fingering beam.
//!
//! Usage: contextual_fingering_eval ANNOTATIONS_DIR COSTS_DIR SESSIONS LAMBDAS
//! See benches/m6-contextual-string-probe.md for the frozen protocol.

use std::{env, fs, path::Path};

use mtab_core::NoteEvent;
use mtab_guitar::{assign_notes, assign_notes_with_string_costs, Capo, FingeringOptions, Tuning};
use serde_json::Value;

#[derive(Clone)]
struct RefNote {
    note: NoteEvent,
    string: u8,
}

#[derive(Default)]
struct Totals {
    files: usize,
    notes: usize,
    correct: usize,
    failures: usize,
}

fn read_refs(path: &Path) -> Result<Vec<RefNote>, Box<dyn std::error::Error>> {
    let root: Value = serde_json::from_slice(&fs::read(path)?)?;
    let mut refs = Vec::new();
    for ann in root["annotations"]
        .as_array()
        .ok_or("missing annotations")?
    {
        if ann["namespace"] != "note_midi" {
            continue;
        }
        let string: u8 = ann["annotation_metadata"]["data_source"]
            .as_str()
            .ok_or("missing data_source")?
            .parse()?;
        for item in ann["data"].as_array().ok_or("missing note data")? {
            let start = item["time"].as_f64().ok_or("missing time")?;
            let duration = item["duration"].as_f64().ok_or("missing duration")?;
            let pitch = item["value"].as_f64().ok_or("missing pitch")?.round() as u8;
            refs.push(RefNote {
                note: NoteEvent {
                    start,
                    end: start + duration,
                    midi: pitch,
                    velocity: 100,
                    instrument: 24,
                },
                string,
            });
        }
    }
    refs.sort_by(|a, b| {
        a.note
            .start
            .total_cmp(&b.note.start)
            .then(a.string.cmp(&b.string))
            .then(a.note.end.total_cmp(&b.note.end))
            .then(a.note.midi.cmp(&b.note.midi))
    });
    Ok(refs)
}

fn read_costs(path: &Path, refs: &[RefNote]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let root: Value = serde_json::from_slice(&fs::read(path)?)?;
    let rows = root["notes"].as_array().ok_or("missing costs notes")?;
    if rows.len() != refs.len() {
        return Err(format!(
            "{}: {} cost rows for {} JAMS notes",
            path.display(),
            rows.len(),
            refs.len()
        )
        .into());
    }
    let mut costs = Vec::with_capacity(rows.len());
    for (index, (row, reference)) in rows.iter().zip(refs).enumerate() {
        let start = row["start"].as_f64().ok_or("missing cost start")?;
        let end = row["end"].as_f64().ok_or("missing cost end")?;
        let midi = row["midi"].as_u64().ok_or("missing cost pitch")? as u8;
        let string = row["reference_string"]
            .as_u64()
            .ok_or("missing cost string")? as u8;
        if (start - reference.note.start).abs() > 1e-9
            || (end - reference.note.end).abs() > 1e-9
            || midi != reference.note.midi
            || string != reference.string
        {
            return Err(format!(
                "{}: cost row {index} mismatch: cost=({start:?},{end:?},{midi},{string}), jams=({:?},{:?},{},{})",
                path.display(), reference.note.start, reference.note.end, reference.note.midi, reference.string
            )
            .into());
        }
        let row_costs = row["costs"].as_array().ok_or("missing candidate costs")?;
        if row_costs.len() != 6 {
            return Err(
                format!("{}: cost row {index} must have six strings", path.display()).into(),
            );
        }
        costs.push(
            row_costs
                .iter()
                .map(|value| value.as_f64().ok_or("invalid cost value"))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(costs)
}

fn feasible(refs: &[RefNote]) -> bool {
    const OPEN: [u8; 6] = [40, 45, 50, 55, 59, 64];
    for (index, current) in refs.iter().enumerate() {
        let now = current.note.start;
        let mut frets = Vec::new();
        let mut per_string = [0u8; 6];
        for reference in refs.iter().take(index + 1) {
            if reference.note.end <= now {
                continue;
            }
            let string = usize::from(reference.string.min(5));
            per_string[string] += 1;
            if per_string[string] > 1 {
                return false;
            }
            let fret = reference.note.midi.saturating_sub(OPEN[string]);
            if fret > 0 && reference.note.midi >= OPEN[string] {
                frets.push(fret);
            }
        }
        if let (Some(min), Some(max)) = (frets.iter().min(), frets.iter().max()) {
            if max - min > 4 {
                return false;
            }
        }
    }
    true
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: contextual_fingering_eval ANNOTATIONS_DIR COSTS_DIR SESSIONS LAMBDAS".into(),
        );
    }
    let annotations = Path::new(&args[1]);
    let costs_dir = Path::new(&args[2]);
    let sessions: Vec<&str> = args[3].split(',').collect();
    let lambdas: Vec<f64> = args[4]
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if lambdas.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err("lambdas must be finite and nonnegative".into());
    }
    let tuning = Tuning::standard();
    let options = FingeringOptions::default();
    let mut baseline = Totals::default();
    let mut prior_totals: Vec<Totals> = lambdas.iter().map(|_| Totals::default()).collect();

    let mut paths: Vec<_> = fs::read_dir(annotations)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jams"))
        .filter(|path| {
            sessions.contains(
                &path.file_stem().unwrap().to_string_lossy()[..2]
                    .to_owned()
                    .as_str(),
            )
        })
        .collect();
    paths.sort();

    let mut feasible_files = 0usize;
    let mut infeasible_files = 0usize;
    for path in paths {
        let refs = read_refs(&path)?;
        let cost_path = costs_dir.join(format!(
            "{}.json",
            path.file_stem().unwrap().to_string_lossy()
        ));
        let costs = read_costs(&cost_path, &refs)?;
        if !feasible(&refs) {
            infeasible_files += 1;
            continue;
        }
        feasible_files += 1;
        let notes: Vec<_> = refs.iter().map(|r| r.note).collect();
        baseline.files += 1;
        baseline.notes += refs.len();
        match assign_notes(&notes, &tuning, Capo::NONE, options) {
            Ok(output) => {
                baseline.correct += output
                    .iter()
                    .zip(&refs)
                    .filter(|(out, r)| out.string == r.string)
                    .count();
            }
            Err(_) => baseline.failures += 1,
        }
        for (i, lambda) in lambdas.iter().enumerate() {
            let scaled: Vec<Vec<f64>> = costs
                .iter()
                .map(|row| row.iter().map(|cost| cost * lambda).collect())
                .collect();
            let totals = &mut prior_totals[i];
            totals.files += 1;
            totals.notes += refs.len();
            match assign_notes_with_string_costs(&notes, &tuning, Capo::NONE, options, &scaled) {
                Ok(output) => {
                    totals.correct += output
                        .iter()
                        .zip(&refs)
                        .filter(|(out, r)| out.string == r.string)
                        .count();
                }
                Err(_) => totals.failures += 1,
            }
        }
        if feasible_files % 10 == 0 {
            eprintln!("scored {feasible_files} feasible files");
        }
    }
    let pct = |correct: usize, notes: usize| {
        if notes == 0 {
            0.0
        } else {
            100.0 * correct as f64 / notes as f64
        }
    };
    println!("feasible files: {feasible_files}; infeasible files skipped: {infeasible_files}");
    println!(
        "baseline: {}/{} correct = {:.2}%; failed feasible files: {}",
        baseline.correct,
        baseline.notes,
        pct(baseline.correct, baseline.notes),
        baseline.failures
    );
    for (lambda, totals) in lambdas.iter().zip(prior_totals) {
        println!(
            "lambda={lambda}: {}/{} correct = {:.2}%; failed feasible files: {}",
            totals.correct,
            totals.notes,
            pct(totals.correct, totals.notes),
            totals.failures
        );
    }
    Ok(())
}
