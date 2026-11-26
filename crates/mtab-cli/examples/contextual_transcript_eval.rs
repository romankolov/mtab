//! Same-note predicted-transcript comparison for contextual string priors.
//!
//! Usage: contextual_transcript_eval TRANSCRIPTS_DIR COSTS_DIR AUDIO_MANIFEST OUT_DIR LAMBDA
//! It writes baseline/prior TabDocument manifests; score each with mtab eval --tab-manifest.

use std::{env, fs, path::Path};

use mtab_core::NoteEvent;
use mtab_guitar::{
    assign_notes_partial, assign_notes_partial_with_string_costs, Capo, FingeringOptions, TabNote,
    Tuning,
};
use serde_json::{json, Value};

fn read_notes(path: &Path) -> Result<Vec<NoteEvent>, Box<dyn std::error::Error>> {
    let mut notes = Vec::new();
    for line in fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let item: Value = serde_json::from_str(line)?;
        let start = item["start"].as_f64().ok_or("missing start")?;
        let end = item["end"].as_f64().ok_or("missing end")?;
        let midi = item["midi"].as_u64().ok_or("missing midi")? as u8;
        let velocity = item["velocity"].as_u64().unwrap_or(100) as u8;
        let instrument = item["instrument"].as_u64().unwrap_or(24) as u16;
        notes.push(NoteEvent {
            start,
            end,
            midi,
            velocity,
            instrument,
        });
    }
    notes.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then(a.end.total_cmp(&b.end))
            .then(a.midi.cmp(&b.midi))
    });
    Ok(notes)
}

fn read_costs(
    path: &Path,
    notes: &[NoteEvent],
) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let root: Value = serde_json::from_slice(&fs::read(path)?)?;
    let rows = root["notes"].as_array().ok_or("missing cost rows")?;
    if rows.len() != notes.len() {
        return Err(format!(
            "{}: {} cost rows for {} transcript notes",
            path.display(),
            rows.len(),
            notes.len()
        )
        .into());
    }
    let mut costs = Vec::with_capacity(rows.len());
    for (index, (row, note)) in rows.iter().zip(notes).enumerate() {
        let start = row["start"].as_f64().ok_or("missing cost start")?;
        let end = row["end"].as_f64().ok_or("missing cost end")?;
        let midi = row["midi"].as_u64().ok_or("missing cost midi")? as u8;
        if (start - note.start).abs() > 1e-9 || (end - note.end).abs() > 1e-9 || midi != note.midi {
            return Err(format!(
                "{}: cost row {index} does not match transcript event",
                path.display()
            )
            .into());
        }
        let values = row["costs"].as_array().ok_or("missing costs")?;
        if values.len() != 6 {
            return Err(format!("{}: expected six string costs", path.display()).into());
        }
        costs.push(
            values
                .iter()
                .map(|v| v.as_f64().ok_or("invalid cost"))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(costs)
}

fn ir_document(title: &str, notes: &[TabNote]) -> Value {
    let tuning = Tuning::standard();
    json!({
        "meta": {
            "title": title,
            "tempo_map": {"changes": [{"time": 0.0, "bpm": 120.0}]},
            "time_signatures": [[0, 4, 4]],
            "frets": 22,
            "source_hash": null
        },
        "tracks": [{
            "tuning": {"strings": tuning.strings, "name": "standard"},
            "capo": 0,
            "notes": notes.iter().map(|note| json!({
                "start": note.start,
                "end": note.end,
                "midi": note.midi,
                "string": note.string,
                "fret": note.fret,
                "velocity": note.velocity,
                "bend": null,
                "technique": [],
                "tie_from_prev": false
            })).collect::<Vec<_>>(),
            "instrument": null
        }],
        "chord_symbols": []
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        return Err("usage: contextual_transcript_eval TRANSCRIPTS_DIR COSTS_DIR AUDIO_MANIFEST OUT_DIR LAMBDA".into());
    }
    let transcripts = Path::new(&args[1]);
    let costs_dir = Path::new(&args[2]);
    let source_manifest = Path::new(&args[3]);
    let output_dir = Path::new(&args[4]);
    let lambda: f64 = args[5].parse()?;
    if !lambda.is_finite() || lambda < 0.0 {
        return Err("lambda must be finite and nonnegative".into());
    }
    fs::create_dir_all(output_dir.join("baseline"))?;
    fs::create_dir_all(output_dir.join("prior"))?;

    let entries: Vec<Value> = serde_json::from_slice(&fs::read(source_manifest)?)?;
    if entries.len() != 60 {
        return Err(format!("expected 60 manifest entries, found {}", entries.len()).into());
    }
    let manifest_dir = source_manifest.parent().ok_or("manifest has no parent")?;
    let mut baseline_manifest = Vec::new();
    let mut prior_manifest = Vec::new();
    let mut processed_entries = 0usize;
    let mut total_notes = 0usize;
    let mut baseline_assigned = 0usize;
    let mut prior_assigned = 0usize;
    let mut changed = 0usize;
    for (index, entry) in entries.iter().enumerate() {
        let n = index + 1;
        let transcript_path = transcripts.join(format!("n{n}.jsonl"));
        let cost_path = costs_dir.join(format!("n{n}.json"));
        if !transcript_path.exists() && !cost_path.exists() {
            continue; // the frozen mtab oracle transcript set omits failed entries
        }
        if !transcript_path.is_file() || !cost_path.is_file() {
            return Err(format!("entry {n}: transcript and cost file must both exist").into());
        }
        processed_entries += 1;
        let notes = read_notes(&transcript_path)?;
        let costs = read_costs(&cost_path, &notes)?;
        let scaled: Vec<Vec<f64>> = costs
            .iter()
            .map(|row| row.iter().map(|x| x * lambda).collect())
            .collect();
        total_notes += notes.len();
        let baseline = assign_notes_partial(
            &notes,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default(),
        )?;
        let prior = assign_notes_partial_with_string_costs(
            &notes,
            &Tuning::standard(),
            Capo::NONE,
            FingeringOptions::default(),
            &scaled,
        )?;
        let base_notes: Vec<TabNote> = baseline.notes.iter().flatten().cloned().collect();
        let prior_notes: Vec<TabNote> = prior.notes.iter().flatten().cloned().collect();
        baseline_assigned += base_notes.len();
        prior_assigned += prior_notes.len();
        changed += baseline
            .notes
            .iter()
            .zip(&prior.notes)
            .filter_map(|(a, b)| Some((a.as_ref()?, b.as_ref()?)))
            .filter(|(a, b)| a.string != b.string || a.fret != b.fret)
            .count();
        let annotation = manifest_dir
            .join(
                entry["annotation"]
                    .as_str()
                    .ok_or("manifest annotation missing")?,
            )
            .canonicalize()?;
        let duration = entry["duration"]
            .as_f64()
            .ok_or("manifest duration missing")?;
        let base_ir = output_dir.join("baseline").join(format!("e{n}.json"));
        let prior_ir = output_dir.join("prior").join(format!("e{n}.json"));
        fs::write(
            &base_ir,
            serde_json::to_vec_pretty(&ir_document(&format!("baseline e{n}"), &base_notes))?,
        )?;
        fs::write(
            &prior_ir,
            serde_json::to_vec_pretty(&ir_document(&format!("contextual e{n}"), &prior_notes))?,
        )?;
        baseline_manifest.push(
            json!({"annotation": annotation, "ir": base_ir.canonicalize()?, "duration": duration}),
        );
        prior_manifest.push(
            json!({"annotation": annotation, "ir": prior_ir.canonicalize()?, "duration": duration}),
        );
        if n % 10 == 0 {
            eprintln!("processed {n}/60 predicted transcripts");
        }
    }
    fs::write(
        output_dir.join("baseline-manifest.json"),
        serde_json::to_vec_pretty(&baseline_manifest)?,
    )?;
    fs::write(
        output_dir.join("prior-manifest.json"),
        serde_json::to_vec_pretty(&prior_manifest)?,
    )?;
    println!("entries: {processed_entries}; notes: {total_notes}; baseline assigned: {baseline_assigned}; prior assigned: {prior_assigned}; note positions changed: {changed}; lambda: {lambda}");
    println!(
        "manifests: {} and {}",
        output_dir.join("baseline-manifest.json").display(),
        output_dir.join("prior-manifest.json").display()
    );
    Ok(())
}
