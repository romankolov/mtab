//! Synthetic JAMS-shaped integration tests; no dataset or weights required.
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static SCRATCH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        loop {
            let seq = SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("mtab-eval-{}-{id}-{seq}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("cannot create temporary directory: {error}"),
            }
        }
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn evaluates_jams_and_protects_annotations() {
    let dir = Scratch::new();
    let jams = dir.0.join("example.jams");
    let mut annotations = Vec::new();
    for string in 0..6 {
        let data = if string == 0 {
            vec![serde_json::json!({"time": 0.0, "duration": 1.0, "value": 40.1})]
        } else if string == 1 {
            vec![serde_json::json!({"time": 1.0, "duration": 1.0, "value": 45.0})]
        } else if string == 4 {
            vec![serde_json::json!({"time": 2.0, "duration": 1.0, "value": 64.0})]
        } else {
            vec![]
        };
        annotations.push(serde_json::json!({"namespace": "note_midi", "annotation_metadata": {"data_source": string.to_string()}, "data": data}));
    }
    let original = serde_json::to_vec(&serde_json::json!({"annotations": annotations})).unwrap();
    std::fs::write(&jams, &original).unwrap();
    let report = dir.0.join("evaluation.md");
    let binary = env!("CARGO_BIN_EXE_mtab");
    let result = Command::new(binary)
        .args(["eval", "--oracle-notes", "--annotations"])
        .arg(&jams)
        .arg("--report")
        .arg(&report)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = std::fs::read_to_string(&report).unwrap();
    assert!(output.contains("Reference notes: 3"), "{output}");
    assert!(
        output.contains("| Bounded beam | 3 | 0 | 2 | 66.67% | 33.33% |"),
        "{output}"
    );
    assert!(
        output.contains("| Adaptive span 4→7 | 3 | 0 | 2 | 66.67% | 33.33% |"),
        "{output}"
    );
    assert!(
        output.contains("| Per-note lowest-fret greedy | 3 | 0 | 2 | 66.67% | 33.33% |"),
        "{output}"
    );
    let result = Command::new(binary)
        .args(["eval", "--oracle-notes", "--annotations"])
        .arg(&jams)
        .arg("--report")
        .arg(&jams)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(std::fs::read(&jams).unwrap(), original);
    let result = Command::new(binary)
        .args(["eval", "--annotations"])
        .arg(&jams)
        .output()
        .unwrap();
    assert!(
        !result.status.success(),
        "full evaluation must not silently claim oracle results"
    );
}

#[test]
fn sequential_annotation_roundoff_does_not_discard_excerpt() {
    let dir = Scratch::new();
    let jams = dir.0.join("boundary.jams");
    let mut annotations = Vec::new();
    for string in 0..6 {
        let data = if string == 0 {
            vec![
                serde_json::json!({"time": 0.0, "duration": 1.0000000000005, "value": 40.0}),
                serde_json::json!({"time": 1.0, "duration": 1.0, "value": 40.0}),
            ]
        } else {
            vec![]
        };
        annotations.push(serde_json::json!({"namespace":"note_midi", "annotation_metadata":{"data_source":string.to_string()}, "data":data}));
    }
    std::fs::write(
        &jams,
        serde_json::to_vec(&serde_json::json!({"annotations": annotations})).unwrap(),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_mtab"))
        .args(["eval", "--oracle-notes", "--annotations"])
        .arg(&jams)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(
        text.contains("| Bounded beam | 2 | 0 | 2 | 100.00% | 0.00% |"),
        "{text}"
    );
}

#[test]
fn scores_predicted_notes_with_onset_offset_and_clipping() {
    let dir = Scratch::new();
    let jams = dir.0.join("reference.jams");
    let predictions = dir.0.join("prediction.jsonl");
    let mut annotations = Vec::new();
    for string in 0..6 {
        let data = match string {
            0 => vec![serde_json::json!({"time":0.0,"duration":1.0,"value":40.0})],
            1 => vec![serde_json::json!({"time":1.0,"duration":1.0,"value":45.0})],
            _ => vec![],
        };
        annotations.push(serde_json::json!({"namespace":"note_midi", "annotation_metadata":{"data_source":string.to_string()}, "data":data}));
    }
    std::fs::write(
        &jams,
        serde_json::to_vec(&serde_json::json!({"annotations":annotations})).unwrap(),
    )
    .unwrap();
    let jsonl = concat!(
        "{\"start\":0.02,\"end\":1.02,\"midi\":40,\"velocity\":100,\"instrument\":24}\n",
        "{\"start\":1.02,\"end\":1.6,\"midi\":45,\"velocity\":100,\"instrument\":24}\n",
        "{\"start\":1.5,\"end\":1.8,\"midi\":50,\"velocity\":100,\"instrument\":24}\n"
    );
    std::fs::write(&predictions, jsonl).unwrap();
    let binary = env!("CARGO_BIN_EXE_mtab");
    let run = |extra: &[&str]| {
        let mut cmd = Command::new(binary);
        cmd.args(["eval", "--annotations"])
            .arg(&jams)
            .arg("--predictions")
            .arg(&predictions)
            .args(extra)
            .output()
            .unwrap()
    };
    let output = run(&[]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(
        text.contains("| Pitch + onset | 2 | 1 | 0 | 66.67% | 100.00% | 80.00% |"),
        "{text}"
    );
    assert!(
        text.contains("| Pitch + onset + offset | 1 | 2 | 1 | 33.33% | 50.00% | 40.00% |"),
        "{text}"
    );
    let output = run(&["--clip-duration", "1"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(
        text.contains("| Pitch + onset + offset | 1 | 0 | 0 | 100.00% | 100.00% | 100.00% |"),
        "{text}"
    );
    assert!(!run(&["--clip-duration", "0"]).status.success());
    assert!(!run(&["--report", predictions.to_str().unwrap()])
        .status
        .success());
    assert_eq!(std::fs::read_to_string(predictions).unwrap(), jsonl);
}

#[test]
fn scores_emitted_ir_framewise_and_rejects_misleading_inputs() {
    use mtab_guitar::{Capo, GuitarTrack, TabDocument, TabNote, TempoMap, TrackMeta, Tuning};
    let scratch = Scratch::new();
    let jams = scratch.0.join("reference.jams");
    let ir = scratch.0.join("prediction.json");
    let mut annotations = Vec::new();
    for string in 0..6 {
        let data = match string {
            0 => vec![serde_json::json!({"time":0.0,"duration":1.0,"value":50.0})],
            1 => vec![serde_json::json!({"time":1.0,"duration":1.0,"value":45.0})],
            _ => vec![],
        };
        annotations.push(serde_json::json!({"namespace":"note_midi", "annotation_metadata":{"data_source":string.to_string()}, "data":data}));
    }
    std::fs::write(
        &jams,
        serde_json::to_vec(&serde_json::json!({"annotations":annotations})).unwrap(),
    )
    .unwrap();
    let note = |start, end, midi, string, fret| TabNote {
        start,
        end,
        midi,
        string,
        fret,
        velocity: 100,
        bend: None,
        technique: vec![],
        tie_from_prev: false,
    };
    let mut doc = TabDocument {
        meta: TrackMeta {
            title: None,
            tempo_map: TempoMap::constant(120.0),
            time_signatures: vec![(0, 4, 4)],
            frets: 22,
            source_hash: None,
        },
        tracks: vec![GuitarTrack {
            tuning: Tuning::standard(),
            capo: Capo::NONE,
            instrument: None,
            notes: vec![note(0.0, 1.0, 50, 1, 5), note(1.0, 2.0, 45, 1, 0)],
        }],
        chord_symbols: vec![],
    };
    let save = |doc: &TabDocument| std::fs::write(&ir, doc.to_json().unwrap()).unwrap();
    save(&doc);
    let run = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_mtab"))
            .args(["eval", "--annotations"])
            .arg(&jams)
            .arg("--tab-ir")
            .arg(&ir)
            .args(extra)
            .output()
            .unwrap()
    };
    let output = run(&["--clip-duration", "2"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = String::from_utf8(output.stdout).unwrap();
    assert!(
        report.contains("| Multi-pitch frames | 200 | 0 | 0 | 100.00% | 100.00% | 100.00% |"),
        "{report}"
    );
    assert!(
        report.contains("| Tab position frames | 100 | 100 | 100 | 50.00% | 50.00% | 50.00% |"),
        "{report}"
    );
    assert!(
        report.contains(
            "TDR-style ratio (correct tab positions / correct unique pitches): 100 / 200 = 50.00%"
        ),
        "{report}"
    );
    assert!(
        report.contains("Wrong-string among pitch-matched string instances: 100 / 200 = 50.00%"),
        "{report}"
    );
    let clipped = run(&["--clip-duration", "1"]);
    assert!(String::from_utf8(clipped.stdout)
        .unwrap()
        .contains("100 / 100 = 100.00%"));
    assert!(!run(&[]).status.success());
    assert!(!run(&["--clip-duration", "0"]).status.success());
    assert!(
        !run(&["--clip-duration", "2", "--report", ir.to_str().unwrap()])
            .status
            .success()
    );
    let second_jams = scratch.0.join("reference2.jams");
    let second_ir = scratch.0.join("prediction2.json");
    std::fs::copy(&jams, &second_jams).unwrap();
    std::fs::copy(&ir, &second_ir).unwrap();
    let manifest = scratch.0.join("manifest.json");
    let entry =
        serde_json::json!({"annotation":"reference.jams", "ir":"prediction.json", "duration":2.0});
    let entry2 = serde_json::json!({"annotation":"reference2.jams", "ir":"prediction2.json", "duration":2.0});
    let write_manifest = |entries: Vec<serde_json::Value>| {
        std::fs::write(&manifest, serde_json::to_vec(&entries).unwrap()).unwrap()
    };
    let run_manifest = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_mtab"))
            .args(["eval", "--tab-manifest"])
            .arg(&manifest)
            .args(extra)
            .output()
            .unwrap()
    };
    write_manifest(vec![entry.clone(), entry2.clone()]);
    let result = run_manifest(&[]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = String::from_utf8(result.stdout).unwrap();
    assert!(
        text.contains("| Multi-pitch frames | 400 | 0 | 0 | 100.00% | 100.00% | 100.00% |"),
        "{text}"
    );
    assert!(
        text.contains("| Tab position frames | 200 | 200 | 200 | 50.00% | 50.00% | 50.00% |"),
        "{text}"
    );
    assert!(
        text.contains(
            "TDR-style ratio (correct tab positions / correct unique pitches): 200 / 400 = 50.00%"
        ),
        "{text}"
    );
    assert!(
        text.contains("Wrong-string among pitch-matched string instances: 200 / 400 = 50.00%"),
        "{text}"
    );
    for protected in [&manifest, &jams, &ir, &second_jams, &second_ir] {
        assert!(!run_manifest(&["--report", protected.to_str().unwrap()])
            .status
            .success());
    }
    assert!(!run_manifest(&["--clip-duration", "2"]).status.success());
    write_manifest(vec![entry.clone(), entry.clone()]);
    assert!(!run_manifest(&[]).status.success());
    write_manifest(vec![
        entry.clone(),
        serde_json::json!({"annotation":"reference2.jams", "ir":"prediction.json", "duration":2.0}),
    ]);
    assert!(!run_manifest(&[]).status.success());
    write_manifest(vec![entry, entry2]);
    std::fs::remove_file(&second_ir).unwrap();
    assert!(!run_manifest(&[]).status.success());
    doc.tracks[0].tuning = Tuning::drop_d();
    save(&doc);
    assert!(!run(&["--clip-duration", "2"]).status.success());
    doc.tracks[0].tuning = Tuning::standard();
    doc.tracks[0].notes[0].fret = 6;
    save(&doc);
    assert!(!run(&["--clip-duration", "2"]).status.success());
}

#[test]
fn audio_manifest_preflights_files_and_protects_inputs() {
    let scratch = Scratch::new();
    let jams = scratch.0.join("reference.jams");
    let wav = scratch.0.join("audio.wav");
    let weights = scratch.0.join("weights.safetensors");
    let manifest = scratch.0.join("audio.json");
    let annotations: Vec<_> = (0..6).map(|s| serde_json::json!({"namespace":"note_midi", "annotation_metadata":{"data_source":s.to_string()}, "data":[]})).collect();
    std::fs::write(
        &jams,
        serde_json::to_vec(&serde_json::json!({"annotations":annotations})).unwrap(),
    )
    .unwrap();
    std::fs::write(&wav, b"not a wav").unwrap();
    std::fs::write(&weights, b"not a checkpoint").unwrap();
    let entry =
        serde_json::json!({"audio":"audio.wav", "annotation":"reference.jams", "duration":5.0});
    let save = |entries: Vec<serde_json::Value>| {
        std::fs::write(&manifest, serde_json::to_vec(&entries).unwrap()).unwrap()
    };
    let run = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_mtab"))
            .args(["eval", "--audio-manifest"])
            .arg(&manifest)
            .arg("--weights")
            .arg(&weights)
            .args(extra)
            .output()
            .unwrap()
    };
    save(vec![entry.clone()]);
    for path in [&manifest, &jams, &wav, &weights] {
        let before = std::fs::read(path).unwrap();
        let output = run(&["--report", path.to_str().unwrap()]);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("overwrite"));
        assert_eq!(std::fs::read(path).unwrap(), before);
    }
    assert!(!run(&["--max-gen-len", "0"]).status.success());
    assert!(!run(&["--max-gen-len", "2001"]).status.success());
    save(vec![entry.clone(), entry]);
    let output = run(&[]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("must not repeat"));
}
