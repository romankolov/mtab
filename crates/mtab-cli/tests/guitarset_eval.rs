//! Synthetic JAMS-shaped integration tests; no dataset or weights required.
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("mtab-eval-{}-{id}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
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
