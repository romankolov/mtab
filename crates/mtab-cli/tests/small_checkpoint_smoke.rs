//! Opt-in end-to-end smoke test for a user-provided small checkpoint.
//!
//! Set MTAB_SMALL_WEIGHTS to a licensed MuScriptor small safetensors file and run:
//! `cargo test -p mtab-cli --release --test small_checkpoint_smoke -- --ignored --nocapture`
//! This is a short execution/determinism check, not a transcription-accuracy or
//! upstream token-parity test. It uses synthetic audio and does not fetch weights.

use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn synthetic_wav(path: &std::path::Path) {
    let sample_rate = 16_000u32;
    let raw = include_bytes!("../../mtab-core/tests/fixtures/model/guitar_plucks.f32");
    let samples = (raw.len() / 4) as u32;
    let mut wav = Vec::with_capacity(44 + 2 * samples as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + samples * 2).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(samples * 2).to_le_bytes());
    for bytes in raw.chunks_exact(4) {
        let sample = f32::from_le_bytes(bytes.try_into().unwrap());
        let pcm = (sample * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
        wav.extend_from_slice(&pcm.to_le_bytes());
    }
    std::fs::write(path, wav).unwrap();
}

#[test]
#[ignore = "requires user-provided licensed small checkpoint; expensive"]
fn real_small_cli_smoke_and_determinism() {
    let weights = std::env::var_os("MTAB_SMALL_WEIGHTS")
        .expect("set MTAB_SMALL_WEIGHTS to a licensed small checkpoint");
    assert!(
        std::path::Path::new(&weights).is_file(),
        "MTAB_SMALL_WEIGHTS must be an existing local file"
    );
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        Scratch(std::env::temp_dir().join(format!("mtab-small-smoke-{}-{id}", std::process::id())));
    std::fs::create_dir(&dir.0).unwrap();
    let input = dir.0.join("tone.wav");
    synthetic_wav(&input);
    let max_tokens = std::env::var("MTAB_SMOKE_MAX_GEN_LEN").unwrap_or_else(|_| "80".into());
    let mut outputs = Vec::new();
    for run in 0..2 {
        let output = dir.0.join(format!("notes-{run}.jsonl"));
        let result = Command::new(env!("CARGO_BIN_EXE_mtab"))
            .arg("transcribe")
            .arg(&input)
            .args(["--size", "small", "--weights"])
            .arg(&weights)
            .args([
                "--instruments",
                "guitar",
                "--format",
                "jsonl",
                "--max-gen-len",
                &max_tokens,
                "--seed",
                "0",
                "-o",
            ])
            .arg(&output)
            .output()
            .expect("launch mtab CLI");
        assert!(
            result.status.success(),
            "CLI failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            result.stdout.is_empty(),
            "file output must not contain stdout records"
        );
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr.contains("transcribing chunks: 1/1"),
            "missing completion progress: {stderr}"
        );
        let bytes = std::fs::read(&output).expect("CLI must create output");
        if max_tokens.parse::<usize>().unwrap() >= 80 {
            assert!(
                !stderr.contains("without EOS"),
                "expected the small model to terminate on the synthetic guitar fixture"
            );
            assert!(
                !bytes.is_empty(),
                "expected note records for the synthetic guitar fixture"
            );
        }
        let mut groups: std::collections::BTreeMap<u16, Vec<mtab_core::NoteEvent>> =
            std::collections::BTreeMap::new();
        for line in std::str::from_utf8(&bytes).unwrap().lines() {
            let note: mtab_core::NoteEvent =
                serde_json::from_str(line).expect("invalid JSONL note");
            assert!(note.start.is_finite() && note.end.is_finite() && note.end >= note.start);
            assert!(note.midi <= 127 && (1..=127).contains(&note.velocity));
            groups.entry(note.instrument).or_default().push(note);
        }
        // Exercise the preliminary M5 library solver on the real model's
        // output, while keeping tracks independent. This is not an accuracy
        // benchmark and is independent of the CLI MIDI export below.
        if max_tokens.parse::<usize>().unwrap() >= 80 {
            for notes in groups.values() {
                let tuning = mtab_guitar::Tuning::standard();
                let tab = mtab_guitar::assign_notes(
                    notes,
                    &tuning,
                    mtab_guitar::Capo::NONE,
                    mtab_guitar::FingeringOptions::default(),
                )
                .expect("synthetic fixture notes should be playable in standard tuning");
                assert_eq!(tab.len(), notes.len());
                for note in &tab {
                    assert_eq!(tuning.strings[note.string as usize] + note.fret, note.midi);
                }
            }
        }
        outputs.push(bytes);
    }
    assert_eq!(
        outputs[0], outputs[1],
        "identical runs must produce identical JSONL bytes"
    );
    eprintln!(
        "small-model CLI produced {} deterministic note records",
        std::str::from_utf8(&outputs[0]).unwrap().lines().count()
    );
    // A third independent inference executes the new CLI MIDI pipeline. Parse
    // the SMF and compare every pitch, estimated velocity and absolute note
    // tick against the JSONL plus the same audio-envelope dynamics estimator
    // (120 BPM, 480 ticks/quarter = 960 ticks/s).
    let midi_path = dir.0.join("notes.mid");
    let ir_path = dir.0.join("notes.ir.json");
    let midi_run = Command::new(env!("CARGO_BIN_EXE_mtab"))
        .arg("transcribe")
        .arg(&input)
        .args(["--size", "small", "--weights"])
        .arg(&weights)
        .args([
            "--instruments",
            "guitar",
            "--max-gen-len",
            &max_tokens,
            "--seed",
            "0",
            "--dynamics",
            "--emit-ir",
        ])
        .arg(&ir_path)
        .arg("-o")
        .arg(&midi_path)
        .output()
        .expect("launch MIDI CLI");
    assert!(
        midi_run.status.success(),
        "MIDI CLI failed: {}",
        String::from_utf8_lossy(&midi_run.stderr)
    );
    let bytes = std::fs::read(&midi_path).unwrap();
    let smf = midly::Smf::parse(&bytes).expect("valid SMF");
    assert_eq!(smf.header.format, midly::Format::Parallel);
    assert_eq!(smf.header.timing, midly::Timing::Metrical(480.into()));
    let mut actual = Vec::new();
    for track in &smf.tracks {
        let mut tick = 0u64;
        for event in track {
            tick += u64::from(event.delta.as_int());
            if let midly::TrackEventKind::Midi { message, .. } = event.kind {
                match message {
                    midly::MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                        actual.push((tick, key.as_int(), vel.as_int(), true))
                    }
                    midly::MidiMessage::NoteOff { key, .. } => {
                        actual.push((tick, key.as_int(), 0, false))
                    }
                    _ => {}
                }
            }
        }
    }
    let mut expected = Vec::new();
    let mut raw_notes: Vec<mtab_core::NoteEvent> = std::str::from_utf8(&outputs[0])
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let original = raw_notes.clone();
    let audio = mtab_audio::load_model_input(&input).unwrap();
    let report = mtab_audio::dynamics::estimate_velocities(&audio, &mut raw_notes).unwrap();
    if max_tokens.parse::<usize>().unwrap() >= 80 {
        assert!(report.estimated > 0);
        assert!(original
            .iter()
            .zip(&raw_notes)
            .any(|(old, new)| old.velocity != new.velocity));
    }
    for n in &raw_notes {
        let start = (n.start * 960.0).round() as u64;
        let end = ((n.end * 960.0).round() as u64).max(start + 1);
        expected.push((start, n.midi, n.velocity, true));
        expected.push((end, n.midi, 0, false));
    }
    actual.sort_unstable();
    expected.sort_unstable();
    assert_eq!(
        actual, expected,
        "SMF note events must match dynamics-adjusted model output"
    );
    let doc: mtab_guitar::TabDocument =
        serde_json::from_slice(&std::fs::read(ir_path).unwrap()).unwrap();
    assert_eq!(
        doc.tracks.iter().map(|t| t.notes.len()).sum::<usize>(),
        expected.len() / 2
    );
    eprintln!(
        "MIDI parsed successfully with {} matching notes ({} velocity estimates)",
        expected.len() / 2,
        report.estimated
    );
}
