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
        for line in std::str::from_utf8(&bytes).unwrap().lines() {
            let note: mtab_core::NoteEvent =
                serde_json::from_str(line).expect("invalid JSONL note");
            assert!(note.start.is_finite() && note.end.is_finite() && note.end >= note.start);
            assert!(note.midi <= 127 && (1..=127).contains(&note.velocity));
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
}
