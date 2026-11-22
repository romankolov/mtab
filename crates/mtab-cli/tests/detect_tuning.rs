//! End-to-end, model-free smoke test for experimental audio tuning detection.
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn standard_open_notes_are_ambiguous_with_extended_tunings() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("mtab-detect-{}-{id}.wav", std::process::id()));
    let mut pcm = Vec::new();
    for midi in [40u8, 45, 50, 55, 59, 64] {
        let hz = 440.0 * 2.0f64.powf((f64::from(midi) - 69.0) / 12.0);
        for i in 0..8000 {
            let sample = if i < 7200 {
                (10000.0 * (2.0 * std::f64::consts::PI * hz * i as f64 / 16000.0).sin()).round()
                    as i16
            } else {
                0
            };
            pcm.extend_from_slice(&sample.to_le_bytes());
        }
    }
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16000u32.to_le_bytes());
    wav.extend_from_slice(&32000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    wav.extend_from_slice(&pcm);
    std::fs::write(&path, wav).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_mtab"))
        .arg("detect-tuning")
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = String::from_utf8(result.stdout).unwrap();
    assert!(
        output.contains("Experimental monophonic tuning estimate"),
        "{output}"
    );
    assert!(output.contains("Best hypothesis:"), "{output}");
    assert!(
        output.contains("Insufficient or ambiguous evidence"),
        "{output}"
    );
}
