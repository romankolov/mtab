//! Regression tests for malformed containers found by the libFuzzer
//! `audio_decode` target. Decoding must return an error, never panic or abort.

use std::io::Write;

use mtab_audio::decode::load;

fn temp_wav(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mtab-decode-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::File::create(&path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
    path
}

/// A 16-bit mono PCM WAVE whose `fmt ` chunk declares a zero sample rate.
fn zero_sample_rate_wav() -> Vec<u8> {
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1u16.to_le_bytes()); // PCM
    fmt.extend_from_slice(&1u16.to_le_bytes()); // mono
    fmt.extend_from_slice(&0u32.to_le_bytes()); // sample rate 0
    fmt.extend_from_slice(&0u32.to_le_bytes()); // byte rate 0
    fmt.extend_from_slice(&2u16.to_le_bytes()); // block align
    fmt.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    let data = [0u8; 8];
    let mut chunks = Vec::new();
    chunks.extend_from_slice(b"fmt ");
    chunks.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    chunks.extend_from_slice(&fmt);
    chunks.extend_from_slice(b"data");
    chunks.extend_from_slice(&(data.len() as u32).to_le_bytes());
    chunks.extend_from_slice(&data);
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(4 + chunks.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(&chunks);
    out
}

#[test]
fn zero_sample_rate_is_an_error_not_a_panic() {
    let path = temp_wav("zero-sample-rate.wav", &zero_sample_rate_wav());
    assert!(
        load(&path).is_err(),
        "a zero sample rate must be rejected before any time base is built"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn truncated_header_is_an_error() {
    let path = temp_wav("truncated.wav", b"RIFF\x10\x00\x00");
    assert!(load(&path).is_err());
    let _ = std::fs::remove_file(path);
}
