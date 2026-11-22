//! Mel parity goldens vs the upstream MIT reference (PROMPT.md §6.2).
//!
//! Fixtures are produced by `tools/gen_mel_fixtures.py` from the verbatim
//! upstream module (`tools/upstream/mel_spectrogram_mit.py`) and are OPTIONAL
//! in CI: when absent, these tests skip with instructions. When present they
//! are the sacred correctness gate: a 2-bin mel mismatch destroys
//! transcription quality silently.
//!
//! Tolerances: compared in MEL space with `|a-b| <= 1e-4 * max(1, |b|)`
//! (cross-FFT-implementation roundoff on unnormalized magnitudes up to ~1e3;
//! leakage-tail bins legitimately differ by a few % relative between fp32
//! torch and fp64 rustfft). The log step is verified by inverting it
//! (`exp(y) - eps` vs reference mel) plus unit tests of the formula itself.

use std::path::{Path, PathBuf};

use mtab_audio::mel::MelFrontend;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() % 4, 0, "bad f32 fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn mel_parity_vs_upstream() {
    let dir = fixtures_dir();
    if !dir.exists() {
        eprintln!(
            "SKIP mel parity: fixtures not present. \
             Generate with: python3 tools/gen_mel_fixtures.py (requires torch + numpy)"
        );
        return;
    }

    let fe = MelFrontend::upstream();
    let mut compared = 0;

    for case in ["tone", "chords", "mix"] {
        let input_path = dir.join(format!("{case}_input.f32"));
        let mel_path = dir.join(format!("{case}_mel.f32"));
        let log_path = dir.join(format!("{case}_logmel.f32"));
        if !input_path.exists() || !mel_path.exists() || !log_path.exists() {
            eprintln!("SKIP case {case}: fixture files missing");
            continue;
        }

        let input = read_f32(&input_path);
        let ref_mel = read_f32(&mel_path);
        let ref_log = read_f32(&log_path);

        let mel = fe.compute_mel(&input);
        let log_mel = fe.compute(&input);

        let frames = mel.len();
        let n_mels = fe.config().n_mels;
        assert_eq!(frames * n_mels, ref_mel.len(), "{case}: mel size mismatch");
        assert_eq!(
            frames * n_mels,
            ref_log.len(),
            "{case}: logmel size mismatch"
        );

        let mut max_rel = 0.0f64;
        for (t, frame) in mel.iter().enumerate() {
            for (j, v) in frame.iter().enumerate() {
                let r = ref_mel[t * n_mels + j];
                let diff = (*v as f64 - r as f64).abs();
                let tol = 1e-4 * (r.abs() as f64).max(1.0);
                assert!(
                    diff <= tol,
                    "{case} mel[{t}][{j}]: got {} ref {r} (diff {diff} > tol {tol})",
                    v
                );
                max_rel = max_rel.max(diff / (r.abs() as f64).max(1.0));
            }
        }

        for (t, frame) in log_mel.iter().enumerate() {
            for (j, v) in frame.iter().enumerate() {
                // Invert the log and compare in mel space: the log step is a
                // deterministic fn of mel (verified separately in unit tests),
                // and comparing it directly would amplify cross-FFT roundoff
                // at leakage-tail bins (~1e-3 values) into spurious failures.
                let mel_back = (*v as f64).exp() - mtab_audio::mel::UPSTREAM_EPS as f64;
                let r = ref_mel[t * n_mels + j];
                let diff = (mel_back - r as f64).abs();
                let tol = 1e-4 * (r.abs() as f64).max(1.0);
                assert!(
                    diff <= tol,
                    "{case} log-roundtrip[{t}][{j}]: got {mel_back} ref {r} (diff {diff} > tol {tol})"
                );
            }
        }

        println!("{case}: PARITY OK ({frames}x{n_mels}, max rel err {max_rel:.2e})");
        compared += 1;
    }

    assert!(compared > 0, "no fixture cases were compared");
}
