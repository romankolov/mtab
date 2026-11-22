//! Opt-in independent GP5 validation. CI installs PyGuitarPro and runs this.
use mtab_export::gp5;
use mtab_guitar::{Capo, GuitarTrack, TabDocument, TabNote, TempoMap, TrackMeta, Tuning};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn note(start: f64, end: f64, midi: u8, string: u8, fret: u8, velocity: u8) -> TabNote {
    TabNote {
        start,
        end,
        midi,
        string,
        fret,
        velocity,
        bend: None,
        technique: vec![],
        tie_from_prev: false,
    }
}

fn fixture() -> TabDocument {
    TabDocument {
        meta: TrackMeta {
            title: Some("GP5 roundtrip".into()),
            tempo_map: TempoMap::constant(120.0),
            time_signatures: vec![(0, 4, 4)],
            frets: 22,
            source_hash: None,
        },
        tracks: vec![
            GuitarTrack {
                tuning: Tuning::standard(),
                capo: Capo::new(2).unwrap(),
                instrument: Some("acoustic_guitar".into()),
                notes: vec![
                    note(0.0, 2.5, 45, 0, 3, 80),
                    note(0.5, 1.0, 49, 1, 2, 112),
                    note(2.0, 2.5, 52, 2, 0, 127),
                ],
            },
            GuitarTrack {
                tuning: Tuning::seven_string(),
                capo: Capo::NONE,
                instrument: Some("distorted_electric_guitar".into()),
                notes: vec![note(0.5, 2.5, 36, 0, 1, 40)],
            },
        ],
        chord_symbols: vec![],
    }
}

#[test]
fn encoded_gp5_is_deterministic_and_invalid_input_does_not_write() {
    let doc = fixture();
    let (first, plan) = gp5::gp5_bytes(&doc).unwrap();
    let (second, _) = gp5::gp5_bytes(&doc).unwrap();
    assert_eq!(first, second);
    assert_eq!(plan.max_boundary_shift_seconds, 0.0);
    assert!(first.starts_with(b"\x18FICHIER GUITAR PRO v5.00"));
    let mut invalid = doc;
    invalid.tracks[0].notes[0]
        .technique
        .push(mtab_guitar::Technique::PalmMute);
    let tmp = std::env::temp_dir().join(format!("mtab-gp5-invalid-{}", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    assert!(gp5::write_gp5(&invalid, &tmp).is_err());
    assert!(!tmp.exists());
}

#[test]
#[ignore = "requires PyGuitarPro; run in the independent-parser CI job"]
fn pyguitarpro_reads_tracks_ties_and_dynamics() {
    let python = std::env::var_os("MTAB_PYGUITARPRO_PYTHON")
        .expect("set MTAB_PYGUITARPRO_PYTHON to a Python with PyGuitarPro 0.10.1");
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        Scratch(std::env::temp_dir().join(format!("mtab-gp5-test-{}-{n}", std::process::id())));
    std::fs::create_dir(&dir.0).unwrap();
    let file = dir.0.join("fixture.gp5");
    gp5::write_gp5(&fixture(), &file).unwrap();
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/check_gp5_pyguitarpro.py");
    let output = Command::new(python).arg(script).arg(file).output().unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
