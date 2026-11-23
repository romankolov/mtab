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

// Opt-in for reproducing external notation-application round trips.
fn keep_fixture(path: &std::path::Path) {
    if let Some(dir) = std::env::var_os("MTAB_GP5_FIXTURE_DIR") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(path, dir.join(path.file_name().unwrap())).unwrap();
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
    let path = std::env::temp_dir().join(format!("mtab-gp5-dispatch-{}.gp5", std::process::id()));
    mtab_export::export(&doc, &path, mtab_export::ExportFormat::GuitarPro5).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), first);
    std::fs::remove_file(path).unwrap();
    let mut invalid = doc;
    invalid.tracks[0].notes[0]
        .technique
        .push(mtab_guitar::Technique::HammerOn);
    let tmp = std::env::temp_dir().join(format!("mtab-gp5-invalid-{}", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    assert!(gp5::write_gp5(&invalid, &tmp).is_err());
    assert!(!tmp.exists());
}

#[test]
fn unsupported_bend_cannot_overwrite_existing_file() {
    let path =
        std::env::temp_dir().join(format!("mtab-gp5-bend-invalid-{}.gp5", std::process::id()));
    let mut doc = fixture();
    std::fs::write(&path, b"original").unwrap();
    doc.tracks[0].notes[0].bend = Some(mtab_guitar::PitchBend {
        start: 0.0,
        peak: 1.25,
        depth_cents: 100.0,
        kind: mtab_guitar::BendKind::Bend,
    });
    let error = gp5::write_gp5(&doc, &path).unwrap_err().to_string();
    assert!(error.contains("bent note must fit one"), "{error}");
    assert_eq!(std::fs::read(&path).unwrap(), b"original");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn gp5_rejects_unrepresentable_title_without_overwriting() {
    let path = std::env::temp_dir().join(format!("mtab-gp5-title-{}.gp5", std::process::id()));
    let mut doc = fixture();
    std::fs::write(&path, b"original").unwrap();
    for title in [
        "日本語".to_owned(),
        "café\nscore".to_owned(),
        "é".repeat(256),
    ] {
        doc.meta.title = Some(title);
        let error = gp5::write_gp5(&doc, &path).unwrap_err().to_string();
        assert!(error.contains("Windows-1252"), "{error}");
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
    }
    doc.meta.title = Some("é".repeat(255));
    assert!(gp5::gp5_bytes(&doc).is_ok());
    std::fs::remove_file(path).unwrap();
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
    let output = Command::new(&python)
        .arg(&script)
        .arg(file)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let mut marked = fixture();
    marked.tracks[0].notes[0].technique = vec![
        mtab_guitar::Technique::PalmMute,
        mtab_guitar::Technique::DeadNote,
    ];
    marked.tracks[0].notes[1].technique = vec![mtab_guitar::Technique::PalmMute];
    marked.tracks[0].notes[2].technique = vec![mtab_guitar::Technique::DeadNote];
    let marked_file = dir.0.join("techniques.gp5");
    gp5::write_gp5(&marked, &marked_file).unwrap();
    keep_fixture(&marked_file);
    let technique_check =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/check_gp5_techniques.py");
    let output = Command::new(&python)
        .arg(technique_check)
        .arg(marked_file)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "techniques parser failure: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut ringing = fixture();
    ringing.tracks[0].notes[0]
        .technique
        .push(mtab_guitar::Technique::LetRing);
    ringing.tracks[0].notes[1]
        .technique
        .push(mtab_guitar::Technique::LetRing);
    let ringing_file = dir.0.join("let-ring.gp5");
    gp5::write_gp5(&ringing, &ringing_file).unwrap();
    let output = Command::new(&python)
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/check_gp5_let_ring.py"))
        .arg(&ringing_file)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "let-ring parser failure: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mut bent = fixture();
    bent.tracks.truncate(1);
    let mut notes = vec![
        note(0.0, 0.5, 42, 0, 0, 80),
        note(0.5, 1.0, 42, 0, 0, 80),
        note(1.0, 1.5, 42, 0, 0, 80),
    ];
    for (index, (kind, peak_position, depth)) in [
        (mtab_guitar::BendKind::Bend, 4.0, 100.0),
        (mtab_guitar::BendKind::Release, 6.0, 200.0),
        (mtab_guitar::BendKind::BendRelease, 8.0, 100.0),
    ]
    .into_iter()
    .enumerate()
    {
        let n = &mut notes[index];
        n.bend = Some(mtab_guitar::PitchBend {
            start: n.start,
            peak: n.start + (n.end - n.start) * peak_position / 12.0,
            depth_cents: depth,
            kind,
        });
    }
    notes[0].technique.push(mtab_guitar::Technique::PalmMute);
    bent.tracks[0].notes = notes;
    let bent_file = dir.0.join("bends.gp5");
    let (bent_bytes, _) = gp5::gp5_bytes(&bent).unwrap();
    assert_eq!(bent_bytes, gp5::gp5_bytes(&bent).unwrap().0);
    std::fs::write(&bent_file, bent_bytes).unwrap();
    keep_fixture(&bent_file);
    let bend_check =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/check_gp5_bends.py");
    let output = Command::new(&python)
        .arg(bend_check)
        .arg(bent_file)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "bend parser failure: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let header_check =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/check_gp5_bar_headers.py");
    for (index, title) in ["Café — déjà vu".to_owned(), "é".repeat(255)]
        .iter()
        .enumerate()
    {
        let mut doc = fixture();
        doc.meta.title = Some(title.clone());
        let file = dir.0.join(format!("title-{index}.gp5"));
        gp5::write_gp5(&doc, &file).unwrap();
        let output = Command::new(&python)
            .arg(&script)
            .arg(file)
            .arg(title)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "title parser failure: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    for bars in 1..=4 {
        let mut doc = fixture();
        doc.tracks.truncate(1);
        doc.tracks[0].notes.truncate(1);
        doc.tracks[0].notes[0].end = bars as f64 * 2.0;
        let file = dir.0.join(format!("bars-{bars}.gp5"));
        gp5::write_gp5(&doc, &file).unwrap();
        let output = Command::new(&python)
            .arg(&header_check)
            .arg(file)
            .arg(bars.to_string())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{bars} bar parser failure: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
