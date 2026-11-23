use mtab_export::{export, musicxml, ExportFormat};
use mtab_guitar::{
    BendKind, Capo, GuitarTrack, PitchBend, TabDocument, TabNote, Technique, TempoMap, TrackMeta,
    Tuning,
};
use std::path::PathBuf;

fn note(start: f64, end: f64, midi: u8, string: u8, fret: u8) -> TabNote {
    TabNote {
        start,
        end,
        midi,
        string,
        fret,
        velocity: 100,
        bend: None,
        technique: vec![],
        tie_from_prev: false,
    }
}
fn fixture() -> TabDocument {
    let mut notes = vec![
        note(0.0, 2.5, 45, 0, 3),
        note(0.5, 1.0, 49, 1, 2),
        note(2.0, 2.5, 52, 2, 0),
    ];
    notes[1].technique.push(Technique::PalmMute);
    notes[2].technique.push(Technique::DeadNote);
    TabDocument {
        meta: TrackMeta {
            title: Some("Café & <guitar> 'one' \"two\"".into()),
            tempo_map: TempoMap::constant(120.0),
            time_signatures: vec![(0, 4, 4)],
            frets: 22,
            source_hash: None,
        },
        tracks: vec![
            GuitarTrack {
                tuning: Tuning::standard(),
                capo: Capo::new(2).unwrap(),
                notes,
                instrument: None,
            },
            GuitarTrack {
                tuning: Tuning::seven_string(),
                capo: Capo::NONE,
                notes: vec![note(0.5, 1.0, 36, 0, 1)],
                instrument: None,
            },
        ],
        chord_symbols: vec![],
    }
}

#[test]
fn xml_is_deterministic_and_rejects_unsupported_without_overwrite() {
    let mut doc = fixture();
    let (text, plan) = musicxml::musicxml(&doc).unwrap();
    assert_eq!(text, musicxml::musicxml(&doc).unwrap().0);
    assert_eq!(plan.max_boundary_shift_seconds, 0.0);
    assert!(text.contains("Café &amp; &lt;guitar&gt; &apos;one&apos; &quot;two&quot;"));
    assert!(text.contains("<staff-lines>7</staff-lines>"));
    let path = std::env::temp_dir().join(format!("mtab-musicxml-{}.xml", std::process::id()));
    export(&doc, &path, ExportFormat::MusicXml).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), text.as_bytes());
    doc.tracks[0].notes[1].bend = Some(PitchBend {
        start: 0.5,
        peak: 0.75,
        depth_cents: 100.0,
        kind: BendKind::Bend,
    });
    assert!(export(&doc, &path, ExportFormat::MusicXml).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), text.as_bytes());
    doc.tracks[0].notes[1].bend = None;
    doc.meta.title = Some("forbidden\u{0}".into());
    assert!(musicxml::musicxml(&doc).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), text.as_bytes());
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "requires lxml and the bundled MusicXML 3.1 schema"]
fn independent_schema_and_semantic_validation() {
    let (text, _) = musicxml::musicxml(&fixture()).unwrap();
    let p = std::env::temp_dir().join(format!("mtab-xml-check-{}.musicxml", std::process::id()));
    std::fs::write(&p, text).unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/check_musicxml.py");
    let python =
        std::env::var_os("MTAB_XML_PYTHON").expect("set MTAB_XML_PYTHON to a Python with lxml");
    let output = std::process::Command::new(python)
        .arg(script)
        .arg(&p)
        .output()
        .unwrap();
    if let Some(dir) = std::env::var_os("MTAB_XML_FIXTURE_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(&p, PathBuf::from(dir).join("fixture.musicxml")).unwrap();
    }
    std::fs::remove_file(p).unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "requires lxml and the bundled MusicXML 3.1 schema"]
fn independent_vibrato_schema_and_semantics() {
    let mut doc = fixture();
    doc.tracks[0].notes[0].technique.push(Technique::Vibrato {
        rate_hz: 5.5,
        depth_cents: 35.0,
    });
    doc.tracks[0].notes[2].technique.push(Technique::Vibrato {
        rate_hz: 6.0,
        depth_cents: 50.0,
    });
    let (text, _) = musicxml::musicxml(&doc).unwrap();
    // Each note is written on both the standard and tab staff, so the count is
    // not asserted here; the checker pins exact per-staff positions.
    assert!(text.contains("<wavy-line type=\"start\""));
    let path =
        std::env::temp_dir().join(format!("mtab-xml-vibrato-{}.musicxml", std::process::id()));
    std::fs::write(&path, text).unwrap();
    let python =
        std::env::var_os("MTAB_XML_PYTHON").expect("set MTAB_XML_PYTHON to a Python with lxml");
    let output = std::process::Command::new(python)
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/check_musicxml_vibrato.py"),
        )
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "requires lxml and the bundled MusicXML 3.1 schema"]
fn independent_let_ring_schema_and_semantics() {
    let mut doc = fixture();
    doc.tracks[0].notes[0].technique.push(Technique::LetRing);
    doc.tracks[0].notes[1].technique.push(Technique::LetRing);
    let (text, _) = musicxml::musicxml(&doc).unwrap();
    let path =
        std::env::temp_dir().join(format!("mtab-xml-let-ring-{}.musicxml", std::process::id()));
    std::fs::write(&path, text).unwrap();
    let python =
        std::env::var_os("MTAB_XML_PYTHON").expect("set MTAB_XML_PYTHON to a Python with lxml");
    let output = std::process::Command::new(python)
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tools/check_musicxml_let_ring.py"),
        )
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
