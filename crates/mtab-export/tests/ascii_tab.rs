use mtab_export::{ascii, export, ExportFormat};
use mtab_guitar::{
    BendKind, Capo, GuitarTrack, PitchBend, TabDocument, TabNote, Technique, TempoMap, TrackMeta,
    Tuning,
};

fn note(start: f64, end: f64, string: u8, fret: u8) -> TabNote {
    TabNote {
        start,
        end,
        midi: [40, 45, 50, 55, 59, 64][string as usize] + fret,
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
        note(0.0, 2.5, 0, 3),
        note(0.5, 1.0, 1, 12),
        note(2.0, 2.5, 2, 0),
    ];
    notes[0].technique = vec![Technique::DeadNote, Technique::PalmMute];
    TabDocument {
        meta: TrackMeta {
            title: None,
            tempo_map: TempoMap::constant(120.0),
            time_signatures: vec![],
            frets: 22,
            source_hash: None,
        },
        tracks: vec![GuitarTrack {
            tuning: Tuning::standard(),
            capo: Capo::NONE,
            instrument: None,
            notes,
        }],
        chord_symbols: vec![],
    }
}

#[test]
fn aligned_multidigit_notes_ties_and_bars_are_deterministic() {
    let doc = fixture();
    let (text, plan) = ascii::ascii_tab(&doc).unwrap();
    assert_eq!(text, ascii::ascii_tab(&doc).unwrap().0);
    assert_eq!(plan.max_boundary_shift_seconds, 0.0);
    assert_eq!(text, concat!(
        "mtab ASCII tab | 120 BPM | 4/4 | 1/32 grid\n",
        "Legend: ~ continuation; x dead note; p palm mute; l let ring; v vibrato; h legato (hammer/pull); s<fret>/S<fret> legato/shift slide to fret, S^<fret> picked slide in; b/r/br bend/release/bend-release; @ peak in twelfths\n",
        "Durations: denominator, optional dot; velocities omitted; fret is relative to capo.\n",
        "\nTrack 1 | capo 0 | tuning MIDI [40, 45, 50, 55, 59, 64]\n",
        "Bar 1\n",
        "dur |4---4----2----|\n",
        " 64 |--------------|\n",
        " 59 |--------------|\n",
        " 55 |--------------|\n",
        " 50 |--------------|\n",
        " 45 |----12--------|\n",
        " 40 |px3-p~x3-p~x3-|\n",
        "Bar 2\n",
        "dur |4----4-2-|\n",
        " 64 |---------|\n",
        " 59 |---------|\n",
        " 55 |---------|\n",
        " 50 |0--------|\n",
        " 45 |---------|\n",
        " 40 |p~x3-----|\n"
    ));
}

#[test]
fn let_ring_is_visible_on_attack_and_tied_segments() {
    let mut doc = fixture();
    doc.tracks[0].notes[0].technique.push(Technique::LetRing);
    let (text, _) = ascii::ascii_tab(&doc).unwrap();
    assert!(text.contains("plx3"));
    assert!(text.contains("pl~x3"));
    assert!(text.contains("l let ring"));
}

#[test]
fn bends_and_failed_exports_are_explicit() {
    let mut doc = fixture();
    doc.tracks[0].notes = vec![note(0.0, 0.5, 0, 10)];
    doc.tracks[0].notes[0].bend = Some(PitchBend {
        start: 0.0,
        peak: 0.25,
        depth_cents: 200.0,
        kind: BendKind::BendRelease,
    });
    assert!(ascii::ascii_tab(&doc).unwrap().0.contains("10br2@6"));
    let path = std::env::temp_dir().join(format!("mtab-ascii-{}.txt", std::process::id()));
    export(&doc, &path, ExportFormat::AsciiTab).unwrap();
    let before = std::fs::read(&path).unwrap();
    doc.tracks[0].notes[0].bend.as_mut().unwrap().depth_cents = 150.0;
    assert!(export(&doc, &path, ExportFormat::AsciiTab).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(path).unwrap();
}
