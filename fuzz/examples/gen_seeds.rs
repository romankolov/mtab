//! Development-only: write the tracked ir_export fuzz seeds from the real IR
//! types, so a seed can never be a hand-written schema guess.
use mtab_guitar::{
    Capo, ChordSymbol, GuitarTrack, PitchBend, TabDocument, TabNote, Technique, TempoMap,
    TrackMeta, Tuning,
};

fn note(start: f64, end: f64, midi: u8, string: u8, fret: u8) -> TabNote {
    TabNote {
        start,
        end,
        midi,
        string,
        fret,
        velocity: 90,
        bend: None,
        technique: Vec::new(),
        tie_from_prev: false,
    }
}

fn document(notes: Vec<TabNote>) -> TabDocument {
    TabDocument {
        meta: TrackMeta {
            title: Some("seed".into()),
            tempo_map: TempoMap::constant(120.0),
            time_signatures: vec![(0, 4, 4)],
            frets: 22,
            source_hash: None,
        },
        tracks: vec![GuitarTrack {
            tuning: Tuning::standard(),
            capo: Capo::NONE,
            instrument: Some("clean_electric_guitar".into()),
            notes,
        }],
        chord_symbols: Vec::new(),
    }
}

fn main() {
    let dir = std::path::Path::new("seeds/ir_export");
    std::fs::create_dir_all(dir).unwrap();
    let write = |name: &str, doc: &TabDocument| {
        std::fs::write(dir.join(name), doc.to_json().unwrap()).unwrap();
    };

    write("plain.json", &document(vec![note(0.0, 0.5, 45, 0, 5), note(0.5, 1.0, 47, 0, 7)]));

    let mut techniques = document(vec![
        note(0.0, 0.5, 45, 0, 5),
        note(0.5, 1.0, 48, 0, 8),
        note(2.0, 2.5, 45, 0, 5),
    ]);
    techniques.tracks[0].notes[0].technique = vec![Technique::HammerOn];
    techniques.tracks[0].notes[1].technique = vec![Technique::PullOff];
    techniques.tracks[0].notes[2].technique = vec![
        Technique::PalmMute,
        Technique::LetRing,
        Technique::Vibrato {
            rate_hz: 5.5,
            depth_cents: 40.0,
        },
    ];
    write("techniques.json", &techniques);

    let mut chords = document(vec![
        note(0.0, 1.5, 45, 0, 5),
        note(1.5, 3.0, 45, 0, 5),
    ]);
    chords.chord_symbols = vec![
        ChordSymbol {
            start: 0.0,
            end: 1.5,
            label: "Em7".into(),
        },
        ChordSymbol {
            start: 1.5,
            end: 3.0,
            label: "C".into(),
        },
    ];
    write("chords.json", &chords);

    let mut bend = document(vec![note(0.0, 1.0, 45, 0, 5)]);
    bend.tracks[0].notes[0].bend = Some(PitchBend {
        start: 0.0,
        peak: 0.5,
        depth_cents: 100.0,
        kind: mtab_guitar::BendKind::Bend,
    });
    write("bend.json", &bend);
    println!("wrote 4 seeds to {}", dir.display());
}
