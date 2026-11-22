//! Independent-parser test fixture, not application inference.
use mtab_export::midi::write_midi;
use mtab_guitar::{
    BendKind, Capo, GuitarTrack, PitchBend, TabDocument, TabNote, TempoMap, TrackMeta, Tuning,
};
use std::path::Path;

fn main() {
    let output = std::env::args().nth(1).expect("output path");
    let doc = TabDocument {
        meta: TrackMeta {
            title: None,
            tempo_map: TempoMap::constant(120.0),
            time_signatures: vec![(0, 4, 4)],
            frets: 22,
            source_hash: None,
        },
        tracks: vec![GuitarTrack {
            instrument: Some("acoustic_guitar".into()),
            tuning: Tuning::standard(),
            capo: Capo::NONE,
            notes: vec![TabNote {
                start: 0.0,
                end: 1.0,
                midi: 60,
                string: 0,
                fret: 0,
                velocity: 92,
                bend: Some(PitchBend {
                    start: 0.0,
                    peak: 0.5,
                    depth_cents: 100.0,
                    kind: BendKind::BendRelease,
                }),
                technique: vec![],
                tie_from_prev: false,
            }],
        }],
        chord_symbols: vec![],
    };
    write_midi(&doc, Path::new(&output)).unwrap();
}
