//! Synthetic MIDI velocity fixture for independent playback tests.
use mtab_export::midi::write_midi;
use mtab_guitar::{Capo, GuitarTrack, TabDocument, TabNote, TempoMap, TrackMeta, Tuning};
use std::path::Path;

fn main() -> mtab_core::Result<()> {
    let path = std::env::args().nth(1).expect("output MIDI path");
    let notes = [(0.2, 0.5, 40), (0.8, 1.1, 80), (1.4, 1.7, 120)]
        .into_iter()
        .map(|(start, end, velocity)| TabNote {
            start,
            end,
            midi: 45,
            string: 1,
            fret: 0,
            velocity,
            bend: None,
            technique: vec![],
            tie_from_prev: false,
        })
        .collect();
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
            notes,
        }],
        chord_symbols: vec![],
    };
    write_midi(&doc, Path::new(&path))
}
