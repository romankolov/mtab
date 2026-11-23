//! Synthetic variable-tempo fixture for independent SMF validation.
use mtab_export::midi::write_midi;
use mtab_guitar::{Capo, GuitarTrack, TabDocument, TabNote, TempoMap, TrackMeta, Tuning};
use std::path::Path;

fn main() -> mtab_core::Result<()> {
    let path = std::env::args().nth(1).expect("output MIDI path");
    let tempo_map = TempoMap::from_quarter_beat_times(&[0.0, 0.5, 1.1, 1.8, 2.3])?;
    let notes = [
        (0.25, 0.5, 40, 0),
        (0.8, 1.1, 45, 1),
        (1.45, 1.8, 50, 2),
        (2.05, 2.3, 55, 3),
    ]
    .into_iter()
    .map(|(start, end, midi, string)| TabNote {
        start,
        end,
        midi,
        string,
        fret: 0,
        velocity: 90,
        bend: None,
        technique: vec![],
        tie_from_prev: false,
    })
    .collect();
    let doc = TabDocument {
        meta: TrackMeta {
            title: None,
            tempo_map,
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
