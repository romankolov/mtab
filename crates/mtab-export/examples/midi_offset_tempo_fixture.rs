//! Synthetic offset-beat variable-tempo fixture for independent SMF validation.
use mtab_export::midi::write_midi;
use mtab_guitar::{Capo, GuitarTrack, TabDocument, TabNote, TempoMap, TrackMeta, Tuning};
use std::path::Path;

fn main() -> mtab_core::Result<()> {
    let path = std::env::args().nth(1).expect("output MIDI path");
    let tempo_map = TempoMap::from_offset_quarter_beat_times(&[0.25, 0.75, 1.35, 2.05])?;
    let notes = [
        (0.125, 0.25, 40, 0),
        (0.5, 0.75, 45, 1),
        (1.05, 1.35, 50, 2),
        (1.7, 2.05, 55, 3),
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
