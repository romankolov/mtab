//! Long synthetic tempo map for an independent playback-time regression.
use mtab_export::midi::write_midi;
use mtab_guitar::{
    BendKind, Capo, GuitarTrack, PitchBend, TabDocument, TabNote, TempoMap, TrackMeta, Tuning,
};
use std::path::Path;

fn main() -> mtab_core::Result<()> {
    let path = std::env::args().nth(1).expect("output MIDI path");
    let beats: Vec<f64> = (0..=10_000).map(|i| f64::from(i) * 0.500_000_49).collect();
    let tempo_map = TempoMap::from_quarter_beat_times(&beats)?;
    let start = beats[9999] + 0.25;
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
            notes: vec![TabNote {
                start,
                end: start + 0.125,
                midi: 40,
                string: 0,
                fret: 0,
                velocity: 90,
                bend: Some(PitchBend {
                    start,
                    peak: start + 0.0625,
                    depth_cents: 100.0,
                    kind: BendKind::BendRelease,
                }),
                technique: vec![],
                tie_from_prev: false,
            }],
        }],
        chord_symbols: vec![],
    };
    write_midi(&doc, Path::new(&path))
}
