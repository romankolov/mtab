//! Build a tablature document by hand for a guitar the model does not know: a
//! seven-string drop-A tuning with a capo, then export it.
//!
//! The restricted notation exporters accept six or seven ascending strings;
//! eight-string documents (which the IR and MIDI exporter do support) are
//! rejected by GP5/ASCII/MusicXML rather than silently losing a string.
//!
//! No model or audio is needed; this shows the IR, the fingering assignment and
//! the exporters as a library API.
//!
//! ```console
//! cargo run -p mtab-cli --example custom_guitar
//! ```
use std::path::Path;

use mtab_guitar::{Capo, FingeringOptions, GuitarTrack, TabDocument, TempoMap, TrackMeta, Tuning};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Seven strings, low to high, in drop A: A1 E2 A2 D3 G3 B3 E4.
    let tuning = Tuning::new("drop-a 7", vec![33, 40, 45, 50, 55, 59, 64])?;
    // A capo shifts every open string up; frets stay relative to the capo.
    let capo = Capo::new(2)?;
    println!(
        "{}-string {} with capo {}: open pitches {:?}",
        tuning.string_count(),
        tuning.name,
        capo.position(),
        (0..tuning.string_count())
            .map(|s| tuning.pitch_of(s, 0))
            .collect::<Vec<_>>()
    );

    // A low chugging riff: E2 power chord, then a moved shape.
    let mut notes = Vec::new();
    for (index, (start, midi)) in [(0.0, 40), (0.5, 40), (1.0, 43), (1.5, 45)]
        .into_iter()
        .enumerate()
    {
        notes.push(mtab_core::NoteEvent {
            start,
            end: start + 0.5,
            midi,
            velocity: 60 + 20 * (index as u8 % 3),
            instrument: 30,
        });
    }
    let options = FingeringOptions {
        max_fret: 24,
        ..FingeringOptions::default()
    };
    // The assigner clamps to the playable part of this neck and reports notes
    // it cannot reach instead of guessing.
    let assigned = mtab_guitar::assign_notes(&notes, &tuning, capo, options)?;
    for note in &assigned {
        println!(
            "midi {} -> string {} fret {} ({:.2}-{:.2} s, velocity {})",
            note.midi, note.string, note.fret, note.start, note.end, note.velocity
        );
    }

    let document = TabDocument {
        meta: TrackMeta {
            title: Some("custom guitar".into()),
            tempo_map: TempoMap::constant(90.0),
            time_signatures: vec![(0, 4, 4)],
            frets: 24,
            source_hash: None,
        },
        tracks: vec![GuitarTrack {
            tuning,
            capo,
            instrument: Some("distorted_electric_guitar".into()),
            notes: assigned,
        }],
        chord_symbols: Vec::new(),
    };
    mtab_export::export(
        &document,
        Path::new("custom-guitar.gp5"),
        mtab_export::ExportFormat::GuitarPro5,
    )?;
    mtab_export::export(
        &document,
        Path::new("custom-guitar.txt"),
        mtab_export::ExportFormat::AsciiTab,
    )?;
    println!("wrote custom-guitar.gp5 and custom-guitar.txt");
    Ok(())
}
