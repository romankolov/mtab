//! Transcribe an audio file and write MIDI, GP5, MusicXML and ASCII tab.
//!
//! This mirrors `mtab transcribe` with happy-path settings: a model size, a
//! local safetensors file, standard tuning and no experimental analysis.
//!
//! Real weights are CC BY-NC 4.0 and gated; obtain them yourself, then:
//!
//! ```console
//! cargo run --release -p mtab-cli --example basic -- song.wav ./muscriptor-small.safetensors
//! ```
use std::path::Path;

use mtab_core::lm::{GenOptions, LmConfig, LmModel};
use mtab_core::model::ModelSize;
use mtab_guitar::{Capo, GuitarTrack, TabDocument, TempoMap, TrackMeta, Tuning};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (input, weights) = match (args.next(), args.next()) {
        (Some(input), Some(weights)) => (input, weights),
        _ => {
            eprintln!("usage: basic <song.wav> <muscriptor-small.safetensors>");
            std::process::exit(2);
        }
    };

    // 1. Decode, mix down (mono mean) and resample to the model's 16 kHz input.
    //    This is upstream's `load_audio` path: no normalization, no DC removal.
    let audio = mtab_audio::load_model_input(Path::new(&input))?;
    println!(
        "loaded {input}: {:.1} s at {} Hz",
        audio.frames() as f32 / audio.sample_rate as f32,
        audio.sample_rate
    );

    // 2. Load the checkpoint and decode with the guitar-family mask.
    let model = LmModel::load(LmConfig::from_size(ModelSize::Small), Path::new(&weights))?;
    let instruments = vec!["clean_electric_guitar".to_string()];
    let result = mtab_audio::transcribe::transcribe(
        &model,
        &audio,
        &instruments,
        &GenOptions::default(),
        true,
        |event| eprintln!("{event:?}"),
    )?;
    println!("model produced {} notes", result.notes.len());

    // 3. Assign string/fret numbers and build the tablature document.
    let tuning = Tuning::standard();
    let capo = Capo::NONE;
    let assigned = mtab_guitar::assign_notes(
        &result.notes,
        &tuning,
        capo,
        mtab_guitar::FingeringOptions::default(),
    )?;
    let document = TabDocument {
        meta: TrackMeta {
            title: Some("basic example".into()),
            tempo_map: TempoMap::constant(120.0),
            time_signatures: vec![(0, 4, 4)],
            frets: 22,
            source_hash: None,
        },
        tracks: vec![GuitarTrack {
            tuning,
            capo,
            instrument: Some("clean_electric_guitar".into()),
            notes: assigned,
        }],
        chord_symbols: Vec::new(),
    };
    // Keep the JSON tablature IR for later, non-notated work.
    std::fs::write("basic.ir.json", document.to_json()?)?;

    // 4. Export. Each format validates the document before writing anything.
    for (path, format) in [
        ("basic.mid", mtab_export::ExportFormat::Midi),
        ("basic.gp5", mtab_export::ExportFormat::GuitarPro5),
        ("basic.musicxml", mtab_export::ExportFormat::MusicXml),
        ("basic.txt", mtab_export::ExportFormat::AsciiTab),
    ] {
        mtab_export::export(&document, Path::new(path), format)?;
        println!("wrote {path}");
    }
    Ok(())
}
