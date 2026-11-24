//! The §6.11 happy path: size + weights in, tablature out.
//!
//! ```console
//! cargo run -p muscriptor-tab --example basic -- song.wav ./muscriptor-small.safetensors
//! ```
//!
//! Writes `song.mid`, `song.gp5`, `song.musicxml` and `song.txt` beside the
//! input and prints the tab. The weights are CC BY-NC 4.0 and gated: obtain
//! them yourself (see `docs/weights.md`), nothing is downloaded here.

use std::path::PathBuf;

use muscriptor_tab::{DecodeConfig, GuitarConfig, ModelSize, ModelSource, Transcriber};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let (Some(audio), Some(weights)) = (args.next(), args.next()) else {
        eprintln!("usage: basic <audio> <weights.safetensors>");
        std::process::exit(2);
    };
    let audio = PathBuf::from(audio);
    let weights = PathBuf::from(weights);

    let transcriber = Transcriber::builder()
        .size(ModelSize::Small)
        .weights(ModelSource::Local(weights))
        // For a first run, say what the guitar is rather than trusting the
        // experimental detector.
        .guitar(GuitarConfig::fixed_standard(22))
        .decode(DecodeConfig {
            // A chunk that runs out of tokens is a wrong transcription, not a
            // warning worth ignoring.
            strict_eos: true,
            ..DecodeConfig::default()
        })
        .build()?;

    let tab = transcriber.transcribe_file_with_progress(&audio, |p| {
        eprintln!("{:?}: {}/{}", p.stage, p.completed, p.total.unwrap_or(0));
    })?;

    let stem = audio.with_extension("");
    tab.export_midi(stem.with_extension("mid"))?;
    tab.export_guitar_pro(stem.with_extension("gp5"))?;
    tab.export_musicxml(stem.with_extension("musicxml"))?;
    tab.export_ascii(stem.with_extension("txt"))?;

    println!("{}", tab.ascii_tab()?);
    for track in &tab.document.tracks {
        println!(
            "track: {} notes, {} capo {}",
            track.notes.len(),
            track.tuning.name,
            track.capo.position()
        );
    }
    Ok(())
}
