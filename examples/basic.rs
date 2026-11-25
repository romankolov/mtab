//! Transcribe one audio file with as little configuration as possible.
//!
//! This is the library equivalent of
//! `mtab transcribe song.wav --size small --weights model.safetensors -o song.mid,song.gp5`.
//! Run it with
//!
//! ```sh
//! cargo run --release --example basic -- song.wav /path/to/muscriptor-small.safetensors
//! ```
//!
//! The weights are **not** bundled: they are CC BY-NC 4.0 and gated on Hugging
//! Face (see `docs/weights.md`). Fetch them yourself and accept the licence.
//!
//! Expected output is two files next to the input: `song.mid` and `song.gp5`,
//! plus the ASCII tab on stdout.

use std::path::PathBuf;

use muscriptor_tab::{GuitarConfig, ModelSize, ModelSource, Transcriber};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let (Some(audio), Some(weights)) = (args.next(), args.next()) else {
        eprintln!("usage: basic <audio.wav> <muscriptor-small.safetensors>");
        std::process::exit(2);
    };
    let audio = PathBuf::from(audio);
    let weights = PathBuf::from(weights);

    let transcriber = Transcriber::builder()
        .size(ModelSize::Small)
        .weights(ModelSource::Local(weights))
        // Detection is the default, but naming it makes the example honest:
        // tuning detection is heuristic (§6.5), and a wrong answer moves every
        // fret number. Set `Fixed(...)` once you know what the recording is.
        .guitar(GuitarConfig::default())
        .build()?;

    let transcription = transcriber.transcribe_file(&audio)?;

    let stem = audio.with_extension("");
    transcription.export_midi(format!("{}.mid", stem.display()))?;
    transcription.export_guitar_pro(format!("{}.gp5", stem.display()))?;

    println!("{} notes", transcription.notes.len());
    for track in &transcription.document.tracks {
        println!(
            "track: {} strings, capo {}, {} notes",
            track.tuning.strings.len(),
            track.capo.position(),
            track.notes.len()
        );
    }
    if !transcription.omitted.is_empty() {
        // Only reachable with `allow_partial`; reported rather than hidden
        // because those notes are absent from every output, MIDI included.
        println!("omitted {} unplayable notes", transcription.omitted.len());
    }
    if !transcription.missing_eos.is_empty() {
        println!("chunks without EOS: {:?}", transcription.missing_eos);
    }
    print!("{}", transcription.ascii_tab()?);
    Ok(())
}
