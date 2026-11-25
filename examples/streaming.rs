//! Report progress while transcribing, and hold one model across many files.
//!
//! This is the shape a server or GUI wants (§4): load the checkpoint once
//! (cold start is the expensive part), then transcribe file after file through
//! the same [`Transcriber`], rendering [`Progress`] events as they arrive.
//!
//! Note what "streaming" does **not** mean here. The decoder generates
//! chunk by chunk, in order, with prelude forcing so a note sustained across a
//! chunk boundary cannot restart with the wrong instrument (§6.3). A chunk can
//! only be decoded after the one before it, so this reports progress through a
//! file in a fixed order; it does not emit notes while the audio is still
//! playing.
//!
//! ```sh
//! cargo run --release --example streaming -- weights.safetensors a.wav b.wav c.wav
//! ```

use std::path::PathBuf;
use std::time::Instant;

use muscriptor_tab::{ModelSize, ModelSource, Progress, Stage, Transcriber};

fn label(stage: Stage) -> &'static str {
    match stage {
        Stage::Audio => "decode",
        Stage::Tuning => "tuning",
        Stage::Decode => "generate",
        Stage::Fingering => "finger",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let Some(weights) = args.next() else {
        eprintln!("usage: streaming <weights.safetensors> <audio...>");
        std::process::exit(2);
    };
    let audio: Vec<PathBuf> = args.map(PathBuf::from).collect();
    if audio.is_empty() {
        eprintln!("usage: streaming <weights.safetensors> <audio...>");
        std::process::exit(2);
    }

    // Cold start: load once, reuse for every file.
    let load_start = Instant::now();
    let transcriber = Transcriber::builder()
        .size(ModelSize::Small)
        .weights(ModelSource::Local(PathBuf::from(weights)))
        .build()?;
    println!(
        "weights loaded in {:.2}s",
        load_start.elapsed().as_secs_f64()
    );

    for path in &audio {
        let start = Instant::now();
        // Render progress on one line. `total` is `None` for stages with no
        // known denominator (audio decode), and a count for the others.
        let transcription = transcriber.transcribe_file_with_progress(
            path,
            |Progress {
                 stage,
                 completed,
                 total,
             }| {
                match total {
                    Some(total) => eprint!("\r{} {completed}/{total}   ", label(stage)),
                    None => eprint!("\r{}   ", label(stage)),
                }
            },
        )?;
        eprintln!("\rtranscribed in {:.2}s", start.elapsed().as_secs_f64());
        println!(
            "{}: {} notes in {} track(s)",
            path.display(),
            transcription.notes.len(),
            transcription.document.tracks.len()
        );
    }
    Ok(())
}
