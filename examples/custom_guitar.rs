//! Transcribe for a guitar that is not in standard tuning at capo 0.
//!
//! Two things change when the instrument is different, and both are the
//! caller's job to state (§6.6):
//!
//! - **Tuning** fixes which pitches each string can produce.
//! - **Capo** shifts every fret number; a chord shape at fret 3 behind a capo
//!   at 3 is written as fret 0.
//!
//! Detection (§6.5) is a convenience for the common case. If you know the
//! instrument, say so: a wrong guess does not fail loudly, it silently moves
//! note after note onto the wrong string.
//!
//! ```sh
//! cargo run --release --example custom_guitar -- song.wav /path/to/weights.safetensors
//! ```

use std::path::PathBuf;

use muscriptor_tab::{
    Capo, CapoChoice, GuitarConfig, ModelSize, ModelSource, Transcriber, Tuning, TuningChoice,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let (Some(audio), Some(weights)) = (args.next(), args.next()) else {
        eprintln!("usage: custom_guitar <audio.wav> <weights.safetensors>");
        std::process::exit(2);
    };
    let audio = PathBuf::from(audio);
    let weights = PathBuf::from(weights);

    // Drop D: the low E down a whole tone, everything else standard. The
    // strings are MIDI pitch numbers, low string first, strictly ascending.
    let drop_d = Tuning::new("drop-d", vec![38, 45, 50, 55, 59, 64])?;
    // A capo at fret 2: shapes sound two semitones higher, frets are relative.
    let capo = Capo::new(2)?;

    let transcriber = Transcriber::builder()
        .size(ModelSize::Small)
        .weights(ModelSource::Local(weights))
        .guitar(GuitarConfig {
            tuning: TuningChoice::Fixed(drop_d),
            capo: CapoChoice::Fixed(capo),
            frets: 22,
            // A drop-D riff that uses the open low string alongside a shape
            // high up the neck can ask for more than four frets of span. Retry
            // with seven rather than losing the whole track — but review the
            // result, because seven frets is not comfortable for everyone.
            wider_span: Some(7),
            allow_partial: false,
        })
        .build()?;

    let transcription = transcriber.transcribe_file(&audio)?;
    for track in &transcription.document.tracks {
        println!(
            "{}: {} notes, capo {}",
            track.tuning.name,
            track.notes.len(),
            track.capo.position()
        );
        // A note's `midi` is the sounding pitch; `fret` is relative to the capo.
        // `midi == tuning.strings[string] + capo + fret` holds for every note.
        for note in track.notes.iter().take(5) {
            let expected = u8::try_from(
                i32::from(track.tuning.strings[note.string as usize])
                    + i32::from(track.capo.position())
                    + i32::from(note.fret),
            )
            .unwrap_or(0);
            assert_eq!(note.midi, expected, "fret is relative to the capo");
            println!(
                "  {:.2}s string {} fret {} (midi {})",
                note.start, note.string, note.fret, note.midi
            );
        }
    }
    Ok(())
}
