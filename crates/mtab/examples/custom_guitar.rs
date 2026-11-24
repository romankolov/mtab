//! Solve for a specific instrument instead of guessing: drop-D, capo 2, 24
//! frets, and a wider fingering search when a grip will not fit four frets.
//!
//! ```console
//! cargo run -p muscriptor-tab --example custom_guitar -- song.wav ./muscriptor-small.safetensors
//! ```

use std::path::PathBuf;

use muscriptor_tab::{
    Capo, CapoChoice, GuitarConfig, ModelSize, ModelSource, Transcriber, Tuning, TuningChoice,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let (Some(audio), Some(weights)) = (args.next(), args.next()) else {
        eprintln!("usage: custom_guitar <audio> <weights.safetensors>");
        std::process::exit(2);
    };

    let transcriber = Transcriber::builder()
        .size(ModelSize::Small)
        .weights(ModelSource::Local(PathBuf::from(weights)))
        .guitar(GuitarConfig {
            tuning: TuningChoice::Fixed(Tuning::from_name("drop-d")?),
            capo: CapoChoice::Fixed(Capo::new(2)?),
            frets: 24,
            // Retry an impossible grip at up to six frets rather than failing
            // the whole track. The result needs review: it is legal, not
            // necessarily comfortable.
            wider_span: Some(6),
            allow_partial: false,
        })
        .build()?;

    let tab = transcriber.transcribe_file(audio)?;
    for track in &tab.document.tracks {
        let span = track
            .notes
            .iter()
            .map(|n| (n.string, n.fret))
            .max_by_key(|(_, fret)| *fret);
        println!(
            "{} notes on {} capo {}; highest position {:?}",
            track.notes.len(),
            track.tuning.name,
            track.capo.position(),
            span
        );
    }
    Ok(())
}
