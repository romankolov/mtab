//! Stream the audio file in its five-second chunks and report per-chunk notes.
//!
//! `mtab_audio::transcribe::transcribe` already chunks internally with forced
//! tie preludes at the boundaries; this example drives
//! `mtab_core::chunk::transcribe_chunks` itself so each chunk can be handled as
//! soon as it is decoded, which is the shape a long-file or live pipeline
//! needs. Peak memory is the model plus one chunk, not the whole song.
//!
//! ```console
//! cargo run --release -p mtab-cli --example streaming -- long-song.wav ./muscriptor-small.safetensors
//! ```
use std::path::Path;

use mtab_audio::mel::MelFrontend;
use mtab_core::chunk::{transcribe_chunks, CHUNK_SAMPLES};
use mtab_core::lm::{GenOptions, LmConfig, LmModel, RowSpec};
use mtab_core::model::ModelSize;
use mtab_core::mt3::instruments;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (input, weights) = match (args.next(), args.next()) {
        (Some(input), Some(weights)) => (input, weights),
        _ => {
            eprintln!("usage: streaming <song.wav> <muscriptor-small.safetensors>");
            std::process::exit(2);
        }
    };

    let audio = mtab_audio::load_model_input(Path::new(&input))?;
    let chunks = audio.samples.len().div_ceil(CHUNK_SAMPLES);
    println!(
        "{} samples = {chunks} chunk(s) of {:.0} s; only one chunk is held at a time",
        audio.samples.len(),
        CHUNK_SAMPLES as f64 / f64::from(audio.sample_rate)
    );

    let model = LmModel::load(LmConfig::from_size(ModelSize::Small), Path::new(&weights))?;
    let groups: Vec<u16> = ["clean_electric_guitar"]
        .iter()
        .map(|name| {
            instruments::group_id(
                &instruments::resolve_instrument_names(&[name.to_string()]).expect("guitar family")
                    [0],
            )
            .expect("resolved name")
        })
        .collect();
    let frontend = MelFrontend::upstream();
    let options = GenOptions::default();

    let result = transcribe_chunks(
        &audio.samples,
        true,
        |index, chunk, prompt| {
            let mel: Vec<f32> = frontend.compute_mel(chunk).into_iter().flatten().collect();
            let frames = mel.len() / 512;
            let conditioned = RowSpec {
                mel: Some(&mel),
                n_mel_frames: frames,
                valid_frames: 500,
                instrument: Some(groups.as_slice()),
                dataset: None,
            };
            let conds = model.encode_conditions(&[conditioned])?;
            // A different seed per chunk keeps sampling reproducible overall.
            let opts = GenOptions {
                seed: options.seed.wrapping_add(index as u64),
                ..GenOptions::default()
            };
            Ok(model.generate(prompt, &conds, &opts).tokens)
        },
        |progress| eprintln!("chunk {}/{}", progress.completed, progress.total),
    )?;

    // Notes come back ordered across chunk boundaries and held notes are
    // joined by the forced tie preludes, so this is the whole performance.
    println!(
        "{} notes, {} chunk(s) without EOS",
        result.notes.len(),
        result.missing_eos.len()
    );
    Ok(())
}
