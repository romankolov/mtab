//! Dump experimental tempo-conditioned beat times for offline evaluation.
use mtab_audio::{beat_path::track_with_tempo, load_model_input, onset::spectral_flux};
use std::{
    io::{self, Write},
    path::Path,
};

fn main() -> mtab_core::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("audio path");
    let bpm = args
        .next()
        .expect("target BPM")
        .parse::<f64>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid BPM"))?;
    let tightness = args
        .next()
        .expect("tightness")
        .parse::<f64>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid tightness"))?;
    let audio = load_model_input(Path::new(&path))?;
    let envelope = spectral_flux(&audio)?;
    let path = track_with_tempo(&envelope, bpm, tightness)?;
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    for time in path.times {
        writeln!(out, "{time:.9}")?;
    }
    Ok(())
}
