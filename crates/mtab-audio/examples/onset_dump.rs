//! Export onset-envelope CSV for independent development checks.
use mtab_audio::{load_model_input, onset::spectral_flux};
use std::{io::Write, path::Path};

fn main() -> mtab_core::Result<()> {
    let path = std::env::args().nth(1).expect("audio path");
    let audio = load_model_input(Path::new(&path))?;
    let envelope = spectral_flux(&audio)?;
    let stdout = std::io::stdout();
    let mut output = std::io::BufWriter::new(stdout.lock());
    for (i, value) in envelope.strength.iter().enumerate() {
        writeln!(
            output,
            "{:.17},{value:.17}",
            envelope.frame_time(i).unwrap()
        )?;
    }
    Ok(())
}
