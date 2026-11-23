//! Print experimental automatic beat times for offline composition checks.
use mtab_audio::{automatic_beat::automatic_beats, load_model_input};
use std::path::Path;

fn main() -> mtab_core::Result<()> {
    let path = std::env::args().nth(1).expect("audio path");
    if let Some(result) = automatic_beats(&load_model_input(Path::new(&path))?)? {
        for time in result.path.times {
            println!("{time:.9}");
        }
    }
    Ok(())
}
