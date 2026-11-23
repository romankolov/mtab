//! Offline JSON adapter for the experimental symbolic chord diagnostic.
//! Reads a JSON array of NoteEvent from stdin and writes chord suggestions.
use std::io::{self, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut json = String::new();
    io::stdin().read_to_string(&mut json)?;
    let notes: Vec<mtab_core::NoteEvent> = serde_json::from_str(&json)?;
    let result = mtab_guitar::suggest_chords(&notes, 0.1)?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
