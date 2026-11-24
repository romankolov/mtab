//! One process, many files: load the weights once, transcribe a directory.
//!
//! §6.11's `.progress(cb)`: the callback gets a [`Progress`] per stage, so a GUI
//! can draw a bar without the library knowing what a widget is.
//!
//! ```console
//! cargo run -p muscriptor-tab --example streaming -- ./album ./muscriptor-small.safetensors ./out
//! ```

use std::path::{Path, PathBuf};

use muscriptor_tab::{DecodeConfig, ModelSize, ModelSource, Progress, Stage, Transcriber};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let (Some(input_dir), Some(weights), out_dir) = (args.next(), args.next(), args.next()) else {
        eprintln!("usage: streaming <audio-dir> <weights.safetensors> [out-dir]");
        std::process::exit(2);
    };
    let input_dir = PathBuf::from(input_dir);
    let out_dir = out_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    // Building is the expensive step: 400 MB to 5.5 GB of weights.
    let transcriber = Transcriber::builder()
        .size(ModelSize::Small)
        .weights(ModelSource::Local(PathBuf::from(weights)))
        .decode(DecodeConfig::default())
        .build()?;

    for path in audio_files(&input_dir)? {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("out")
            .to_owned();
        eprintln!("== {stem}");
        let tab = transcriber.transcribe_file_with_progress(&path, |p: Progress| {
            let total = p.total.map_or_else(|| "?".to_owned(), |t| t.to_string());
            let bar = match p.stage {
                Stage::Audio => "decode",
                Stage::Tuning => "tuning",
                Stage::Decode => "generate",
                Stage::Fingering => "fingering",
            };
            eprint!("\r  {bar:9} {}/{total}", p.completed);
            if p.stage == Stage::Fingering && p.total == Some(p.completed) && p.completed > 0 {
                eprintln!();
            }
        })?;
        tab.export_midi(out_dir.join(format!("{stem}.mid")))?;
        tab.export_guitar_pro(out_dir.join(format!("{stem}.gp5")))?;
        eprintln!(
            "  -> {} notes, {} chunk(s) without EOS",
            tab.notes.len(),
            tab.missing_eos.len()
        );
    }
    Ok(())
}

fn audio_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    const AUDIO: &[&str] = &["wav", "mp3", "flac", "ogg", "m4a"];
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| AUDIO.contains(&e.to_ascii_lowercase().as_str()))
        })
        .collect();
    files.sort();
    Ok(files)
}
