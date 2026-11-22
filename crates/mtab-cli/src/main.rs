//! `mtab` CLI (PROMPT.md §6.12) — thin shell over the library crates.
//!
//! Scaffold status: `--models`, `detect-tuning --list`, and
//! `inspect-weights` work today; transcription and export land per the
//! milestone plan (M2–M6). Every failure prints an actionable next step
//! (§6.4).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

use mtab_core::{ModelSize, ModelSource, MtabError, Result, WeightsInfo};
use mtab_export::ExportFormat;
use mtab_guitar::{Capo, Grid, Tuning};

#[derive(Parser, Debug)]
#[command(
    name = "mtab",
    version,
    about = "Audio → guitar tablature (MIDI + Guitar Pro) on MuScriptor",
    long_about = "Convert any audio file into guitar tablature using the MuScriptor \
transcription model. The only inputs are the model size (small/medium/large) and \
its safetensors weights; everything else is built in.\n\n\
Weights: MuScriptor weights are CC BY-NC 4.0 (non-commercial) and gated on \
Hugging Face — accept the license on the model page and pass your own weights \
file. mtab never bundles or silently downloads weights."
)]
struct Cli {
    /// List the MuScriptor model variants and exit
    #[arg(long, global = true)]
    models: bool,

    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Transcribe audio into MIDI / Guitar Pro / MusicXML / ASCII tab
    Transcribe(TranscribeArgs),
    /// Inspect a safetensors weights file (size class, params, dtypes)
    InspectWeights {
        /// Path to a .safetensors file
        path: PathBuf,
    },
    /// Download gated weights (explicit, user-invoked only — PROMPT.md §1)
    Download {
        /// Model variant to fetch
        #[arg(long, default_value = "medium")]
        size: ModelSize,
        /// Environment variable holding the Hugging Face token
        #[arg(long = "token-env", default_value = "HF_TOKEN")]
        token_env: String,
    },
    /// Detect tuning and capo from audio (or list known tunings)
    DetectTuning {
        /// Audio file (omit with --list)
        input: Option<PathBuf>,
        /// List supported tuning names and exit
        #[arg(long)]
        list: bool,
    },
    /// Run the evaluation harness (GuitarSet etc., PROMPT.md §6.14)
    Eval {
        /// Corpus to evaluate against
        #[arg(long, default_value = "guitarset")]
        corpus: String,
        /// Write a markdown report to this path
        #[arg(long)]
        report: Option<PathBuf>,
    },
}

#[derive(Args, Debug)]
struct TranscribeArgs {
    /// Audio file (wav/mp3/flac/ogg/m4a)
    input: PathBuf,

    /// Model variant
    #[arg(long, default_value = "medium")]
    size: ModelSize,

    /// Weights: `<name>.safetensors` path, `hf://<org>/<repo>`, or http(s) URL
    #[arg(long)]
    weights: Option<String>,

    /// Restrict decoding to guitar-family instruments (token masking)
    #[arg(long)]
    instruments: Option<String>,

    /// Tuning: auto or a name from `mtab detect-tuning --list`
    #[arg(long, default_value = "auto")]
    tuning: String,

    /// Capo position: auto or 0-12
    #[arg(long, default_value = "auto")]
    capo: String,

    /// Fretboard size
    #[arg(long, default_value_t = 22)]
    frets: u8,

    /// Isolate the guitar stem before transcription (Demucs)
    #[arg(long)]
    separate: bool,

    /// Detect bends/vibrato/dead notes/slides (§6.8)
    #[arg(long)]
    techniques: bool,

    /// Add chord symbols above the tab (§6.9)
    #[arg(long)]
    chords: bool,

    /// Quantization grid: auto, off, 1/4..1/64 (T suffix = triplets)
    #[arg(long, default_value = "1/16")]
    grid: Grid,

    /// Beam-search width (default: greedy decoding)
    #[arg(long)]
    beam_size: Option<u8>,

    /// Sampling temperature (seeded; deterministic with --seed)
    #[arg(long)]
    temperature: Option<f32>,

    /// RNG seed for deterministic decoding
    #[arg(long)]
    seed: Option<u64>,

    /// Output files, comma-separated (extensions: .mid .gp5 .musicxml .txt)
    #[arg(short, long, value_delimiter = ',')]
    outputs: Vec<PathBuf>,

    /// Also dump the TabDocument IR as JSON to this path
    #[arg(long)]
    emit_ir: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(cli: Cli) -> Result<()> {
    if cli.models {
        print_models();
        return Ok(());
    }
    match cli.command {
        None => {
            print_models();
            Ok(())
        }
        Some(Cmd::InspectWeights { path }) => inspect_weights(&path),
        Some(Cmd::DetectTuning { list, input }) => detect_tuning(list, input.as_deref()),
        Some(Cmd::Download { size, token_env }) => download(size, &token_env),
        Some(Cmd::Eval { corpus, report }) => eval(&corpus, report.as_deref()),
        Some(Cmd::Transcribe(args)) => transcribe(args),
    }
}

fn print_models() {
    println!("MuScriptor model variants (weights are CC BY-NC 4.0, gated on Hugging Face):");
    for size in ModelSize::ALL {
        let (params, layers, dim) = size.shape();
        println!(
            "  {:<6} ~{:<4}M params, {:>2} layers, dim {:>4}  {}",
            size.to_string(),
            params / 1_000_000,
            layers,
            dim,
            size.hf_repo()
        );
    }
}

fn inspect_weights(path: &std::path::Path) -> Result<()> {
    let info = WeightsInfo::from_file(path)?;
    println!("file: {}", path.display());
    println!("tensors: {}", info.tensor_count());
    println!("parameters: {}", info.total_params());
    print!("dtype histogram:");
    for (dtype, count) in info.dtypes() {
        print!(" {dtype}×{count}");
    }
    println!();
    match info.likely_size() {
        Some(size) => println!(
            "likely variant: {size} (~{}M nominal)",
            size.shape().0 / 1_000_000
        ),
        None => println!("likely variant: unknown (parameter count matches no official size)"),
    }
    println!("largest tensors:");
    for (name, elements) in info.largest(5) {
        println!("  {name}: {elements} elements");
    }
    println!();
    println!("license note: MuScriptor weights are CC BY-NC 4.0 (non-commercial use only).");
    Ok(())
}

fn detect_tuning(list: bool, input: Option<&std::path::Path>) -> Result<()> {
    if list {
        println!("supported tunings (open-string MIDI pitches, low string first):");
        for t in Tuning::library() {
            println!("  {t}");
        }
        return Ok(());
    }
    let Some(input) = input else {
        return Err(MtabError::InvalidTuning(
            "provide an audio file, or use --list".to_owned(),
        ));
    };
    if !input.exists() {
        return Err(MtabError::UnsupportedAudio(input.display().to_string()));
    }
    Err(MtabError::NotImplemented(
        "tuning/capo detection (milestone M5)",
    ))
}

fn download(size: ModelSize, token_env: &str) -> Result<()> {
    let _ = std::env::var(token_env).map_err(|_| MtabError::LicenseGateRequired {
        repo: size.hf_repo().to_owned(),
    })?;
    Err(MtabError::NotImplemented(
        "weights download helper (milestone M3)",
    ))
}

fn eval(corpus: &str, report: Option<&std::path::Path>) -> Result<()> {
    let _ = (corpus, report);
    Err(MtabError::NotImplemented(
        "evaluation harness (milestone M5)",
    ))
}

fn transcribe(args: TranscribeArgs) -> Result<()> {
    // Validate everything we can today (§6.4: errors carry actionable hints).
    if !args.input.exists() {
        return Err(MtabError::UnsupportedAudio(
            args.input.display().to_string(),
        ));
    }
    if args.frets == 0 || args.frets > 36 {
        return Err(MtabError::InvalidTuning(format!(
            "--frets {} out of range (1-36)",
            args.frets
        )));
    }
    if args.tuning != "auto" {
        Tuning::from_name(&args.tuning)?;
    }
    if args.capo != "auto" {
        let pos: u8 = args
            .capo
            .parse()
            .map_err(|_| MtabError::InvalidCapo(u8::MAX))?;
        Capo::new(pos)?;
    }
    let source = match &args.weights {
        Some(w) => w.parse::<ModelSource>()?,
        None => {
            return Err(MtabError::WeightsNotFound {
                path: PathBuf::from("<--weights not provided>"),
            })
        }
    };
    for out in &args.outputs {
        if ExportFormat::from_path(out).is_none() {
            return Err(MtabError::InvalidModelSource(format!(
                "unknown output format for '{}' (use .mid/.gp5/.musicxml/.txt)",
                out.display()
            )));
        }
    }
    let _ = (
        source,
        args.instruments,
        args.beam_size,
        args.temperature,
        args.seed,
    );
    Err(MtabError::NotImplemented(
        "transcription pipeline (milestones M2-M4)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn clap_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_full_transcribe_invocation() {
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            "song.wav",
            "--size",
            "small",
            "--weights",
            "m.safetensors",
            "--tuning",
            "drop-d",
            "--grid",
            "1/8T",
            "-o",
            "song.mid,song.gp5",
        ])
        .unwrap();
        match cli.command {
            Some(Cmd::Transcribe(a)) => {
                assert_eq!(a.size, ModelSize::Small);
                assert_eq!(a.tuning, "drop-d");
                assert_eq!(a.outputs.len(), 2);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_grid() {
        assert!(Cli::try_parse_from(["mtab", "transcribe", "s.wav", "--grid", "banana"]).is_err());
    }
}
