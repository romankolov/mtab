//! `mtab` CLI (PROMPT.md §6.12) — thin shell over the library crates.
//!
//! JSONL transcription is available with local safetensors weights. Guitar
//! fingering and other export formats remain under development.

use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

use mtab_core::lm::{GenOptions, LmConfig, LmModel};
use mtab_core::{ModelSize, ModelSource, MtabError, Result, WeightsInfo};
use mtab_guitar::{Grid, Tuning};

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

    /// Output format (currently only jsonl: one raw note per line)
    #[arg(long)]
    format: Option<String>,

    /// Fail if a chunk reaches the generation limit without EOS
    #[arg(long)]
    strict_eos: bool,

    /// Maximum generated tokens per chunk
    #[arg(long, default_value_t = 2000)]
    max_gen_len: usize,

    /// Output files, comma-separated (JSONL currently accepts one path)
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
    validate_transcribe(&args)?;
    let source = args
        .weights
        .as_deref()
        .ok_or_else(|| MtabError::WeightsNotFound {
            path: PathBuf::from("<--weights not provided>"),
        })?
        .parse::<ModelSource>()?;
    let path = match source {
        ModelSource::Local(path) => path,
        _ => return Err(MtabError::NotImplemented("remote weight fetching; download the licensed safetensors file separately and pass its local path")),
    };
    if !path.is_file() {
        return Err(MtabError::WeightsNotFound { path });
    }
    let model = LmModel::load(LmConfig::from_size(args.size), &path)?;
    transcribe_with_model(&args, &model)
}

fn validate_transcribe(args: &TranscribeArgs) -> Result<()> {
    if !args.input.is_file() {
        return Err(MtabError::UnsupportedAudio(format!(
            "file not found: {}",
            args.input.display()
        )));
    }
    if args.format.as_deref() != Some("jsonl") {
        return Err(MtabError::NotImplemented(
            "tablature exports; use --format jsonl for raw note events",
        ));
    }
    if args.outputs.len() > 1 {
        return Err(MtabError::InvalidModelSource(
            "JSONL accepts at most one output file; omit -o for stdout".into(),
        ));
    }
    if let Some(out) = args.outputs.first() {
        if let Ok(out_path) = std::fs::canonicalize(out) {
            let same_file =
                |path: &std::path::Path| std::fs::canonicalize(path).is_ok_and(|p| p == out_path);
            if same_file(&args.input)
                || args
                    .weights
                    .as_deref()
                    .is_some_and(|w| same_file(std::path::Path::new(w)))
            {
                return Err(MtabError::InvalidModelSource(
                    "output must not overwrite the input audio or weights".into(),
                ));
            }
        }
    }
    if args
        .instruments
        .as_deref()
        .is_some_and(|v| v.split(',').all(|s| s.trim().is_empty()))
    {
        return Err(MtabError::DecodeFailure {
            chunk_index: 0,
            reason: "--instruments was empty; provide a name such as guitar or omit the option"
                .into(),
        });
    }
    if args.separate || args.techniques || args.chords || args.emit_ir.is_some() {
        return Err(MtabError::NotImplemented(
            "separation, techniques, chords and TabDocument IR (future milestones)",
        ));
    }
    if args.tuning != "auto"
        || args.capo != "auto"
        || args.frets != 22
        || args.grid != "1/16".parse::<Grid>()?
    {
        return Err(MtabError::NotImplemented(
            "tuning, capo, fretting and rhythm options are not applied to raw JSONL notes",
        ));
    }
    Ok(())
}

fn transcribe_with_model(args: &TranscribeArgs, model: &LmModel) -> Result<()> {
    let audio = mtab_audio::load_model_input(&args.input)?;
    let names: Vec<String> = args.instruments.as_deref().map_or_else(Vec::new, |value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    });
    let options = GenOptions {
        max_gen_len: args.max_gen_len,
        beam_size: args.beam_size.unwrap_or(1) as usize,
        use_sampling: args.temperature.is_some(),
        temp: args.temperature.unwrap_or(1.0),
        seed: args.seed.unwrap_or(0),
        ..GenOptions::default()
    };
    let result = mtab_audio::transcribe::transcribe(
        model,
        &audio,
        &names,
        &options,
        args.strict_eos,
        |p| {
            eprintln!("transcribing chunks: {}/{}", p.completed, p.total);
        },
    )?;
    for index in result.missing_eos {
        eprintln!("warning: chunk {index} reached the token limit without EOS; use --strict-eos to treat this as an error");
    }
    let writer: Box<dyn Write> = match args.outputs.first() {
        Some(path) => Box::new(std::fs::File::create(path)?),
        None => Box::new(std::io::stdout()),
    };
    write_notes(writer, &result.notes)
}

fn write_notes(writer: impl Write, notes: &[mtab_core::NoteEvent]) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    for note in notes {
        serde_json::to_writer(&mut writer, note)?;
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
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
    #[test]
    fn jsonl_writer_writes_one_record_per_note() {
        let notes = [
            mtab_core::NoteEvent {
                start: 0.5,
                end: 1.0,
                midi: 64,
                velocity: 100,
                instrument: 26,
            },
            mtab_core::NoteEvent {
                start: 1.0,
                end: 1.5,
                midi: 60,
                velocity: 90,
                instrument: 24,
            },
        ];
        let mut out = Vec::new();
        write_notes(&mut out, &notes).unwrap();
        let decoded: Vec<mtab_core::NoteEvent> = std::str::from_utf8(&out)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(decoded, notes);
    }

    fn make_test_wav() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mtab-cli-{}-{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let n = 1024u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + n * 2).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&16000u32.to_le_bytes());
        bytes.extend_from_slice(&32000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(n * 2).to_le_bytes());
        bytes.resize(44 + (n * 2) as usize, 0);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn jsonl_transcription_runs_with_synthetic_model() {
        let audio = make_test_wav();
        let output = audio.with_extension("jsonl");
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mtab-core/tests/fixtures/model/tiny.fixture");
        let model = LmModel::load(LmConfig::fixture_tiny(), &model_path).unwrap();
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--format",
            "jsonl",
            "--max-gen-len",
            "2",
            "--instruments",
            "guitar",
            "-o",
            output.to_str().unwrap(),
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        validate_transcribe(&args).unwrap();
        transcribe_with_model(&args, &model).unwrap();
        let contents = std::fs::read_to_string(&output).unwrap();
        for line in contents.lines() {
            let _: mtab_core::NoteEvent = serde_json::from_str(line).unwrap();
        }
        std::fs::remove_file(&audio).unwrap();
        std::fs::remove_file(&output).unwrap();
    }

    #[test]
    fn jsonl_refuses_overwriting_input() {
        let audio = make_test_wav();
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--format",
            "jsonl",
            "-o",
            audio.to_str().unwrap(),
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        assert!(validate_transcribe(&args).is_err());
        assert!(std::fs::metadata(&audio).unwrap().len() > 0);
        std::fs::remove_file(audio).unwrap();
    }
}
