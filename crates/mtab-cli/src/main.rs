//! `mtab` CLI (PROMPT.md §6.12) — thin shell over the library crates.
//!
//! Local-weight transcription supports raw JSONL, preliminary MIDI and a
//! restricted experimental GP5 subset. Other exports are under development.

mod guitarset;
mod guitarset_notes;
mod guitarset_tab;

use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

mod download;

use mtab_core::lm::{GenOptions, LmConfig, LmModel};
use mtab_core::{ModelSize, ModelSource, MtabError, Result, WeightsInfo};
use mtab_export::ExportFormat;
use mtab_guitar::{
    Capo, FingeringOptions, Grid, GuitarTrack, TabDocument, TempoMap, TrackMeta, Tuning,
};

#[derive(Parser, Debug)]
#[command(
    name = "mtab",
    version,
    about = "Audio → guitar tablature (MIDI + Guitar Pro) on MuScriptor",
    long_about = "Experimental audio-to-guitar transcription using MuScriptor. Some automatic analysis and export features remain incomplete; see README.md for current limitations.\n\nWeights are CC BY-NC 4.0 (non-commercial) and gated on Hugging Face. Accept the license and supply your own local or previously cached weights. mtab never bundles or silently downloads weights."
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
    /// Fetch the gated weights explicitly, with your own Hugging Face token
    #[command(
        long_about = "Downloads the CC BY-NC 4.0 weights for one variant into a local \
file. This is the only command that touches the network, it only runs when you \
call it, and it requires a Hugging Face token whose account has accepted the \
license. The token is read from an environment variable, never written to disk \
or printed. The file is downloaded to <output>.part and only renamed once it \
parses as safetensors and matches the requested size, so an interrupted run \
cannot leave a file that looks usable but is not."
    )]
    Download {
        /// Model variant to fetch
        #[arg(long, default_value = "medium")]
        size: ModelSize,
        /// Environment variable holding the Hugging Face token
        #[arg(long = "token-env", default_value = "HF_TOKEN")]
        token_env: String,
        /// Destination file (default: muscriptor-<size>.safetensors)
        #[arg(long, short)]
        out: Option<std::path::PathBuf>,
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
    Eval(EvalArgs),
}

#[derive(Args, Debug)]
struct EvalArgs {
    /// Corpus to evaluate against
    #[arg(long, default_value = "guitarset")]
    corpus: String,
    /// Path to a local GuitarSet .jams file or directory of .jams files
    #[arg(long)]
    annotations: Option<PathBuf>,
    /// Evaluate string assignment on ground-truth notes (not audio transcription)
    #[arg(long, conflicts_with_all = ["predictions", "tab_ir", "tab_manifest", "audio_manifest"])]
    oracle_notes: bool,
    /// Score a JSONL note transcription against one matching GuitarSet JAMS file
    #[arg(long, conflicts_with_all = ["oracle_notes", "tab_ir", "tab_manifest", "audio_manifest"])]
    predictions: Option<PathBuf>,
    /// Score one emitted TabDocument IR against matching GuitarSet JAMS
    #[arg(long, conflicts_with_all = ["oracle_notes", "predictions", "tab_manifest", "audio_manifest"], requires = "clip_duration")]
    tab_ir: Option<PathBuf>,
    /// Score an explicit JSON manifest of annotation, IR, and duration entries
    #[arg(long, conflicts_with_all = ["oracle_notes", "predictions", "tab_ir", "annotations", "clip_duration", "audio_manifest"])]
    tab_manifest: Option<PathBuf>,
    /// Run end-to-end evaluation over an explicit JSON audio/annotation manifest
    #[arg(long, conflicts_with_all = ["oracle_notes", "predictions", "tab_ir", "tab_manifest", "annotations", "clip_duration"], requires = "weights")]
    audio_manifest: Option<PathBuf>,
    /// Local safetensors weights for end-to-end evaluation (no downloads)
    #[arg(long, requires = "audio_manifest")]
    weights: Option<PathBuf>,
    /// Model architecture for end-to-end evaluation
    #[arg(long, default_value = "small")]
    size: ModelSize,
    /// Retry failed fingering with partial assignments in audio-manifest evaluation
    #[arg(long, requires = "audio_manifest")]
    partial_fingering: bool,
    /// Maximum generated tokens per chunk for end-to-end evaluation
    #[arg(long, default_value_t = 2000)]
    max_gen_len: usize,
    /// Evaluation duration in seconds (required for tab-IR frame evaluation)
    #[arg(long)]
    clip_duration: Option<f64>,
    /// Write a markdown report to this path (default: stdout)
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct TranscribeArgs {
    /// Audio file (wav/mp3/flac/ogg/m4a)
    input: PathBuf,

    /// Model variant
    #[arg(long, default_value = "medium")]
    size: ModelSize,

    /// Local safetensors, cached size keyword, or cached `hf://<org>/<repo>` (no downloads)
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

    /// Retry impossible fingerings with up to seven frets of span (review for comfort)
    #[arg(long)]
    allow_wide_stretch: bool,

    /// Retry unplayable tracks with partial fingering and write omitted notes to JSON
    #[arg(long, conflicts_with = "allow_wide_stretch")]
    partial_report: Option<PathBuf>,

    /// Isolate the guitar stem before transcription (Demucs)
    #[arg(long)]
    separate: bool,

    /// Estimate relative velocities from audio (experimental; mix energy, not isolated notes)
    #[arg(long)]
    dynamics: bool,

    /// Experimental isolated-note bend inference (MIDI only; manual review required)
    #[arg(long)]
    experimental_bends: bool,

    /// Suggest dead notes and palm mutes from audio (experimental; uncalibrated)
    #[arg(long)]
    techniques: bool,

    /// Add chord symbols above the tab (§6.9)
    #[arg(long)]
    chords: bool,

    /// Use an explicit constant integer tempo (4–255 BPM); no audio beat detection
    #[arg(long, conflicts_with = "beat_times")]
    bpm: Option<u16>,

    /// JSON quarter-beat seconds, possibly offset (MIDI only; no detection)
    #[arg(long, conflicts_with = "bpm")]
    beat_times: Option<PathBuf>,

    /// Experimental audio beat hypothesis (MIDI only; often inaccurate; review output)
    #[arg(long, conflicts_with_all = ["bpm", "beat_times"])]
    experimental_beat_tracking: bool,

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

    /// Output format: jsonl, midi, gp5, musicxml, or ascii (or infer from extension)
    #[arg(long)]
    format: Option<String>,

    /// Fail if a chunk reaches the generation limit without EOS
    #[arg(long)]
    strict_eos: bool,

    /// Maximum generated tokens per chunk
    #[arg(long, default_value_t = 2000)]
    max_gen_len: usize,

    /// Output files, comma-separated (.mid/.gp5/.musicxml/.xml/.txt/.tab)
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
        Some(Cmd::Download {
            size,
            token_env,
            out,
        }) => {
            let destination = out.unwrap_or_else(|| {
                std::path::PathBuf::from(format!("muscriptor-{size}.safetensors"))
            });
            download::download(&download::DownloadRequest {
                size,
                destination,
                token_env,
            })
            .map(|path| println!("{}", path.display()))
        }
        Some(Cmd::Eval(args)) => eval(args),
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
    let audio = mtab_audio::load_model_input(input)?;
    let histogram = mtab_audio::pitch_histogram::pitch_histogram(&audio)?;
    let detection =
        mtab_guitar::detect_tuning(&histogram.weights, &mtab_guitar::standard_candidates())?;
    println!(
        "Experimental monophonic tuning estimate; polyphony and fretted notes can mislead it."
    );
    println!(
        "Frames: {} voiced / {} analysed",
        histogram.voiced_frames, histogram.frames_examined
    );
    if let Some(best) = &detection.best {
        println!(
            "Best hypothesis: {} capo {} (open-pitch fraction {:.3}, margin {:.3})",
            best.candidate.tuning.name,
            best.candidate.capo.position(),
            best.open_fraction,
            detection.margin
        );
    }
    if let Some(next) = &detection.runner_up {
        println!(
            "Runner-up: {} capo {} (open-pitch fraction {:.3})",
            next.candidate.tuning.name,
            next.candidate.capo.position(),
            next.open_fraction
        );
    }
    if detection.sufficient_evidence {
        println!("Evidence threshold met (uncalibrated heuristic; verify manually).");
    } else {
        println!("Insufficient or ambiguous evidence; default is standard tuning, capo 0.");
    }
    Ok(())
}

fn eval(args: EvalArgs) -> Result<()> {
    if args.corpus != "guitarset" {
        return Err(MtabError::NotImplemented(
            "only GuitarSet evaluation is supported",
        ));
    }
    if let Some(manifest) = args.audio_manifest.as_deref() {
        return guitarset_tab::evaluate_audio_manifest(
            manifest,
            args.weights.as_deref().expect("clap requires weights"),
            args.size,
            args.max_gen_len,
            args.partial_fingering,
            args.report.as_deref(),
        );
    }
    if let Some(manifest) = args.tab_manifest.as_deref() {
        return guitarset_tab::evaluate_manifest(manifest, args.report.as_deref());
    }
    if !args.oracle_notes && args.predictions.is_none() && args.tab_ir.is_none() {
        return Err(MtabError::NotImplemented("automatic end-to-end corpus evaluation; provide --predictions <JSONL>, --tab-ir, --tab-manifest, or use --oracle-notes"));
    }
    let path = args.annotations.as_deref().ok_or_else(|| {
        MtabError::InvalidModelSource(
            "provide --annotations <local GuitarSet .jams file or directory>".into(),
        )
    })?;
    if let Some(tab_ir) = args.tab_ir.as_deref() {
        guitarset_tab::evaluate(
            path,
            tab_ir,
            args.clip_duration.expect("clap requires duration"),
            args.report.as_deref(),
        )
    } else if let Some(predictions) = args.predictions.as_deref() {
        guitarset_notes::evaluate(
            path,
            predictions,
            args.clip_duration,
            args.report.as_deref(),
        )
    } else {
        if args.clip_duration.is_some() {
            return Err(MtabError::InvalidModelSource(
                "--clip-duration requires --predictions or --tab-ir".into(),
            ));
        }
        guitarset::evaluate(path, args.report.as_deref())
    }
}

fn transcribe(args: TranscribeArgs) -> Result<()> {
    let tempo = validate_transcribe(&args)?;
    let source = args
        .weights
        .as_deref()
        .ok_or_else(|| MtabError::WeightsNotFound {
            path: PathBuf::from("<--weights not provided>"),
        })?
        .parse::<ModelSource>()?;
    let path = match source {
        ModelSource::Local(path) => path,
        ModelSource::Cached { size } => {
            if size != args.size {
                return Err(MtabError::InvalidModelSource(format!("cached weights size {size} does not match --size {}; specify matching sizes", args.size)));
            }
            mtab_core::model::resolve_hf_cache(size.hf_repo(), &mtab_core::model::hf_hub_cache_dir()?)?
        },
        ModelSource::HfRepo(repo) => mtab_core::model::resolve_hf_cache(&repo, &mtab_core::model::hf_hub_cache_dir()?)?,
        ModelSource::Url(_) => return Err(MtabError::NotImplemented("remote weight fetching; download the licensed safetensors file separately and pass its local path")),
    };
    if !path.is_file() {
        return Err(MtabError::WeightsNotFound { path });
    }
    protect_resolved_weights(&args, &path)?;
    let model = LmModel::load(LmConfig::from_size(args.size), &path)?;
    transcribe_with_model(&args, &model, &tempo)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    Jsonl,
    Document,
}

fn output_mode(args: &TranscribeArgs) -> Result<OutputMode> {
    match args.format.as_deref() {
        Some("jsonl") => Ok(OutputMode::Jsonl),
        Some("midi" | "mid" | "gp5" | "musicxml" | "xml" | "ascii" | "txt" | "tab") => {
            Ok(OutputMode::Document)
        }
        None if !args.outputs.is_empty() => Ok(OutputMode::Document),
        None => Ok(OutputMode::Jsonl),
        _ => Err(MtabError::NotImplemented(
            "choose --format jsonl, midi, gp5, musicxml, or ascii",
        )),
    }
}

fn validate_transcribe(args: &TranscribeArgs) -> Result<TempoMap> {
    if !args.input.is_file() {
        return Err(MtabError::UnsupportedAudio(format!(
            "file not found: {}",
            args.input.display()
        )));
    }
    let mode = output_mode(args)?;
    match mode {
        OutputMode::Jsonl if args.outputs.len() > 1 => {
            return Err(MtabError::InvalidModelSource(
                "JSONL accepts at most one output file; omit -o for stdout".into(),
            ))
        }
        OutputMode::Document if args.outputs.is_empty() => {
            return Err(MtabError::InvalidModelSource(
                "document output requires at least one -o path".into(),
            ))
        }
        OutputMode::Document => {
            for path in &args.outputs {
                let format = ExportFormat::from_path(path);
                if format.is_none() {
                    return Err(MtabError::InvalidModelSource("unsupported output extension; use .mid, .gp5, .musicxml, .xml, .txt or .tab".into()));
                }
                let explicit = match args.format.as_deref() {
                    Some("midi" | "mid") => Some(ExportFormat::Midi),
                    Some("gp5") => Some(ExportFormat::GuitarPro5),
                    Some("musicxml" | "xml") => Some(ExportFormat::MusicXml),
                    Some("ascii" | "txt" | "tab") => Some(ExportFormat::AsciiTab),
                    _ => None,
                };
                if explicit.is_some() && format != explicit {
                    return Err(MtabError::InvalidModelSource(
                        "--format must agree with each output extension; omit it for mixed outputs"
                            .into(),
                    ));
                }
            }
        }
        _ => {}
    }
    let local_weights =
        args.weights
            .as_deref()
            .and_then(|value| match value.parse::<ModelSource>() {
                Ok(ModelSource::Local(path)) => Some(path),
                _ => None,
            });
    let protected: Vec<&std::path::Path> = std::iter::once(args.input.as_path())
        .chain(local_weights.as_deref())
        .chain(args.beat_times.as_deref())
        .collect();
    let protected_paths: std::collections::HashSet<PathBuf> =
        protected.iter().filter_map(|p| path_identity(p)).collect();
    let mut protected_files = std::collections::HashSet::new();
    for path in &protected {
        if let Some(identity) = existing_file_identity(path)? {
            protected_files.insert(identity);
        }
    }
    let mut output_paths = std::collections::HashSet::new();
    let mut output_files = std::collections::HashSet::new();
    for path in args
        .outputs
        .iter()
        .chain(args.emit_ir.iter())
        .chain(args.partial_report.iter())
    {
        if let Some(identity) = path_identity(path) {
            if protected_paths.contains(&identity) {
                return Err(MtabError::InvalidOutput(
                    "output must not overwrite input audio, weights or beat times".into(),
                ));
            }
            if !output_paths.insert(identity) {
                return Err(MtabError::InvalidOutput(
                    "output paths must be distinct".into(),
                ));
            }
        }
        // Canonical paths do not identify hard links: compare existing files
        // by OS file identity too, before opening any output for writing.
        if let Some(identity) = existing_file_identity(path)? {
            if protected_files.contains(&identity) {
                return Err(MtabError::InvalidOutput(
                    "output must not overwrite input audio, weights or beat times (hard-link alias)".into(),
                ));
            }
            if !output_files.insert(identity) {
                return Err(MtabError::InvalidOutput(
                    "output paths must be distinct (hard-link alias)".into(),
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
    if args.separate {
        return Err(MtabError::NotImplemented(
            "separation requires an auxiliary model and is not implemented",
        ));
    }
    if mode == OutputMode::Jsonl {
        if args.emit_ir.is_some()
            || args.tuning != "auto"
            || args.capo != "auto"
            || args.frets != 22
            || args.allow_wide_stretch
            || args.partial_report.is_some()
            || args.bpm.is_some()
            || args.beat_times.is_some()
            || args.experimental_beat_tracking
            || args.experimental_bends
            || args.techniques
            || args.chords
            || args.grid != Grid::SIXTEENTH
        {
            return Err(MtabError::NotImplemented("tuning, capo, fretting, rhythm and TabDocument IR are not applied to raw JSONL notes"));
        }
    } else {
        if (args.beat_times.is_some() || args.experimental_beat_tracking)
            && args
                .outputs
                .iter()
                .any(|path| ExportFormat::from_path(path) != Some(ExportFormat::Midi))
        {
            return Err(MtabError::InvalidTempo(
                "supplied or experimental beat tracking supports MIDI outputs only; GP5, MusicXML and ASCII do not support changing tempos".into(),
            ));
        }
        if args.experimental_bends
            && args
                .outputs
                .iter()
                .any(|path| ExportFormat::from_path(path) != Some(ExportFormat::Midi))
        {
            return Err(MtabError::NotImplemented(
                "experimental bend inference supports MIDI outputs only",
            ));
        }
        if args.bpm.is_some_and(|bpm| !(4..=255).contains(&bpm)) {
            return Err(MtabError::InvalidTempo(
                "--bpm must be an integer from 4 to 255".into(),
            ));
        }
        if args.frets == 0 || args.frets > 36 {
            return Err(MtabError::InvalidTuning("--frets must be 1–36".into()));
        }
        if args.tuning != "auto" {
            Tuning::from_name(&args.tuning)?;
        }
        if args.capo != "auto" {
            let position = args
                .capo
                .parse::<u8>()
                .map_err(|_| MtabError::InvalidCapo(u8::MAX))?;
            Capo::new(position)?;
            if position > args.frets {
                return Err(MtabError::InvalidTuning(
                    "capo cannot exceed fretboard size".into(),
                ));
            }
        }
        if args.grid != Grid::SIXTEENTH && args.grid != Grid::Off {
            return Err(MtabError::NotImplemented(
                "rhythm quantization; use --grid off for unquantized MIDI",
            ));
        }
    }
    resolve_tempo(args)
}

/// Read and validate supplied beat times once, before model loading or output.
fn resolve_tempo(args: &TranscribeArgs) -> Result<TempoMap> {
    const MAX_BYTES: usize = 2 * 1024 * 1024;
    let Some(path) = &args.beat_times else {
        return Ok(TempoMap::constant(f64::from(args.bpm.unwrap_or(120))));
    };
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_BYTES {
        return Err(MtabError::InvalidTempo(
            "--beat-times file exceeds 2 MiB".into(),
        ));
    }
    let times: Vec<f64> = serde_json::from_slice(&bytes).map_err(|error| {
        MtabError::InvalidTempo(format!(
            "--beat-times must be a JSON array of seconds: {error}"
        ))
    })?;
    let map = TempoMap::from_offset_quarter_beat_times(&times)?;
    if times[0] > 0.0 {
        eprintln!("warning: offset beat times use a synthetic pre-roll tempo; 4/4 bars start at zero, not necessarily at the first beat or downbeat");
    }
    Ok(map)
}

/// Canonical identity for an existing path or an as-yet-unwritten file whose
/// parent exists; catches aliases such as ./song.mid vs song.mid.
fn path_identity(path: &std::path::Path) -> Option<PathBuf> {
    std::fs::canonicalize(path).ok().or_else(|| {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."));
        Some(std::fs::canonicalize(parent).ok()?.join(path.file_name()?))
    })
}

/// Capture OS file identity for existing regular files (including hard links).
/// Missing paths are handled by canonical path identity instead. Opening an
/// existing file can fail, in which case preflight fails closed.
fn existing_file_identity(path: &std::path::Path) -> Result<Option<same_file::Handle>> {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(Some(same_file::Handle::from_path(path)?)),
        Ok(_) => Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Preflight a report path against input paths, including hard-link aliases.
/// This is not atomic against concurrent filesystem changes after the check.
struct ReportGuard {
    path: Option<PathBuf>,
    file: Option<same_file::Handle>,
}

impl ReportGuard {
    fn new(report: Option<&std::path::Path>) -> Result<Self> {
        Ok(Self {
            path: report.and_then(path_identity),
            file: match report {
                Some(path) => existing_file_identity(path)?,
                None => None,
            },
        })
    }

    fn protect(&self, input: &std::path::Path) -> Result<()> {
        if self.path.is_some() && self.path == path_identity(input)
            || self.file.is_some() && self.file == existing_file_identity(input)?
        {
            return Err(MtabError::InvalidOutput(
                "report must not overwrite an input, including a hard-link alias".into(),
            ));
        }
        Ok(())
    }
}

/// Protect resolved cache blobs too, including aliases through symlinks or hard links.
fn protect_resolved_weights(args: &TranscribeArgs, path: &std::path::Path) -> Result<()> {
    let identity = path_identity(path);
    let file = existing_file_identity(path)?;
    for output in args
        .outputs
        .iter()
        .chain(args.emit_ir.iter())
        .chain(args.partial_report.iter())
    {
        if identity.is_some() && identity == path_identity(output)
            || file.is_some() && file == existing_file_identity(output)?
        {
            return Err(MtabError::InvalidOutput(
                "output must not overwrite resolved weights, including cache aliases".into(),
            ));
        }
    }
    Ok(())
}

fn transcribe_with_model(args: &TranscribeArgs, model: &LmModel, tempo: &TempoMap) -> Result<()> {
    let audio = mtab_audio::load_model_input(&args.input)?;
    let estimated_tempo = if args.experimental_beat_tracking {
        eprintln!("warning: experimental beat tracking is inaccurate, can lock to offbeats or subdivisions, has no confidence or downbeat detection, and requires review; meter is assumed 4/4");
        let hypothesis = mtab_audio::automatic_beat::automatic_beats(&audio)?.ok_or_else(|| {
            MtabError::InvalidTempo("experimental tracker found fewer than two beats; supply --bpm or --beat-times instead".into())
        })?;
        eprintln!(
            "experimental beat hypothesis: {:.2} BPM, {} beats (not a confidence score)",
            hypothesis.tempo_evidence.bpm,
            hypothesis.path.times.len()
        );
        Some(TempoMap::from_offset_quarter_beat_times(
            &hypothesis.path.times,
        )?)
    } else {
        None
    };
    let tempo = estimated_tempo.as_ref().unwrap_or(tempo);
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
    let mut result = mtab_audio::transcribe::transcribe(
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
    if args.dynamics {
        eprintln!("warning: experimental dynamics estimate uses mixed audio; overlapping notes and other instruments can bias velocities");
        let report = mtab_audio::dynamics::estimate_velocities(&audio, &mut result.notes)?;
        if report.skipped > 0 {
            eprintln!(
                "warning: retained model velocity for {} notes without usable attack energy",
                report.skipped
            );
        }
    }
    match output_mode(args)? {
        OutputMode::Jsonl => {
            let writer: Box<dyn Write> = match args.outputs.first() {
                Some(path) => Box::new(std::fs::File::create(path)?),
                None => Box::new(std::io::stdout()),
            };
            write_notes(writer, &result.notes)
        }
        OutputMode::Document => {
            let (doc, omissions) = build_midi_document(args, &audio, &result.notes, tempo)?;
            let partial_report = args
                .partial_report
                .as_ref()
                .map(|_| partial_report_bytes(result.notes.len(), &omissions))
                .transpose()?;
            write_document_outputs(args, &doc, partial_report.as_deref())
        }
    }
}

fn partial_report_bytes(note_count: usize, omissions: &[serde_json::Value]) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec_pretty(&serde_json::json!({
        "version": 1,
        "method": "strict search with span-four partial retry",
        "input_note_count": note_count,
        "assigned_note_count": note_count - omissions.len(),
        "unassigned": omissions,
    }))?)
}

/// Render all requested formats before writing any, so an unsupported score
/// cannot leave a misleading partial MIDI or notation output behind. I/O errors
/// during the subsequent writes can still leave earlier outputs on disk.
fn write_document_outputs(
    args: &TranscribeArgs,
    doc: &TabDocument,
    partial_report: Option<&[u8]>,
) -> Result<()> {
    let wants_midi = args
        .outputs
        .iter()
        .any(|p| ExportFormat::from_path(p) == Some(ExportFormat::Midi));
    let wants_gp5 = args
        .outputs
        .iter()
        .any(|p| ExportFormat::from_path(p) == Some(ExportFormat::GuitarPro5));
    let midi = wants_midi
        .then(|| mtab_export::midi::midi_bytes(doc))
        .transpose()?;
    let gp5 = if wants_gp5 {
        let (bytes, plan) = mtab_export::gp5::gp5_bytes(doc)?;
        eprintln!("warning: experimental GP5 uses fixed {} BPM, 4/4 and 1/32 quantization regardless of --grid; maximum timing shift {:.1} ms; review the score", plan.bpm, plan.max_boundary_shift_seconds * 1000.0);
        Some(bytes)
    } else {
        None
    };
    let wants_xml = args
        .outputs
        .iter()
        .any(|p| ExportFormat::from_path(p) == Some(ExportFormat::MusicXml));
    let wants_ascii = args
        .outputs
        .iter()
        .any(|p| ExportFormat::from_path(p) == Some(ExportFormat::AsciiTab));
    let xml = if wants_xml {
        let (text, plan) = mtab_export::musicxml::musicxml(doc)?;
        eprintln!("warning: experimental MusicXML uses fixed {} BPM, 4/4 and 1/32 quantization regardless of --grid; maximum timing shift {:.1} ms; velocities omitted, staves may double playback; MuseScore may change fingerings and lose capo or palm mutes", plan.bpm, plan.max_boundary_shift_seconds * 1000.0);
        Some(text.into_bytes())
    } else {
        None
    };
    let ascii = if wants_ascii {
        let (text, plan) = mtab_export::ascii::ascii_tab(doc)?;
        eprintln!("warning: experimental ASCII tab uses fixed {} BPM, 4/4 and 1/32 quantization regardless of --grid; maximum timing shift {:.1} ms; velocities omitted", plan.bpm, plan.max_boundary_shift_seconds * 1000.0);
        Some(text.into_bytes())
    } else {
        None
    };
    let ir = args
        .emit_ir
        .as_ref()
        .map(|_| serde_json::to_vec_pretty(doc))
        .transpose()?;
    if args.partial_report.is_some() != partial_report.is_some() {
        return Err(MtabError::InvalidModelSource(
            "partial report data must match the requested report path".into(),
        ));
    }
    for path in &args.outputs {
        let bytes = match ExportFormat::from_path(path) {
            Some(ExportFormat::Midi) => midi.as_ref(),
            Some(ExportFormat::GuitarPro5) => gp5.as_ref(),
            Some(ExportFormat::MusicXml) => xml.as_ref(),
            Some(ExportFormat::AsciiTab) => ascii.as_ref(),
            _ => None,
        }
        .ok_or_else(|| {
            MtabError::InvalidModelSource("unsupported document output extension".into())
        })?;
        std::fs::write(path, bytes)?;
    }
    if let (Some(path), Some(data)) = (&args.emit_ir, ir) {
        std::fs::write(path, data)?;
    }
    if let (Some(path), Some(data)) = (&args.partial_report, partial_report) {
        std::fs::write(path, data)?;
    }
    Ok(())
}

fn resolve_tuning_and_capo(
    args: &TranscribeArgs,
    audio: &mtab_audio::AudioBuffer,
) -> Result<(Tuning, Capo)> {
    let tuning = if args.tuning == "auto" {
        None
    } else {
        Some(Tuning::from_name(&args.tuning)?)
    };
    let capo = if args.capo == "auto" {
        None
    } else {
        Some(Capo::new(
            args.capo
                .parse::<u8>()
                .map_err(|_| MtabError::InvalidCapo(u8::MAX))?,
        )?)
    };
    if let (Some(t), Some(c)) = (&tuning, capo) {
        return Ok((t.clone(), c));
    }
    let histogram = mtab_audio::pitch_histogram::pitch_histogram(audio)?;
    let candidates: Vec<_> = mtab_guitar::standard_candidates()
        .into_iter()
        .filter(|c| tuning.as_ref().map_or(true, |t| &c.tuning == t))
        .filter(|c| capo.map_or(true, |capo| c.capo == capo))
        .filter(|c| c.capo.position() <= args.frets)
        .collect();
    let detection = mtab_guitar::detect_tuning(&histogram.weights, &candidates)?;
    if detection.sufficient_evidence {
        let best = detection
            .best
            .expect("sufficient evidence has a best candidate");
        eprintln!("warning: experimental tuning estimate {} capo {} (uncalibrated margin {:.3}); verify it, especially for polyphonic audio", best.candidate.tuning.name, best.candidate.capo.position(), detection.margin);
        Ok((best.candidate.tuning, best.candidate.capo))
    } else {
        let t = tuning.unwrap_or_else(Tuning::standard);
        let c = capo.unwrap_or(Capo::NONE);
        eprintln!(
            "warning: tuning/capo evidence is insufficient or ambiguous; using {} capo {}",
            t.name,
            c.position()
        );
        Ok((t, c))
    }
}

fn build_midi_document(
    args: &TranscribeArgs,
    audio: &mtab_audio::AudioBuffer,
    notes: &[mtab_core::NoteEvent],
    tempo: &TempoMap,
) -> Result<(TabDocument, Vec<serde_json::Value>)> {
    let (tuning, capo) = resolve_tuning_and_capo(args, audio)?;
    if args.experimental_beat_tracking {
        eprintln!("warning: MIDI uses raw note timing with an experimental beat hypothesis and assumed 4/4 meter; inspect the result");
    } else if args.beat_times.is_some() {
        eprintln!("warning: MIDI uses raw timing, supplied quarter beats and assumed 4/4 meter");
    } else {
        let bpm = args.bpm.unwrap_or(120);
        eprintln!("warning: no automatic beat tracking selected; MIDI uses raw timing and a fixed {bpm} BPM, 4/4 tempo map");
    }
    let bend_assessments = if args.experimental_bends {
        eprintln!("warning: experimental bend inference is uncalibrated and can mistake mixture pitch, tuning or vibrato for bends; inspect MIDI pitch wheels and the IR");
        Some(mtab_audio::bend_analysis::suggest_bends(audio, notes)?)
    } else {
        None
    };
    let articulation_assessments = if args.techniques {
        eprintln!("warning: experimental dead-note/palm-mute suggestions use fixed thresholds on a mix; review every mark before trusting it");
        Some(mtab_audio::articulation::suggest_articulations(
            audio, notes,
        )?)
    } else {
        None
    };
    let mut suggested_bends = 0usize;
    let mut suggested_dead_notes = 0usize;
    let mut suggested_palm_mutes = 0usize;
    let mut grouped: std::collections::BTreeMap<u16, Vec<(usize, mtab_core::NoteEvent)>> =
        std::collections::BTreeMap::new();
    for (index, note) in notes.iter().enumerate() {
        grouped
            .entry(note.instrument)
            .or_default()
            .push((index, *note));
    }
    let mut omissions: Vec<(usize, serde_json::Value)> = Vec::new();
    let mut tracks = Vec::new();
    for (program, indexed_notes) in grouped {
        let track_notes: Vec<_> = indexed_notes.iter().map(|(_, note)| *note).collect();
        let options = FingeringOptions {
            max_fret: args.frets,
            ..FingeringOptions::default()
        };
        let mut skipped_local = Vec::new();
        let mut assigned = if args.allow_wide_stretch {
            let assignment =
                mtab_guitar::assign_notes_with_stretch(&track_notes, &tuning, capo, options, 7)?;
            if assignment.used_wider_span {
                eprintln!("warning: track program {program} required a wider fingering search (up to 7 frets); review physical comfort");
            }
            assignment.notes
        } else if args.partial_report.is_some() {
            match mtab_guitar::assign_notes(&track_notes, &tuning, capo, options) {
                Ok(assigned) => assigned,
                Err(MtabError::UnplayableNote { .. }) => {
                    let partial =
                        mtab_guitar::assign_notes_partial(&track_notes, &tuning, capo, options)?;
                    skipped_local.clone_from(&partial.unassigned_indices);
                    for &index in &partial.unassigned_indices {
                        let (input_index, note) = indexed_notes[index];
                        omissions.push((
                            input_index,
                            serde_json::json!({"input_index": input_index, "note": note}),
                        ));
                    }
                    if !partial.unassigned_indices.is_empty() {
                        eprintln!("warning: partial fingering omitted {} notes from track program {program} in ALL exports including MIDI; see {}", partial.unassigned_indices.len(), args.partial_report.as_ref().unwrap().display());
                    }
                    partial.notes.into_iter().flatten().collect()
                }
                Err(error) => return Err(error),
            }
        } else {
            mtab_guitar::assign_notes(&track_notes, &tuning, capo, options)?
        };
        if let Some(assessments) = &bend_assessments {
            let retained = indexed_notes
                .iter()
                .enumerate()
                .filter(|(i, _)| skipped_local.binary_search(i).is_err());
            for (target, (_, (source_index, _))) in assigned.iter_mut().zip(retained) {
                use mtab_audio::bend_analysis::{BendAssessment, BendShape};
                if let BendAssessment::Candidate(candidate) = assessments[*source_index] {
                    // Fingering can clip a sub-nanosecond overlap at the end.
                    // Reject instead of silently changing an invalid contour.
                    if candidate.peak_time <= target.start || candidate.peak_time >= target.end {
                        continue;
                    }
                    target.bend = Some(mtab_guitar::PitchBend {
                        start: target.start,
                        peak: candidate.peak_time,
                        depth_cents: candidate.depth_cents as f32,
                        kind: match candidate.shape {
                            BendShape::Bend => mtab_guitar::BendKind::Bend,
                            BendShape::Release => mtab_guitar::BendKind::Release,
                            BendShape::BendRelease => mtab_guitar::BendKind::BendRelease,
                        },
                    });
                    suggested_bends += 1;
                }
            }
        }
        if let Some(assessments) = &articulation_assessments {
            let retained = indexed_notes
                .iter()
                .enumerate()
                .filter(|(i, _)| skipped_local.binary_search(i).is_err());
            for (target, (_, (source_index, _))) in assigned.iter_mut().zip(retained) {
                use mtab_audio::articulation::Articulation;
                match assessments[*source_index].articulation {
                    Articulation::DeadNote => {
                        target.technique.push(mtab_guitar::Technique::DeadNote);
                        suggested_dead_notes += 1;
                    }
                    Articulation::PalmMute => {
                        target.technique.push(mtab_guitar::Technique::PalmMute);
                        suggested_palm_mutes += 1;
                    }
                    _ => {}
                }
            }
        }
        tracks.push(GuitarTrack {
            tuning: tuning.clone(),
            capo,
            instrument: Some(mtab_core::mt3::instruments::instrument_for_program(program)),
            notes: assigned,
        });
    }
    if bend_assessments.is_some() {
        eprintln!("experimental bends: {suggested_bends} candidates attached to {} input notes; absence is not proof of no bend", notes.len());
    }
    if articulation_assessments.is_some() {
        eprintln!("experimental techniques: {suggested_dead_notes} dead-note and {suggested_palm_mutes} palm-mute suggestions attached to {} input notes; absence is not proof of an open note", notes.len());
    }
    if tracks.is_empty() {
        eprintln!(
            "warning: no notes were detected; notation exports contain an empty guitar track"
        );
        tracks.push(GuitarTrack {
            tuning,
            capo,
            instrument: None,
            notes: Vec::new(),
        });
    }
    let chord_symbols = if args.chords {
        let detected = mtab_guitar::suggest_chords(notes, 0.25)?;
        eprintln!(
            "chords: {} symbols from detected notes (limited vocabulary; a symbol spans at least 0.25 s)",
            detected.len()
        );
        detected
    } else {
        Vec::new()
    };
    omissions.sort_by_key(|(index, _)| *index);
    Ok((
        TabDocument {
            meta: TrackMeta {
                title: args
                    .input
                    .file_stem()
                    .and_then(|v| v.to_str())
                    .map(str::to_owned),
                tempo_map: tempo.clone(),
                time_signatures: vec![(0, 4, 4)],
                frets: args.frets,
                source_hash: None,
            },
            tracks,
            chord_symbols,
        },
        omissions.into_iter().map(|(_, value)| value).collect(),
    ))
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
    fn partial_evaluation_requires_audio_manifest() {
        assert!(Cli::try_parse_from([
            "mtab",
            "eval",
            "--partial-fingering",
            "--oracle-notes",
            "--annotations",
            "a.jams"
        ])
        .is_err());
        assert!(Cli::try_parse_from([
            "mtab",
            "eval",
            "--partial-fingering",
            "--audio-manifest",
            "manifest.json",
            "--weights",
            "w.safetensors"
        ])
        .is_ok());
    }

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
        static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mtab-cli-{}-{}-{sequence}.wav",
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
        let tempo = validate_transcribe(&args).unwrap();
        transcribe_with_model(&args, &model, &tempo).unwrap();
        let contents = std::fs::read_to_string(&output).unwrap();
        for line in contents.lines() {
            let _: mtab_core::NoteEvent = serde_json::from_str(line).unwrap();
        }
        std::fs::remove_file(&audio).unwrap();
        std::fs::remove_file(&output).unwrap();
    }

    #[test]
    fn synthetic_silent_transcription_writes_all_four_formats() {
        let audio = make_test_wav();
        let midi = audio.with_extension("mid");
        let gp5 = audio.with_extension("gp5");
        let xml = audio.with_extension("musicxml");
        let ascii = audio.with_extension("txt");
        let outputs = format!(
            "{},{},{},{}",
            midi.display(),
            gp5.display(),
            xml.display(),
            ascii.display()
        );
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mtab-core/tests/fixtures/model/tiny.fixture");
        let model = LmModel::load(LmConfig::fixture_tiny(), &model_path).unwrap();
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--tuning",
            "standard",
            "--capo",
            "0",
            "--max-gen-len",
            "2",
            "--instruments",
            "guitar",
            "-o",
            &outputs,
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        let tempo = validate_transcribe(&args).unwrap();
        transcribe_with_model(&args, &model, &tempo).unwrap();
        assert!(std::fs::read(&midi).unwrap().starts_with(b"MThd"));
        assert!(std::fs::read(&gp5)
            .unwrap()
            .starts_with(b"\x18FICHIER GUITAR PRO v5.00"));
        assert!(std::fs::read_to_string(&xml).unwrap().contains("<rest/>"));
        assert!(std::fs::read_to_string(&ascii).unwrap().contains("Bar 1"));
        if let Some(python) = std::env::var_os("MTAB_PYGUITARPRO_PYTHON") {
            let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tools/check_empty_gp5.py");
            let output = std::process::Command::new(python)
                .arg(script)
                .arg(&gp5)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        if let Some(python) = std::env::var_os("MTAB_XML_PYTHON") {
            let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tools/check_musicxml_schema.py");
            let output = std::process::Command::new(python)
                .arg(script)
                .arg(&xml)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        for path in [audio, midi, gp5, xml, ascii] {
            std::fs::remove_file(path).unwrap();
        }
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
    #[test]
    fn constant_tempo_propagates_to_all_formats() {
        let audio = make_test_wav();
        let midi = audio.with_extension("mid");
        let gp5 = audio.with_extension("gp5");
        let xml = audio.with_extension("musicxml");
        let ascii = audio.with_extension("txt");
        let ir = audio.with_extension("json");
        let outputs = format!(
            "{},{},{},{}",
            midi.display(),
            gp5.display(),
            xml.display(),
            ascii.display()
        );
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--tuning",
            "standard",
            "--capo",
            "0",
            "--bpm",
            "96",
            "--emit-ir",
            ir.to_str().unwrap(),
            "-o",
            &outputs,
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        validate_transcribe(&args).unwrap();
        let data = mtab_audio::AudioBuffer {
            samples: vec![],
            sample_rate: 16_000,
            channels: 1,
        };
        let events = [
            mtab_core::NoteEvent {
                start: 0.0,
                end: 0.3125,
                midi: 40,
                velocity: 81,
                instrument: 24,
            },
            mtab_core::NoteEvent {
                start: 0.625,
                end: 1.25,
                midi: 45,
                velocity: 104,
                instrument: 24,
            },
        ];
        let (doc, omitted) =
            build_midi_document(&args, &data, &events, &resolve_tempo(&args).unwrap()).unwrap();
        assert!(omitted.is_empty());
        assert_eq!(doc.meta.tempo_map, TempoMap::constant(96.0));
        assert_eq!(doc.tracks[0].notes[1].start, 0.625);
        let (_, plan) = mtab_export::gp5::gp5_bytes(&doc).unwrap();
        assert_eq!(plan.bpm, 96);
        assert_eq!(plan.max_boundary_shift_seconds, 0.0);
        write_document_outputs(&args, &doc, None).unwrap();
        if let Some(python) = std::env::var_os("MTAB_TEMPO_PYTHON") {
            let tools = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools");
            for (script, paths) in [
                ("check_cli_fixed_tempo.py", vec![&midi, &xml, &ascii, &ir]),
                ("check_gp5_cli_ir.py", vec![&gp5, &ir]),
            ] {
                let output = std::process::Command::new(&python)
                    .arg(tools.join(script))
                    .args(paths)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{script}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
        for path in [audio, midi, gp5, xml, ascii, ir] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn supplied_beats_flow_to_midi_and_ir_without_reloading() {
        let audio = make_test_wav();
        let beats = audio.with_extension("beats.json");
        let midi = audio.with_extension("mid");
        let ir = audio.with_extension("ir.json");
        std::fs::write(&beats, "[0.0, 0.5, 1.1, 1.8, 2.3]").unwrap();
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--tuning",
            "standard",
            "--capo",
            "0",
            "--beat-times",
            beats.to_str().unwrap(),
            "--emit-ir",
            ir.to_str().unwrap(),
            "-o",
            midi.to_str().unwrap(),
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        let tempo = validate_transcribe(&args).unwrap();
        // Preflight owns a snapshot: later edits do not change a long-running job.
        std::fs::write(&beats, "[0, 1]").unwrap();
        let data = mtab_audio::AudioBuffer {
            samples: vec![],
            sample_rate: 16_000,
            channels: 1,
        };
        let events: Vec<_> = [
            (0.25, 0.5, 40),
            (0.8, 1.1, 45),
            (1.45, 1.8, 50),
            (2.05, 2.3, 55),
        ]
        .into_iter()
        .map(|(start, end, midi)| mtab_core::NoteEvent {
            start,
            end,
            midi,
            velocity: 90,
            instrument: 24,
        })
        .collect();
        let (doc, omitted) = build_midi_document(&args, &data, &events, &tempo).unwrap();
        assert!(omitted.is_empty());
        assert_eq!(
            doc.meta.tempo_map,
            TempoMap::from_quarter_beat_times(&[0.0, 0.5, 1.1, 1.8, 2.3]).unwrap()
        );
        write_document_outputs(&args, &doc, None).unwrap();
        let ir_doc = TabDocument::from_json(&std::fs::read_to_string(&ir).unwrap()).unwrap();
        assert_eq!(ir_doc.meta.tempo_map.changes.len(), tempo.changes.len());
        for (actual, expected) in ir_doc.meta.tempo_map.changes.iter().zip(&tempo.changes) {
            assert!((actual.time - expected.time).abs() < 1e-12);
            assert!((actual.bpm - expected.bpm).abs() < 1e-12);
        }
        if let Some(python) = std::env::var_os("MTAB_TEMPO_PYTHON") {
            let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tools/check_midi_variable_tempo.py");
            let output = std::process::Command::new(python)
                .arg(script)
                .arg(&midi)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        for path in [audio, beats, midi, ir] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn offset_supplied_beats_flow_through_cli_document_to_midi() {
        let audio = make_test_wav();
        let beats = audio.with_extension("beats.json");
        let midi = audio.with_extension("mid");
        std::fs::write(&beats, "[0.25, 0.75, 1.35, 2.05]").unwrap();
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--tuning",
            "standard",
            "--capo",
            "0",
            "--beat-times",
            beats.to_str().unwrap(),
            "-o",
            midi.to_str().unwrap(),
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        let tempo = validate_transcribe(&args).unwrap();
        let data = mtab_audio::AudioBuffer {
            samples: vec![],
            sample_rate: 16_000,
            channels: 1,
        };
        let events: Vec<_> = [
            (0.125, 0.25, 40),
            (0.5, 0.75, 45),
            (1.05, 1.35, 50),
            (1.7, 2.05, 55),
        ]
        .into_iter()
        .map(|(start, end, midi)| mtab_core::NoteEvent {
            start,
            end,
            midi,
            velocity: 90,
            instrument: 24,
        })
        .collect();
        let (doc, omitted) = build_midi_document(&args, &data, &events, &tempo).unwrap();
        assert!(omitted.is_empty());
        write_document_outputs(&args, &doc, None).unwrap();
        if let Some(python) = std::env::var_os("MTAB_TEMPO_PYTHON") {
            let output = std::process::Command::new(python)
                .arg(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../tools/check_midi_offset_tempo.py"),
                )
                .arg(&midi)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        for path in [audio, beats, midi] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn technique_and_chord_flags_validate_but_jsonsl_and_separation_do_not() {
        let audio = make_test_wav();
        let args = |options: &[&str]| {
            let mut argv = vec!["mtab", "transcribe", audio.to_str().unwrap()];
            argv.extend_from_slice(options);
            let Some(Cmd::Transcribe(args)) = Cli::try_parse_from(argv).unwrap().command else {
                panic!()
            };
            args
        };
        // Audio-derived techniques are attached to the IR, which drives the
        // notation exports and MIDI, so every document output is supported.
        validate_transcribe(&args(&["--techniques", "-o", "a.mid"])).unwrap();
        validate_transcribe(&args(&["--chords", "-o", "a.txt"])).unwrap();
        validate_transcribe(&args(&["--techniques", "--chords", "-o", "a.gp5"])).unwrap();
        // Raw JSONL has no IR, so these flags are rejected there.
        assert!(validate_transcribe(&args(&["--techniques", "--format", "jsonl"])).is_err());
        assert!(validate_transcribe(&args(&["--chords", "--format", "jsonl"])).is_err());
        // Separation still needs an auxiliary model.
        assert!(validate_transcribe(&args(&["--separate"])).is_err());
        std::fs::remove_file(audio).unwrap();
    }

    #[test]
    fn experimental_bends_reject_unsupported_formats() {
        let audio = make_test_wav();
        for options in [
            vec!["--format", "jsonl"],
            vec!["-o", "test.gp5"],
            vec!["-o", "test.musicxml"],
            vec!["-o", "test.txt"],
        ] {
            let mut argv = vec![
                "mtab",
                "transcribe",
                audio.to_str().unwrap(),
                "--experimental-bends",
            ];
            argv.extend(options);
            let Some(Cmd::Transcribe(args)) = Cli::try_parse_from(argv).unwrap().command else {
                panic!()
            };
            assert!(validate_transcribe(&args).is_err());
        }
        std::fs::remove_file(audio).unwrap();
    }

    #[test]
    fn technique_suggestions_reach_the_ir_and_chords_reach_exports() {
        use std::f64::consts::PI;
        let path = make_test_wav();
        let text = path.with_extension("txt");
        let Some(Cmd::Transcribe(args)) = Cli::try_parse_from([
            "mtab",
            "transcribe",
            path.to_str().unwrap(),
            "--techniques",
            "--chords",
            "--tuning",
            "standard",
            "--capo",
            "0",
            "-o",
            text.to_str().unwrap(),
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        let tempo = validate_transcribe(&args).unwrap();
        // A muted chuck then an open chord, so one dead note and one chord
        // symbol are expected; the point is that the wiring works at all.
        let mut samples = vec![0.0_f32; 16_000 * 3 / 2];
        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f64 / 16_000.0;
            if t < 0.25 {
                // Short, dark, fast-decaying low tone: a dead note.
                let (a, b, c) = (
                    2.0 * PI * 82.41 * t,
                    2.0 * PI * 164.82 * t,
                    2.0 * PI * 247.23 * t,
                );
                *sample =
                    (0.3 * (a.sin() + b.sin() / 2.0 + c.sin() / 3.0) * (-80.0 * t).exp()) as f32;
            } else if t < 1.4 {
                let local = t - 0.25;
                *sample = (0.3
                    * (-local).exp()
                    * ((2.0 * PI * 261.63 * local).sin()
                        + (2.0 * PI * 329.63 * local).sin()
                        + (2.0 * PI * 392.0 * local).sin())) as f32;
            }
        }
        let audio = mtab_audio::AudioBuffer {
            samples,
            sample_rate: 16_000,
            channels: 1,
        };
        let notes = [
            mtab_core::NoteEvent {
                start: 0.0,
                end: 0.25,
                midi: 40,
                velocity: 90,
                instrument: 24,
            },
            mtab_core::NoteEvent {
                start: 0.25,
                end: 1.4,
                midi: 60,
                velocity: 90,
                instrument: 24,
            },
            mtab_core::NoteEvent {
                start: 0.25,
                end: 1.4,
                midi: 64,
                velocity: 90,
                instrument: 24,
            },
            mtab_core::NoteEvent {
                start: 0.25,
                end: 1.4,
                midi: 67,
                velocity: 90,
                instrument: 24,
            },
        ];
        let (doc, omitted) = build_midi_document(&args, &audio, &notes, &tempo).unwrap();
        assert!(omitted.is_empty());
        // The dead note is marked, and the open notes are not.
        let dead = doc.tracks[0]
            .notes
            .iter()
            .filter(|n| n.technique.contains(&mtab_guitar::Technique::DeadNote))
            .count();
        assert_eq!(dead, 1, "{:?}", doc.tracks[0].notes);
        assert_eq!(doc.chord_symbols.len(), 1);
        let (rendered, _) = mtab_export::ascii::ascii_tab(&doc).unwrap();
        // C major is recognized from the three open notes and prints on the
        // beat that starts at 0.25 s, not on the dead-note beat.
        assert!(rendered.contains("chd |---C"), "{rendered}");
        // The same marks survive the TabDocument JSON round trip used by
        // --emit-ir.
        let round_tripped = mtab_guitar::TabDocument::from_json(&doc.to_json().unwrap()).unwrap();
        assert_eq!(round_tripped.chord_symbols.len(), 1);
        assert!(round_tripped.tracks[0]
            .notes
            .iter()
            .any(|n| n.technique.contains(&mtab_guitar::Technique::DeadNote)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn experimental_bend_inference_maps_partial_assignments_to_correct_notes() {
        use std::f64::consts::PI;
        let path = make_test_wav();
        let midi = path.with_extension("mid");
        let ir = path.with_extension("ir.json");
        let partial = path.with_extension("partial.json");
        let Some(Cmd::Transcribe(args)) = Cli::try_parse_from([
            "mtab",
            "transcribe",
            path.to_str().unwrap(),
            "--experimental-bends",
            "--tuning",
            "standard",
            "--capo",
            "0",
            "--partial-report",
            partial.to_str().unwrap(),
            "--emit-ir",
            ir.to_str().unwrap(),
            "-o",
            midi.to_str().unwrap(),
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        let tempo = validate_transcribe(&args).unwrap();
        let mut phase = 0.0_f64;
        let samples = (0..19_200)
            .map(|i| {
                let t = i as f64 / 16_000.0 - 0.2;
                if t < 0.0 {
                    return 0.0;
                }
                let cents = ((t - 0.2) / 0.35).clamp(0.0, 1.0) * 100.0;
                phase += 2.0 * PI * 110.0 * 2.0_f64.powf(cents / 1200.0) / 16_000.0;
                (0.4 * phase.sin()) as f32
            })
            .collect();
        let audio = mtab_audio::AudioBuffer {
            samples,
            sample_rate: 16_000,
            channels: 1,
        };
        let notes = [
            mtab_core::NoteEvent {
                start: 0.0,
                end: 0.1,
                midi: 10,
                velocity: 90,
                instrument: 24,
            },
            mtab_core::NoteEvent {
                start: 0.2,
                end: 1.2,
                midi: 45,
                velocity: 90,
                instrument: 24,
            },
        ];
        let (doc, omitted) = build_midi_document(&args, &audio, &notes, &tempo).unwrap();
        assert_eq!(omitted.len(), 1);
        assert_eq!(omitted[0]["input_index"], 0);
        assert_eq!(doc.tracks[0].notes.len(), 1);
        let note = &doc.tracks[0].notes[0];
        assert_eq!(note.midi, 45);
        let bend = note.bend.unwrap();
        assert_eq!(bend.kind, mtab_guitar::BendKind::Bend);
        assert!((bend.depth_cents - 100.0).abs() < 15.0);
        let report = partial_report_bytes(notes.len(), &omitted).unwrap();
        write_document_outputs(&args, &doc, Some(&report)).unwrap();
        let parsed = TabDocument::from_json(&std::fs::read_to_string(&ir).unwrap()).unwrap();
        assert!(parsed.tracks[0].notes[0].bend.is_some());
        if let Some(python) = std::env::var_os("MTAB_TEMPO_PYTHON") {
            let output = std::process::Command::new(python)
                .arg(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../tools/check_midi_inferred_bend.py"),
                )
                .arg(&midi)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        for path in [path, midi, ir, partial] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn experimental_bends_skip_overlapping_notes_across_tracks() {
        let path = make_test_wav();
        let midi = path.with_extension("mid");
        let Some(Cmd::Transcribe(args)) = Cli::try_parse_from([
            "mtab",
            "transcribe",
            path.to_str().unwrap(),
            "--experimental-bends",
            "--tuning",
            "standard",
            "--capo",
            "0",
            "-o",
            midi.to_str().unwrap(),
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        let tempo = validate_transcribe(&args).unwrap();
        let audio = mtab_audio::AudioBuffer {
            samples: vec![0.0; 16_000],
            sample_rate: 16_000,
            channels: 1,
        };
        let notes = [
            mtab_core::NoteEvent {
                start: 0.0,
                end: 0.8,
                midi: 45,
                velocity: 90,
                instrument: 24,
            },
            mtab_core::NoteEvent {
                start: 0.2,
                end: 1.0,
                midi: 50,
                velocity: 90,
                instrument: 25,
            },
        ];
        let (doc, omissions) = build_midi_document(&args, &audio, &notes, &tempo).unwrap();
        assert!(omissions.is_empty());
        assert_eq!(doc.tracks.len(), 2);
        assert!(doc
            .tracks
            .iter()
            .flat_map(|track| &track.notes)
            .all(|note| note.bend.is_none()));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn experimental_beat_tracking_is_opt_in_and_rejects_unsupported_combinations() {
        let audio = make_test_wav();
        let midi = audio.with_extension("mid");
        let gp5 = audio.with_extension("gp5");
        let parse = |extra: Vec<&str>| {
            let mut argv = vec![
                "mtab",
                "transcribe",
                audio.to_str().unwrap(),
                "--experimental-beat-tracking",
            ];
            argv.extend(extra);
            Cli::try_parse_from(argv)
        };
        assert!(parse(vec!["--bpm", "120"]).is_err());
        assert!(parse(vec!["--beat-times", "beats.json"]).is_err());
        let Some(Cmd::Transcribe(args)) = parse(vec!["--format", "jsonl"]).unwrap().command else {
            panic!()
        };
        assert!(validate_transcribe(&args).is_err());
        let Some(Cmd::Transcribe(args)) = parse(vec!["-o", gp5.to_str().unwrap()]).unwrap().command
        else {
            panic!()
        };
        assert!(validate_transcribe(&args).is_err());
        let Some(Cmd::Transcribe(args)) =
            parse(vec!["-o", midi.to_str().unwrap()]).unwrap().command
        else {
            panic!()
        };
        assert!(validate_transcribe(&args).is_ok());
        let Some(Cmd::Transcribe(default_args)) = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "-o",
            midi.to_str().unwrap(),
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        assert!(!default_args.experimental_beat_tracking);
        std::fs::remove_file(audio).unwrap();
    }

    #[test]
    fn experimental_beat_tracker_silence_does_not_overwrite_output() {
        let audio = make_test_wav();
        let midi = audio.with_extension("mid");
        std::fs::write(&midi, b"original").unwrap();
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mtab-core/tests/fixtures/model/tiny.fixture");
        let model = LmModel::load(LmConfig::fixture_tiny(), &model_path).unwrap();
        let Some(Cmd::Transcribe(args)) = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--experimental-beat-tracking",
            "-o",
            midi.to_str().unwrap(),
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        let tempo = validate_transcribe(&args).unwrap();
        assert!(transcribe_with_model(&args, &model, &tempo)
            .unwrap_err()
            .to_string()
            .contains("fewer than two beats"));
        assert_eq!(std::fs::read(&midi).unwrap(), b"original");
        for path in [audio, midi] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn experimental_beat_tracker_flows_through_synthetic_transcription() {
        let audio = make_test_wav();
        let midi = audio.with_extension("mid");
        let ir = audio.with_extension("ir.json");
        let n = 16_000u32 * 5;
        let mut bytes = std::fs::read(&audio).unwrap();
        bytes.truncate(44);
        bytes[4..8].copy_from_slice(&(36 + n * 2).to_le_bytes());
        bytes[40..44].copy_from_slice(&(n * 2).to_le_bytes());
        for i in 0..n {
            let pulse = i % 8000;
            let value: i16 = if pulse < 120 {
                if pulse % 2 == 0 {
                    20000
                } else {
                    -20000
                }
            } else {
                0
            };
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&audio, bytes).unwrap();
        let data = mtab_audio::load_model_input(&audio).unwrap();
        let hypothesis = mtab_audio::automatic_beat::automatic_beats(&data)
            .unwrap()
            .unwrap();
        let expected = TempoMap::from_offset_quarter_beat_times(&hypothesis.path.times).unwrap();
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mtab-core/tests/fixtures/model/tiny.fixture");
        let model = LmModel::load(LmConfig::fixture_tiny(), &model_path).unwrap();
        let Some(Cmd::Transcribe(args)) = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--experimental-beat-tracking",
            "--tuning",
            "standard",
            "--capo",
            "0",
            "--max-gen-len",
            "2",
            "--instruments",
            "guitar",
            "--emit-ir",
            ir.to_str().unwrap(),
            "-o",
            midi.to_str().unwrap(),
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        let tempo = validate_transcribe(&args).unwrap();
        transcribe_with_model(&args, &model, &tempo).unwrap();
        let doc = TabDocument::from_json(&std::fs::read_to_string(&ir).unwrap()).unwrap();
        assert_eq!(doc.meta.tempo_map.changes.len(), expected.changes.len());
        for (a, b) in doc.meta.tempo_map.changes.iter().zip(&expected.changes) {
            assert!((a.time - b.time).abs() < 1e-12 && (a.bpm - b.bpm).abs() < 1e-10);
        }
        let bytes = std::fs::read(&midi).unwrap();
        let smf = midly::Smf::parse(&bytes).unwrap();
        assert_eq!(
            smf.tracks[0]
                .iter()
                .filter(|e| matches!(
                    e.kind,
                    midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(_))
                ))
                .count(),
            expected.changes.len()
        );
        // The tiny model emits no notes; independently validate playback with
        // supplied synthetic notes through the same auto-tempo document path.
        let notes =
            [(0.2, 0.35, 40), (2.1, 2.35, 45)].map(|(start, end, midi)| mtab_core::NoteEvent {
                start,
                end,
                midi,
                velocity: 90,
                instrument: 24,
            });
        let (with_notes, omitted) = build_midi_document(&args, &data, &notes, &expected).unwrap();
        assert!(omitted.is_empty());
        write_document_outputs(&args, &with_notes, None).unwrap();
        if let Some(python) = std::env::var_os("MTAB_TEMPO_PYTHON") {
            let output = std::process::Command::new(python)
                .arg(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../tools/check_midi_auto_beat.py"),
                )
                .arg(&midi)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        for path in [audio, midi, ir] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn synthetic_model_transcription_uses_supplied_beats() {
        let audio = make_test_wav();
        let beats = audio.with_extension("beats.json");
        let midi = audio.with_extension("mid");
        let ir = audio.with_extension("ir.json");
        std::fs::write(&beats, "[0, 0.5, 1.1, 1.8, 2.3]").unwrap();
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mtab-core/tests/fixtures/model/tiny.fixture");
        let model = LmModel::load(LmConfig::fixture_tiny(), &model_path).unwrap();
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--tuning",
            "standard",
            "--capo",
            "0",
            "--max-gen-len",
            "2",
            "--instruments",
            "guitar",
            "--beat-times",
            beats.to_str().unwrap(),
            "--emit-ir",
            ir.to_str().unwrap(),
            "-o",
            midi.to_str().unwrap(),
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        let tempo = validate_transcribe(&args).unwrap();
        transcribe_with_model(&args, &model, &tempo).unwrap();
        let bytes = std::fs::read(&midi).unwrap();
        let smf = midly::Smf::parse(&bytes).unwrap();
        let mut tick = 0;
        let mut tempos = Vec::new();
        for event in &smf.tracks[0] {
            tick += event.delta.as_int();
            if let midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(us)) = event.kind {
                tempos.push((tick, us.as_int()));
            }
        }
        assert_eq!(
            tempos,
            [
                (0, 500_000),
                (480, 600_000),
                (960, 700_000),
                (1440, 500_000)
            ]
        );
        let doc = TabDocument::from_json(&std::fs::read_to_string(&ir).unwrap()).unwrap();
        assert_eq!(doc.meta.tempo_map.changes.len(), 4);
        for path in [audio, beats, midi, ir] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn transcribe_rejects_hard_links_to_inputs_and_between_outputs() {
        let audio = make_test_wav();
        let beats = audio.with_extension("beats.json");
        let alias = audio.with_extension("alias.json");
        let midi = audio.with_extension("mid");
        let midi_alias = audio.with_extension("second.mid");
        std::fs::write(&beats, "[0, 0.5, 1]").unwrap();
        std::fs::hard_link(&beats, &alias).unwrap();
        let parse = || {
            let cli = Cli::try_parse_from([
                "mtab",
                "transcribe",
                audio.to_str().unwrap(),
                "--beat-times",
                beats.to_str().unwrap(),
                "--emit-ir",
                alias.to_str().unwrap(),
                "-o",
                midi.to_str().unwrap(),
            ])
            .unwrap();
            let Some(Cmd::Transcribe(args)) = cli.command else {
                panic!("wrong command")
            };
            args
        };
        assert!(validate_transcribe(&parse())
            .unwrap_err()
            .to_string()
            .contains("hard-link"));
        assert_eq!(std::fs::read_to_string(&beats).unwrap(), "[0, 0.5, 1]");
        let mut audio_alias = parse();
        std::fs::remove_file(&alias).unwrap();
        std::fs::hard_link(&audio, &alias).unwrap();
        assert!(validate_transcribe(&audio_alias)
            .unwrap_err()
            .to_string()
            .contains("hard-link"));
        assert!(std::fs::metadata(&audio).unwrap().len() > 0);
        let weights = audio.with_extension("safetensors");
        std::fs::write(&weights, b"dummy weights").unwrap();
        std::fs::remove_file(&alias).unwrap();
        std::fs::hard_link(&weights, &alias).unwrap();
        audio_alias.weights = Some(weights.to_str().unwrap().into());
        assert!(validate_transcribe(&audio_alias)
            .unwrap_err()
            .to_string()
            .contains("hard-link"));
        assert_eq!(std::fs::read(&weights).unwrap(), b"dummy weights");
        audio_alias.emit_ir = None;
        std::fs::write(&midi, b"old output").unwrap();
        std::fs::hard_link(&midi, &midi_alias).unwrap();
        audio_alias.outputs.push(midi_alias.clone());
        assert!(validate_transcribe(&audio_alias)
            .unwrap_err()
            .to_string()
            .contains("hard-link"));
        assert_eq!(std::fs::read(&midi).unwrap(), b"old output");
        for path in [audio, beats, weights, alias, midi, midi_alias] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn supplied_beats_preflight_rejects_invalid_or_overwritten_inputs() {
        let audio = make_test_wav();
        let beats = audio.with_extension("beats.json");
        let midi = audio.with_extension("mid");
        let xml = audio.with_extension("musicxml");
        let parse = |output: &std::path::Path| {
            let cli = Cli::try_parse_from([
                "mtab",
                "transcribe",
                audio.to_str().unwrap(),
                "--beat-times",
                beats.to_str().unwrap(),
                "-o",
                output.to_str().unwrap(),
            ])
            .unwrap();
            let Some(Cmd::Transcribe(args)) = cli.command else {
                panic!("wrong command")
            };
            args
        };
        for content in [
            "{}",
            "[0]",
            "[-1, 2]",
            "[0, 0.5, 0.5]",
            "[0, 17]",
            "[0, null]",
        ] {
            std::fs::write(&beats, content).unwrap();
            let args = parse(&midi);
            assert!(validate_transcribe(&args).is_err(), "{content}");
            assert!(
                transcribe(args).unwrap_err().to_string().contains("tempo"),
                "{content}"
            );
        }
        std::fs::write(&beats, vec![b' '; 2 * 1024 * 1024 + 1]).unwrap();
        assert!(validate_transcribe(&parse(&midi))
            .unwrap_err()
            .to_string()
            .contains("2 MiB"));
        std::fs::write(&beats, "[0, 0.5, 1]").unwrap();
        assert!(validate_transcribe(&parse(&xml)).is_err());
        let mut mixed = parse(&midi);
        mixed.outputs.push(xml.clone());
        assert!(validate_transcribe(&mixed)
            .unwrap_err()
            .to_string()
            .contains("MIDI outputs only"));
        assert!(validate_transcribe(&parse(&beats)).is_err());
        let mut alias = parse(&midi);
        alias.emit_ir = Some(beats.clone());
        assert!(validate_transcribe(&alias).is_err());
        alias.emit_ir = None;
        alias.partial_report = Some(beats.clone());
        assert!(validate_transcribe(&alias).is_err());
        let jsonl = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--format",
            "jsonl",
            "--beat-times",
            beats.to_str().unwrap(),
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = jsonl.command else {
            panic!("wrong command")
        };
        assert!(validate_transcribe(&args).is_err());
        assert!(Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--beat-times",
            beats.to_str().unwrap(),
            "--bpm",
            "120",
            "-o",
            midi.to_str().unwrap()
        ])
        .is_err());
        assert_eq!(std::fs::read_to_string(&beats).unwrap(), "[0, 0.5, 1]");
        assert!(!midi.exists() && !xml.exists());
        std::fs::remove_file(&beats).unwrap();
        assert!(validate_transcribe(&parse(&midi)).is_err());
        std::fs::remove_file(audio).unwrap();
    }

    #[test]
    fn tempo_override_is_bounded_and_rejected_for_raw_jsonl() {
        let audio = make_test_wav();
        for (bpm, valid) in [("3", false), ("4", true), ("255", true), ("256", false)] {
            let cli = Cli::try_parse_from([
                "mtab",
                "transcribe",
                audio.to_str().unwrap(),
                "--bpm",
                bpm,
                "-o",
                "output.mid",
            ])
            .unwrap();
            let Some(Cmd::Transcribe(args)) = cli.command else {
                panic!("wrong command")
            };
            assert_eq!(validate_transcribe(&args).is_ok(), valid, "bpm={bpm}");
        }
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--bpm",
            "96",
            "--format",
            "jsonl",
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        assert!(validate_transcribe(&args).is_err());
        assert!(Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--bpm",
            "96.5",
            "-o",
            "x.mid"
        ])
        .is_err());
        std::fs::remove_file(audio).unwrap();
    }

    #[test]
    fn midi_document_preserves_notes_and_tuning() {
        let audio = make_test_wav();
        let output = audio.with_extension("mid");
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--tuning",
            "drop-d",
            "--capo",
            "2",
            "--grid",
            "off",
            "-o",
            output.to_str().unwrap(),
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        validate_transcribe(&args).unwrap();
        let note = mtab_core::NoteEvent {
            start: 0.5,
            end: 1.0,
            midi: 40,
            velocity: 89,
            instrument: 24,
        };
        let audio_data = mtab_audio::AudioBuffer {
            samples: vec![],
            sample_rate: 16000,
            channels: 1,
        };
        let doc = build_midi_document(&args, &audio_data, &[note], &resolve_tempo(&args).unwrap())
            .unwrap()
            .0;
        assert_eq!(doc.tracks.len(), 1);
        assert_eq!(doc.tracks[0].notes[0].midi, 40);
        assert_eq!(doc.tracks[0].notes[0].velocity, 89);
        assert_eq!(doc.tracks[0].tuning.name, "drop-d");
        assert_eq!(doc.tracks[0].capo.position(), 2);
        assert_eq!(
            (doc.tracks[0].notes[0].string, doc.tracks[0].notes[0].fret),
            (0, 0)
        );
        let data = mtab_export::midi::midi_bytes(&doc).unwrap();
        assert_eq!(&data[..4], b"MThd");
        std::fs::remove_file(audio).unwrap();
    }

    #[test]
    fn rejects_ir_and_midi_output_aliases_before_writing() {
        let audio = make_test_wav();
        let output = audio.with_extension("mid");
        let alias = output
            .parent()
            .unwrap()
            .join(".")
            .join(output.file_name().unwrap());
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--emit-ir",
            alias.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        assert!(validate_transcribe(&args).is_err());
        std::fs::remove_file(audio).unwrap();
    }
    #[test]
    fn partial_report_is_explicit_and_records_original_indices() {
        let audio = make_test_wav();
        let midi = audio.with_extension("mid");
        let report = audio.with_extension("omissions.json");
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--tuning",
            "standard",
            "--capo",
            "0",
            "--partial-report",
            report.to_str().unwrap(),
            "-o",
            midi.to_str().unwrap(),
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        validate_transcribe(&args).unwrap();
        let data = mtab_audio::AudioBuffer {
            samples: vec![],
            sample_rate: 16_000,
            channels: 1,
        };
        let event = |midi, instrument| mtab_core::NoteEvent {
            start: 0.0,
            end: 1.0,
            midi,
            velocity: 90,
            instrument,
        };
        let notes = [event(20, 27), event(40, 24), event(64, 27), event(40, 24)];
        let (doc, omissions) =
            build_midi_document(&args, &data, &notes, &resolve_tempo(&args).unwrap()).unwrap();
        let first = partial_report_bytes(notes.len(), &omissions).unwrap();
        assert_eq!(
            first,
            partial_report_bytes(notes.len(), &omissions).unwrap()
        );
        let parsed: serde_json::Value = serde_json::from_slice(&first).unwrap();
        assert_eq!(parsed["input_note_count"], 4);
        assert_eq!(parsed["assigned_note_count"], 2);
        assert_eq!(parsed["unassigned"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["unassigned"][0]["input_index"], 0);
        assert_eq!(parsed["unassigned"][0]["note"]["midi"], 20);
        let duplicate_index = parsed["unassigned"][1]["input_index"].as_u64().unwrap() as usize;
        assert!(duplicate_index == 1 || duplicate_index == 3);
        assert_eq!(parsed["unassigned"][1]["note"]["instrument"], 24);
        assert_eq!(doc.tracks.iter().map(|t| t.notes.len()).sum::<usize>(), 2);
        write_document_outputs(&args, &doc, Some(&first)).unwrap();
        assert_eq!(std::fs::read(&report).unwrap(), first);
        assert_eq!(&std::fs::read(&midi).unwrap()[..4], b"MThd");
        std::fs::remove_file(audio).unwrap();
        std::fs::remove_file(midi).unwrap();
        std::fs::remove_file(report).unwrap();
    }

    #[test]
    fn partial_report_refuses_aliases_jsonl_and_wide_stretch() {
        let audio = make_test_wav();
        let midi = audio.with_extension("mid");
        let make = |report: &std::path::Path, format: &str| {
            let cli = Cli::try_parse_from([
                "mtab",
                "transcribe",
                audio.to_str().unwrap(),
                "--format",
                format,
                "--partial-report",
                report.to_str().unwrap(),
                "-o",
                midi.to_str().unwrap(),
            ])
            .unwrap();
            let Some(Cmd::Transcribe(args)) = cli.command else {
                panic!("wrong command")
            };
            args
        };
        assert!(validate_transcribe(&make(&audio, "midi")).is_err());
        assert!(validate_transcribe(&make(&midi, "midi")).is_err());
        assert!(
            validate_transcribe(&make(&audio.with_extension("omissions.json"), "jsonl")).is_err()
        );
        assert!(Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--allow-wide-stretch",
            "--partial-report",
            "report.json",
            "-o",
            "song.mid"
        ])
        .is_err());
        std::fs::remove_file(audio).unwrap();
    }

    #[test]
    fn dynamics_flag_is_opt_in() {
        for (extra, expected) in [(vec![], false), (vec!["--dynamics"], true)] {
            let mut argv = vec!["mtab", "transcribe", "song.wav", "--format", "jsonl"];
            argv.extend(extra);
            let cli = Cli::try_parse_from(argv).unwrap();
            let Some(Cmd::Transcribe(args)) = cli.command else {
                panic!("wrong command")
            };
            assert_eq!(args.dynamics, expected);
        }
    }
    fn isolated_pitch_sequence(pitches: &[u8]) -> mtab_audio::AudioBuffer {
        let mut samples = Vec::new();
        for &pitch in pitches {
            let hz = 440.0 * 2.0f64.powf((f64::from(pitch) - 69.0) / 12.0);
            for i in 0..8000 {
                let sample = if i < 7200 {
                    (0.3 * (2.0 * std::f64::consts::PI * hz * i as f64 / 16000.0).sin()) as f32
                } else {
                    0.0
                };
                samples.push(sample);
            }
        }
        mtab_audio::AudioBuffer {
            samples,
            sample_rate: 16000,
            channels: 1,
        }
    }

    fn args_for_tuning(extra: &[&str]) -> TranscribeArgs {
        let mut argv = vec!["mtab", "transcribe", "song.wav", "-o", "song.mid"];
        argv.extend_from_slice(extra);
        let Some(Cmd::Transcribe(args)) = Cli::try_parse_from(argv).unwrap().command else {
            panic!("wrong command")
        };
        args
    }

    #[test]
    fn tuning_pipeline_abstains_on_shared_standard_strings() {
        let audio = isolated_pitch_sequence(&[40, 45, 50, 55, 59, 64]);
        let histogram = mtab_audio::pitch_histogram::pitch_histogram(&audio).unwrap();
        let detection =
            mtab_guitar::detect_tuning(&histogram.weights, &mtab_guitar::standard_candidates())
                .unwrap();
        assert!(!detection.sufficient_evidence, "{detection:?}");
        let (tuning, capo) = resolve_tuning_and_capo(&args_for_tuning(&[]), &audio).unwrap();
        assert_eq!(tuning, Tuning::standard());
        assert_eq!(capo, Capo::NONE);
    }

    #[test]
    fn tuning_pipeline_can_detect_capo_with_explicit_tuning() {
        let audio = isolated_pitch_sequence(&[42, 47, 52, 57, 61, 66]);
        let args = args_for_tuning(&["--tuning", "standard"]);
        let (tuning, capo) = resolve_tuning_and_capo(&args, &audio).unwrap();
        assert_eq!(tuning, Tuning::standard());
        assert_eq!(capo.position(), 2);
    }

    #[test]
    fn tuning_pipeline_can_identify_synthetic_eight_string_open_notes() {
        let audio = isolated_pitch_sequence(&[30, 35, 40, 45, 50, 55, 59, 64]);
        let (tuning, capo) = resolve_tuning_and_capo(&args_for_tuning(&[]), &audio).unwrap();
        assert_eq!(tuning, Tuning::eight_string());
        assert_eq!(capo, Capo::NONE);
    }

    #[test]
    fn tuning_fallback_preserves_explicit_overrides() {
        let audio = mtab_audio::AudioBuffer {
            samples: vec![0.0; 4096],
            sample_rate: 16000,
            channels: 1,
        };
        let (tuning, capo) =
            resolve_tuning_and_capo(&args_for_tuning(&["--tuning", "drop-d"]), &audio).unwrap();
        assert_eq!(tuning, Tuning::drop_d());
        assert_eq!(capo, Capo::NONE);
        let (tuning, capo) =
            resolve_tuning_and_capo(&args_for_tuning(&["--capo", "2"]), &audio).unwrap();
        assert_eq!(tuning, Tuning::standard());
        assert_eq!(capo.position(), 2);
    }
    #[test]
    fn wider_stretch_is_opt_in_and_preserves_all_notes() {
        let audio = mtab_audio::AudioBuffer {
            samples: vec![],
            sample_rate: 16000,
            channels: 1,
        };
        let note = mtab_core::NoteEvent {
            start: 0.0,
            end: 1.0,
            midi: 52,
            velocity: 100,
            instrument: 24,
        };
        let strict = args_for_tuning(&["--tuning", "standard", "--capo", "0"]);
        assert!(build_midi_document(
            &strict,
            &audio,
            &[note, note],
            &resolve_tempo(&strict).unwrap()
        )
        .is_err());
        let wide = args_for_tuning(&[
            "--tuning",
            "standard",
            "--capo",
            "0",
            "--allow-wide-stretch",
        ]);
        let doc = build_midi_document(&wide, &audio, &[note, note], &resolve_tempo(&wide).unwrap())
            .unwrap()
            .0;
        assert_eq!(doc.tracks[0].notes.len(), 2);
        assert_ne!(doc.tracks[0].notes[0].string, doc.tracks[0].notes[1].string);
        assert!(mtab_export::midi::midi_bytes(&doc).is_ok());
    }
    #[test]
    fn mixed_midi_gp5_outputs_and_format_conflicts() {
        let audio = make_test_wav();
        let midi = audio.with_extension("mid");
        let gp5 = audio.with_extension("gp5");
        let ir = audio.with_extension("json");
        let xml = audio.with_extension("musicxml");
        let ascii = audio.with_extension("tab");
        let outputs = format!(
            "{},{},{},{}",
            midi.display(),
            gp5.display(),
            xml.display(),
            ascii.display()
        );
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--tuning",
            "standard",
            "--capo",
            "0",
            "--grid",
            "off",
            "--emit-ir",
            ir.to_str().unwrap(),
            "-o",
            &outputs,
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        validate_transcribe(&args).unwrap();
        let event = mtab_core::NoteEvent {
            start: 0.5,
            end: 1.0,
            midi: 40,
            velocity: 80,
            instrument: 24,
        };
        let audio_data = mtab_audio::AudioBuffer {
            samples: vec![],
            sample_rate: 16000,
            channels: 1,
        };
        let doc = build_midi_document(&args, &audio_data, &[event], &resolve_tempo(&args).unwrap())
            .unwrap()
            .0;
        write_document_outputs(&args, &doc, None).unwrap();
        assert!(std::fs::read(&midi).unwrap().starts_with(b"MThd"));
        assert!(std::fs::read(&gp5)
            .unwrap()
            .starts_with(b"\x18FICHIER GUITAR PRO v5.00"));
        if let Some(python) = std::env::var_os("MTAB_PYGUITARPRO_PYTHON") {
            let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tools/check_gp5_cli_fixture.py");
            let check = std::process::Command::new(python)
                .arg(script)
                .arg(&gp5)
                .output()
                .unwrap();
            assert!(
                check.status.success(),
                "independent GP5 parser failed: {}",
                String::from_utf8_lossy(&check.stderr)
            );
        }
        if let Some(python) = std::env::var_os("MTAB_XML_PYTHON") {
            let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tools/check_musicxml_schema.py");
            let output = std::process::Command::new(python)
                .arg(script)
                .arg(&xml)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let xml_content = std::fs::read_to_string(&xml).unwrap();
        assert!(xml_content.starts_with("<?xml"));
        assert!(xml_content.contains("<string>6</string><fret>0</fret>"));
        let ascii_content = std::fs::read_to_string(&ascii).unwrap();
        assert!(ascii_content.contains("mtab ASCII tab | 120 BPM"));
        assert!(ascii_content.contains(" 40 |--0-"));
        for (format, path) in [
            ("midi", &midi),
            ("gp5", &gp5),
            ("musicxml", &xml),
            ("ascii", &ascii),
        ] {
            let cli = Cli::try_parse_from([
                "mtab",
                "transcribe",
                audio.to_str().unwrap(),
                "--format",
                format,
                "-o",
                path.to_str().unwrap(),
            ])
            .unwrap();
            let Some(Cmd::Transcribe(args)) = cli.command else {
                panic!("wrong command")
            };
            validate_transcribe(&args).unwrap();
        }
        if let Some(python) = std::env::var_os("MTAB_PYGUITARPRO_PYTHON") {
            let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tools/check_gp5_cli_ir.py");
            let output = std::process::Command::new(python)
                .arg(script)
                .arg(&gp5)
                .arg(&ir)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let ir_doc: TabDocument = serde_json::from_slice(&std::fs::read(&ir).unwrap()).unwrap();
        assert_eq!(ir_doc, doc);
        let mut mismatched = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--format",
            "midi",
            "-o",
            &outputs,
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = mismatched.command.take() else {
            panic!("wrong command")
        };
        assert!(validate_transcribe(&args).is_err());
        for file in [audio, midi, gp5, ir, xml, ascii] {
            std::fs::remove_file(file).unwrap();
        }
    }

    #[test]
    fn gp5_validation_failure_does_not_write_other_outputs() {
        let audio = make_test_wav();
        let midi = audio.with_extension("mid");
        let gp5 = audio.with_extension("gp5");
        let outputs = format!("{},{}", midi.display(), gp5.display());
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--tuning",
            "standard",
            "--capo",
            "0",
            "-o",
            &outputs,
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        validate_transcribe(&args).unwrap();
        let event = mtab_core::NoteEvent {
            start: 0.0,
            end: 0.5,
            midi: 40,
            velocity: 80,
            instrument: 24,
        };
        let audio_data = mtab_audio::AudioBuffer {
            samples: vec![],
            sample_rate: 16000,
            channels: 1,
        };
        let mut doc =
            build_midi_document(&args, &audio_data, &[event], &resolve_tempo(&args).unwrap())
                .unwrap()
                .0;
        doc.tracks[0].notes[0]
            .technique
            .push(mtab_guitar::Technique::HammerOn);
        std::fs::write(&midi, b"do not overwrite").unwrap();
        assert!(write_document_outputs(&args, &doc, None).is_err());
        assert_eq!(std::fs::read(&midi).unwrap(), b"do not overwrite");
        assert!(!gp5.exists());
        std::fs::remove_file(audio).unwrap();
        std::fs::remove_file(midi).unwrap();
    }
    #[test]
    fn gp5_ir_reconstruction_handles_ties_and_quantized_notes() {
        let audio = make_test_wav();
        let gp5 = audio.with_extension("gp5");
        let ir = audio.with_extension("json");
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--tuning",
            "standard",
            "--capo",
            "0",
            "--emit-ir",
            ir.to_str().unwrap(),
            "-o",
            gp5.to_str().unwrap(),
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        validate_transcribe(&args).unwrap();
        let events = [
            mtab_core::NoteEvent {
                start: 0.0,
                end: 2.5,
                midi: 40,
                velocity: 80,
                instrument: 24,
            },
            mtab_core::NoteEvent {
                start: 0.51,
                end: 1.01,
                midi: 45,
                velocity: 100,
                instrument: 24,
            },
            mtab_core::NoteEvent {
                start: 2.01,
                end: 2.05,
                midi: 50,
                velocity: 127,
                instrument: 24,
            },
        ];
        let audio_data = mtab_audio::AudioBuffer {
            samples: vec![],
            sample_rate: 16000,
            channels: 1,
        };
        let doc = build_midi_document(&args, &audio_data, &events, &resolve_tempo(&args).unwrap())
            .unwrap()
            .0;
        let (_, plan) = mtab_export::gp5::gp5_bytes(&doc).unwrap();
        assert!(plan.tracks[0]
            .bars
            .iter()
            .flatten()
            .flat_map(|b| &b.notes)
            .any(|n| n.tied));
        assert!(plan.max_boundary_shift_seconds > 0.0);
        write_document_outputs(&args, &doc, None).unwrap();
        if let Some(python) = std::env::var_os("MTAB_PYGUITARPRO_PYTHON") {
            let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tools/check_gp5_cli_ir.py");
            let output = std::process::Command::new(python)
                .arg(script)
                .arg(&gp5)
                .arg(&ir)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        for path in [audio, gp5, ir] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn musicxml_rejection_does_not_overwrite_any_mixed_output() {
        let audio = make_test_wav();
        let midi = audio.with_extension("mid");
        let xml = audio.with_extension("xml");
        let ascii = audio.with_extension("txt");
        let outputs = format!("{},{},{}", midi.display(), xml.display(), ascii.display());
        let cli = Cli::try_parse_from([
            "mtab",
            "transcribe",
            audio.to_str().unwrap(),
            "--tuning",
            "standard",
            "--capo",
            "0",
            "-o",
            &outputs,
        ])
        .unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        validate_transcribe(&args).unwrap();
        let event = mtab_core::NoteEvent {
            start: 0.0,
            end: 0.5,
            midi: 40,
            velocity: 80,
            instrument: 24,
        };
        let audio_data = mtab_audio::AudioBuffer {
            samples: vec![],
            sample_rate: 16000,
            channels: 1,
        };
        let mut doc =
            build_midi_document(&args, &audio_data, &[event], &resolve_tempo(&args).unwrap())
                .unwrap()
                .0;
        doc.tracks[0].notes[0].bend = Some(mtab_guitar::PitchBend {
            start: 0.0,
            peak: 0.25,
            depth_cents: 100.0,
            kind: mtab_guitar::BendKind::Bend,
        });
        for path in [&midi, &xml, &ascii] {
            std::fs::write(path, b"original").unwrap();
        }
        assert!(write_document_outputs(&args, &doc, None).is_err());
        for path in [&midi, &xml, &ascii] {
            assert_eq!(std::fs::read(path).unwrap(), b"original");
            std::fs::remove_file(path).unwrap();
        }
        std::fs::remove_file(audio).unwrap();
    }
    #[test]
    fn unspecified_format_without_outputs_defaults_to_jsonl() {
        let audio = make_test_wav();
        let cli = Cli::try_parse_from(["mtab", "transcribe", audio.to_str().unwrap()]).unwrap();
        let Some(Cmd::Transcribe(args)) = cli.command else {
            panic!("wrong command")
        };
        assert_eq!(output_mode(&args).unwrap(), OutputMode::Jsonl);
        validate_transcribe(&args).unwrap();
        std::fs::remove_file(audio).unwrap();
    }
}
