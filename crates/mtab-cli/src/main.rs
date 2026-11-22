//! `mtab` CLI (PROMPT.md §6.12) — thin shell over the library crates.
//!
//! Local-weight transcription supports raw JSONL, preliminary MIDI and a
//! restricted experimental GP5 subset. Other exports are under development.

mod guitarset;
mod guitarset_notes;

use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

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
        /// Path to a local GuitarSet .jams file or directory of .jams files
        #[arg(long)]
        annotations: Option<PathBuf>,
        /// Evaluate string assignment on ground-truth notes (not audio transcription)
        #[arg(long, conflicts_with = "predictions")]
        oracle_notes: bool,
        /// Score a JSONL note transcription against one matching GuitarSet JAMS file
        #[arg(long, conflicts_with = "oracle_notes")]
        predictions: Option<PathBuf>,
        /// Limit both note sets to this many seconds, clipping note offsets
        #[arg(long, requires = "predictions")]
        clip_duration: Option<f64>,
        /// Write a markdown report to this path (default: stdout)
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

    /// Retry impossible fingerings with up to seven frets of span (review for comfort)
    #[arg(long)]
    allow_wide_stretch: bool,

    /// Isolate the guitar stem before transcription (Demucs)
    #[arg(long)]
    separate: bool,

    /// Estimate relative velocities from audio (experimental; mix energy, not isolated notes)
    #[arg(long)]
    dynamics: bool,

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
        Some(Cmd::Download { size, token_env }) => download(size, &token_env),
        Some(Cmd::Eval {
            corpus,
            annotations,
            oracle_notes,
            predictions,
            clip_duration,
            report,
        }) => eval(
            &corpus,
            annotations.as_deref(),
            oracle_notes,
            predictions.as_deref(),
            clip_duration,
            report.as_deref(),
        ),
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

fn download(size: ModelSize, token_env: &str) -> Result<()> {
    let _ = std::env::var(token_env).map_err(|_| MtabError::LicenseGateRequired {
        repo: size.hf_repo().to_owned(),
    })?;
    Err(MtabError::NotImplemented(
        "weights download helper (milestone M3)",
    ))
}

fn eval(
    corpus: &str,
    annotations: Option<&std::path::Path>,
    oracle_notes: bool,
    predictions: Option<&std::path::Path>,
    clip_duration: Option<f64>,
    report: Option<&std::path::Path>,
) -> Result<()> {
    if corpus != "guitarset" {
        return Err(MtabError::NotImplemented(
            "only GuitarSet evaluation is supported",
        ));
    }
    if !oracle_notes && predictions.is_none() {
        return Err(MtabError::NotImplemented("automatic end-to-end corpus evaluation; provide --predictions <JSONL> or use --oracle-notes"));
    }
    let path = annotations.ok_or_else(|| {
        MtabError::InvalidModelSource(
            "provide --annotations <local GuitarSet .jams file or directory>".into(),
        )
    })?;
    if let Some(predictions) = predictions {
        guitarset_notes::evaluate(path, predictions, clip_duration, report)
    } else {
        guitarset::evaluate(path, report)
    }
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

fn validate_transcribe(args: &TranscribeArgs) -> Result<()> {
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
    let protected = std::iter::once(args.input.as_path())
        .chain(args.weights.as_deref().map(std::path::Path::new));
    let protected: Vec<PathBuf> = protected.filter_map(path_identity).collect();
    let mut outputs = std::collections::HashSet::new();
    for path in args.outputs.iter().chain(args.emit_ir.iter()) {
        if let Some(identity) = path_identity(path) {
            if protected.contains(&identity) {
                return Err(MtabError::InvalidModelSource(
                    "output must not overwrite the input audio or weights".into(),
                ));
            }
            if !outputs.insert(identity) {
                return Err(MtabError::InvalidModelSource(
                    "output paths must be distinct".into(),
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
    if args.separate || args.techniques || args.chords {
        return Err(MtabError::NotImplemented(
            "separation, techniques and chords (future milestones)",
        ));
    }
    if mode == OutputMode::Jsonl {
        if args.emit_ir.is_some()
            || args.tuning != "auto"
            || args.capo != "auto"
            || args.frets != 22
            || args.allow_wide_stretch
            || args.grid != Grid::SIXTEENTH
        {
            return Err(MtabError::NotImplemented("tuning, capo, fretting, rhythm and TabDocument IR are not applied to raw JSONL notes"));
        }
    } else {
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
    Ok(())
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
            let doc = build_midi_document(args, &audio, &result.notes)?;
            write_document_outputs(args, &doc)
        }
    }
}

/// Render all requested formats before writing any, so an unsupported score
/// cannot leave a misleading partial MIDI or notation output behind. I/O errors
/// during the subsequent writes can still leave earlier outputs on disk.
fn write_document_outputs(args: &TranscribeArgs, doc: &TabDocument) -> Result<()> {
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
        eprintln!("warning: experimental GP5 uses fixed 120 BPM, 4/4 and 1/32 quantization regardless of --grid; maximum timing shift {:.1} ms; review the score", plan.max_boundary_shift_seconds * 1000.0);
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
        eprintln!("warning: experimental MusicXML uses fixed 120 BPM, 4/4 and 1/32 quantization regardless of --grid; maximum timing shift {:.1} ms; velocities omitted and duplicated staves may double playback", plan.max_boundary_shift_seconds * 1000.0);
        Some(text.into_bytes())
    } else {
        None
    };
    let ascii = if wants_ascii {
        let (text, plan) = mtab_export::ascii::ascii_tab(doc)?;
        eprintln!("warning: experimental ASCII tab uses fixed 120 BPM, 4/4 and 1/32 quantization regardless of --grid; maximum timing shift {:.1} ms; velocities omitted", plan.max_boundary_shift_seconds * 1000.0);
        Some(text.into_bytes())
    } else {
        None
    };
    let ir = args
        .emit_ir
        .as_ref()
        .map(|_| serde_json::to_vec_pretty(doc))
        .transpose()?;
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
) -> Result<TabDocument> {
    let (tuning, capo) = resolve_tuning_and_capo(args, audio)?;
    eprintln!("warning: beat tracking and quantization are not implemented; MIDI uses raw timing and a 120 BPM, 4/4 tempo map");
    let mut grouped: std::collections::BTreeMap<u16, Vec<mtab_core::NoteEvent>> =
        std::collections::BTreeMap::new();
    for note in notes {
        grouped.entry(note.instrument).or_default().push(*note);
    }
    let mut tracks = Vec::new();
    for (program, track_notes) in grouped {
        let options = FingeringOptions {
            max_fret: args.frets,
            ..FingeringOptions::default()
        };
        let assigned = if args.allow_wide_stretch {
            let assignment =
                mtab_guitar::assign_notes_with_stretch(&track_notes, &tuning, capo, options, 7)?;
            if assignment.used_wider_span {
                eprintln!("warning: track program {program} required a wider fingering search (up to 7 frets); review physical comfort");
            }
            assignment.notes
        } else {
            mtab_guitar::assign_notes(&track_notes, &tuning, capo, options)?
        };
        tracks.push(GuitarTrack {
            tuning: tuning.clone(),
            capo,
            instrument: Some(mtab_core::mt3::instruments::instrument_for_program(program)),
            notes: assigned,
        });
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
    Ok(TabDocument {
        meta: TrackMeta {
            title: args
                .input
                .file_stem()
                .and_then(|v| v.to_str())
                .map(str::to_owned),
            tempo_map: TempoMap::constant(120.0),
            time_signatures: vec![(0, 4, 4)],
            frets: args.frets,
            source_hash: None,
        },
        tracks,
        chord_symbols: Vec::new(),
    })
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
        validate_transcribe(&args).unwrap();
        transcribe_with_model(&args, &model).unwrap();
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
        let doc = build_midi_document(&args, &audio_data, &[note]).unwrap();
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
        assert!(build_midi_document(&strict, &audio, &[note, note]).is_err());
        let wide = args_for_tuning(&[
            "--tuning",
            "standard",
            "--capo",
            "0",
            "--allow-wide-stretch",
        ]);
        let doc = build_midi_document(&wide, &audio, &[note, note]).unwrap();
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
        let doc = build_midi_document(&args, &audio_data, &[event]).unwrap();
        write_document_outputs(&args, &doc).unwrap();
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
        let mut doc = build_midi_document(&args, &audio_data, &[event]).unwrap();
        doc.tracks[0].notes[0]
            .technique
            .push(mtab_guitar::Technique::HammerOn);
        std::fs::write(&midi, b"do not overwrite").unwrap();
        assert!(write_document_outputs(&args, &doc).is_err());
        assert_eq!(std::fs::read(&midi).unwrap(), b"do not overwrite");
        assert!(!gp5.exists());
        std::fs::remove_file(audio).unwrap();
        std::fs::remove_file(midi).unwrap();
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
        let mut doc = build_midi_document(&args, &audio_data, &[event]).unwrap();
        doc.tracks[0].notes[0].bend = Some(mtab_guitar::PitchBend {
            start: 0.0,
            peak: 0.25,
            depth_cents: 100.0,
            kind: mtab_guitar::BendKind::Bend,
        });
        for path in [&midi, &xml, &ascii] {
            std::fs::write(path, b"original").unwrap();
        }
        assert!(write_document_outputs(&args, &doc).is_err());
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
