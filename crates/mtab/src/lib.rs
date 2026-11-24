//! `muscriptor-tab` — the library facade the CLI is a thin shell over.
//!
//! PROMPT.md §6.11 asks for an API that reads like this:
//!
//! ```ignore
//! use muscriptor_tab::{Transcriber, ModelSize, ModelSource, GuitarConfig, Tuning, Capo};
//!
//! let tr = Transcriber::builder()
//!     .size(ModelSize::Medium)
//!     .weights(ModelSource::Local("muscriptor-medium.safetensors".into()))
//!     .guitar(GuitarConfig { tuning: TuningChoice::Auto, capo: CapoChoice::Auto, frets: 22 })
//!     .build()?;
//! let tab = tr.transcribe_file("song.wav")?;
//! tab.export_guitar_pro("song.gp5")?;
//! ```
//!
//! This crate composes the four working crates (`mtab-core`, `mtab-audio`,
//! `mtab-guitar`, `mtab-export`) into that shape. It holds no logic of its own
//! that the stage crates do not already own; the point is one obvious entry
//! point, so callers do not have to re-derive the pipeline from
//! [`mtab_audio::transcribe`], [`mtab_guitar::assign_notes`] and the exporters
//! by hand.
//!
//! # What is honest about it
//!
//! - **Load the model once, transcribe many times.** [`Transcriber`] owns a
//!   loaded [`LmModel`]; `build()` is the expensive call, not the transcribe
//!   calls after it.
//! - **Tuning and capo detection is experimental** (§6.5). `Auto` is the
//!   default and it falls back to standard tuning at capo 0 when the evidence
//!   is ambiguous. Set an explicit tuning unless you have heard the result.
//! - **Rhythm is unquantized.** There is no reliable beat tracker yet
//!   (`--experimental-beat-tracking` scores 53% beat F1 on GuitarSet), so the
//!   document carries a constant tempo map — 120 BPM, 4/4 unless you set one —
//!   and note times stay the model's raw seconds. [`Grid`] is accepted and
//!   recorded so callers can pin the intended grid, but it cannot yet reshape
//!   the timing; anything else is a loud [`MtabError::NotImplemented`].
//! - **No stem separation.** Transcription and every audio analysis run on the
//!   mix, so a guitar competing with bass and drums will transcribe worse than
//!   it would from an isolated stem (§7.1).
//! - **Techniques and dynamics are not inferred here.** `Transcription::notes`
//!   carries the model's velocities; [`NoteEvent::velocity`] is upstream's
//!   fixed 100 unless you run [`mtab_audio::dynamics`] yourself. Bend and
//!   articulation inference exist as opt-in heuristics in `mtab-audio` and are
//!   not called by this facade, because they are uncalibrated.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::path::{Path, PathBuf};

use mtab_core::{MtabError, Result};

pub use mtab_core::lm::LmModel;
pub use mtab_core::model::{ModelSize, ModelSource};
pub use mtab_core::{mt3, NoteEvent};
pub use mtab_export::ExportFormat;
pub use mtab_guitar::{
    assign_notes, assign_notes_partial, assign_notes_with_stretch, suggest_chords, Capo,
    ChordSymbol, FingeringOptions, Grid, GuitarTrack, TabDocument, TabNote, TempoMap, TrackMeta,
    Tuning,
};

/// The pipeline stages reported through [`Progress`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Decoding, downmixing and resampling the audio.
    Audio,
    /// Tuning/capo detection, when either is `Auto`.
    Tuning,
    /// Chunked autoregressive generation.
    Decode,
    /// String/fret assignment per instrument track.
    Fingering,
}

/// Progress in the form a GUI can render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Progress {
    /// Which stage this event belongs to.
    pub stage: Stage,
    /// Completed units within the stage.
    pub completed: usize,
    /// Total units when the stage has a fixed length (generation chunks,
    /// instrument tracks). `None` for stages measured before the total is
    /// known.
    pub total: Option<usize>,
}

/// Where the tuning comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum TuningChoice {
    /// Experimental detection, falling back to standard tuning on ambiguous
    /// evidence (§6.5).
    Auto,
    /// Use this tuning exactly.
    Fixed(Tuning),
}

/// Where the capo position comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapoChoice {
    /// Experimental detection alongside the tuning, falling back to 0.
    Auto,
    /// Use this position exactly.
    Fixed(Capo),
}

/// What instrument to solve for.
#[derive(Debug, Clone)]
pub struct GuitarConfig {
    /// Tuning source.
    pub tuning: TuningChoice,
    /// Capo source.
    pub capo: CapoChoice,
    /// Fretboard length, 1–36.
    pub frets: u8,
    /// Retry an unplayable track with up to this span rather than failing.
    /// `None` keeps the four-fret default and errors instead; the usual
    /// reasons for failure are listed on [`MtabError::UnplayableNote`].
    pub wider_span: Option<u8>,
    /// Drop unfingered notes and record them in
    /// [`Transcription::omitted`] instead of failing. This removes notes from
    /// **every** output, MIDI included — §6.6's playability pass working as
    /// designed, but a lossy one.
    pub allow_partial: bool,
}

impl Default for GuitarConfig {
    fn default() -> Self {
        Self {
            tuning: TuningChoice::Auto,
            capo: CapoChoice::Auto,
            frets: 22,
            wider_span: None,
            allow_partial: false,
        }
    }
}

impl GuitarConfig {
    /// Standard tuning, no capo, no detection. The configuration to reach for
    /// when detection has already been verified by ear.
    #[must_use]
    pub fn fixed_standard(frets: u8) -> Self {
        Self {
            tuning: TuningChoice::Fixed(Tuning::standard()),
            capo: CapoChoice::Fixed(Capo::NONE),
            frets,
            ..Self::default()
        }
    }
}

/// Decoding settings that are not about the guitar.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeConfig {
    /// Restrict decoding to these instrument names, e.g. `["guitar"]`.
    /// Names come from [`mt3::instruments`]; an empty list disables the mask.
    pub instruments: Vec<String>,
    /// Beam width; 1 is greedy.
    pub beam_size: usize,
    /// Sampling temperature. `None` is greedy; `Some` samples and makes
    /// [`DecodeConfig::seed`] meaningful.
    pub temperature: Option<f32>,
    /// RNG seed for sampling. Greedy decoding ignores it.
    pub seed: u64,
    /// Tokens per chunk before giving up.
    pub max_gen_len: usize,
    /// Treat a chunk that hits the token limit as an error instead of a
    /// warning. Recommended: silent truncation is a wrong transcription.
    pub strict_eos: bool,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            instruments: vec!["guitar".to_owned()],
            beam_size: 1,
            temperature: None,
            seed: 0,
            max_gen_len: 2000,
            strict_eos: false,
        }
    }
}

/// The finished transcription, before it becomes files.
#[derive(Debug, Clone)]
pub struct Transcription {
    /// The tablature document: tracks, notes, tuning, tempo map.
    pub document: TabDocument,
    /// Every note the model emitted, in input order.
    pub notes: Vec<NoteEvent>,
    /// Raw notes left out of `document` because they could not be fingered.
    /// Always empty unless [`GuitarConfig::allow_partial`] was set.
    pub omitted: Vec<NoteEvent>,
    /// Chunk indices that reached the token limit without an end-of-sequence.
    pub missing_eos: Vec<usize>,
}

impl Transcription {
    /// Write a MIDI file. Byte-identical for identical input and settings —
    /// the §6.14.6 determinism requirement.
    pub fn export_midi(&self, path: impl AsRef<Path>) -> Result<()> {
        mtab_export::midi::write_midi(&self.document, path.as_ref())
    }

    /// Write a Guitar Pro 5 file.
    pub fn export_guitar_pro(&self, path: impl AsRef<Path>) -> Result<()> {
        mtab_export::gp5::write_gp5(&self.document, path.as_ref()).map(|_| ())
    }

    /// Write MusicXML 3.1 (tab staff plus standard staff).
    pub fn export_musicxml(&self, path: impl AsRef<Path>) -> Result<()> {
        mtab_export::musicxml::write_musicxml(&self.document, path.as_ref())
    }

    /// Render ASCII tablature.
    pub fn ascii_tab(&self) -> Result<String> {
        mtab_export::ascii::ascii_tab(&self.document).map(|(text, _)| text)
    }

    /// Write ASCII tablature to a file.
    pub fn export_ascii(&self, path: impl AsRef<Path>) -> Result<()> {
        mtab_export::ascii::write_ascii(&self.document, path.as_ref())
    }

    /// Serialize the in-memory IR as JSON for debugging (`--emit-ir`).
    pub fn ir_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.document)
            .map_err(|e| MtabError::InvalidTuning(format!("IR serialization failed: {e}")))
    }
}

/// Builder for [`Transcriber`]. Obtain it from [`Transcriber::builder`].
///
/// Not `Clone`: it may hold a loaded model, which is hundreds of megabytes to
/// several gigabytes.
pub struct TranscriberBuilder {
    size: ModelSize,
    weights: Option<ModelSource>,
    loaded: Option<LmModel>,
    guitar: GuitarConfig,
    decode: DecodeConfig,
    grid: Grid,
    tempo: Option<TempoMap>,
}

impl TranscriberBuilder {
    /// Model architecture. Must agree with the weights file.
    #[must_use]
    pub fn size(mut self, size: ModelSize) -> Self {
        self.size = size;
        self
    }

    /// Where the weights come from. There is no default: the weights are
    /// CC BY-NC 4.0 and gated, so the caller always supplies them.
    #[must_use]
    pub fn weights(mut self, source: ModelSource) -> Self {
        self.weights = Some(source);
        self
    }

    /// Use an already-loaded model instead of loading one.
    ///
    /// Useful when the caller owns the weights lifecycle — a server keeping one
    /// model resident, or a test with a synthetic checkpoint whose shape is not
    /// one of the three shipped sizes. Takes precedence over
    /// [`TranscriberBuilder::weights`].
    #[must_use]
    pub fn model(mut self, model: LmModel) -> Self {
        self.loaded = Some(model);
        self
    }

    /// Guitar configuration.
    #[must_use]
    pub fn guitar(mut self, guitar: GuitarConfig) -> Self {
        self.guitar = guitar;
        self
    }

    /// Decoding configuration.
    #[must_use]
    pub fn decode(mut self, decode: DecodeConfig) -> Self {
        self.decode = decode;
        self
    }

    /// Intended rhythm grid.
    ///
    /// Only [`Grid::SIXTEENTH`] (the default) and [`Grid::Off`] are accepted
    /// today; any other subdivision returns [`MtabError::NotImplemented`] from
    /// [`TranscriberBuilder::build`] instead of being silently ignored.
    #[must_use]
    pub fn grid(mut self, grid: Grid) -> Self {
        self.grid = grid;
        self
    }

    /// Force a tempo map instead of the 120 BPM / 4/4 default.
    #[must_use]
    pub fn tempo(mut self, tempo: TempoMap) -> Self {
        self.tempo = Some(tempo);
        self
    }

    /// Load the weights and return a ready transcriber.
    ///
    /// This is the expensive call — reading a 400 MB or 5.5 GB safetensors
    /// file. Do it once and reuse the [`Transcriber`].
    pub fn build(self) -> Result<Transcriber> {
        if self.guitar.frets == 0 || self.guitar.frets > 36 {
            return Err(MtabError::InvalidTuning(
                "frets must be in 1..=36".to_owned(),
            ));
        }
        if self.grid != Grid::SIXTEENTH && self.grid != Grid::Off {
            return Err(MtabError::NotImplemented(
                "rhythm quantization; only --grid 1/16 and --grid off are available",
            ));
        }
        let model = match self.loaded {
            Some(model) => model,
            None => {
                let source = self.weights.ok_or_else(|| MtabError::WeightsNotFound {
                    path: PathBuf::from("<none>"),
                })?;
                let path = resolve_source(&source, self.size)?;
                let config = mtab_core::lm::LmConfig::from_size(self.size);
                LmModel::load(config, &path)?
            }
        };
        Ok(Transcriber {
            model,
            size: self.size,
            guitar: self.guitar,
            decode: self.decode,
            tempo: self.tempo.unwrap_or_else(|| TempoMap::constant(120.0)),
        })
    }
}

fn resolve_source(source: &ModelSource, size: ModelSize) -> Result<PathBuf> {
    match source {
        ModelSource::Local(path) => Ok(path.clone()),
        ModelSource::Cached { .. } => mtab_core::model::resolve_hf_cache(
            size.hf_repo(),
            &mtab_core::model::hf_hub_cache_dir()?,
        ),
        ModelSource::HfRepo(repo) => {
            mtab_core::model::resolve_hf_cache(repo, &mtab_core::model::hf_hub_cache_dir()?)
        }
        ModelSource::Url(url) => Err(MtabError::WeightsNotFound {
            path: PathBuf::from(format!(
                "{url} (remote fetching is not implemented; mtab never downloads CC BY-NC weights on its own — fetch it yourself and pass a local path)"
            )),
        }),
    }
}

/// A loaded model plus the settings to run it with.
///
/// Cheap to call [`Transcriber::transcribe_file`] repeatedly: the weights are
/// already in memory.
pub struct Transcriber {
    model: LmModel,
    size: ModelSize,
    guitar: GuitarConfig,
    decode: DecodeConfig,
    tempo: TempoMap,
}

impl std::fmt::Debug for TranscriberBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscriberBuilder")
            .field("size", &self.size)
            .field("weights", &self.weights)
            .field("model", &self.loaded.as_ref().map(|_| "<loaded>"))
            .field("guitar", &self.guitar)
            .field("decode", &self.decode)
            .field("grid", &self.grid)
            .field("tempo", &self.tempo)
            .finish()
    }
}

impl std::fmt::Debug for Transcriber {
    // The loaded weights are 400 MB to 5.5 GB of f64; printing them would be
    // an accident waiting to happen, so only the settings are shown.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transcriber")
            .field("size", &self.size)
            .field("guitar", &self.guitar)
            .field("decode", &self.decode)
            .field("tempo", &self.tempo)
            .finish_non_exhaustive()
    }
}

impl Transcriber {
    /// Start configuring one.
    #[must_use]
    pub fn builder() -> TranscriberBuilder {
        TranscriberBuilder {
            size: ModelSize::Medium,
            weights: None,
            loaded: None,
            guitar: GuitarConfig::default(),
            decode: DecodeConfig::default(),
            grid: Grid::SIXTEENTH,
            tempo: None,
        }
    }

    /// Which architecture was loaded.
    #[must_use]
    pub fn size(&self) -> ModelSize {
        self.size
    }

    /// Transcribe a file with no progress reporting.
    pub fn transcribe_file(&self, path: impl AsRef<Path>) -> Result<Transcription> {
        self.transcribe_file_with_progress(path, |_| {})
    }

    /// Transcribe a file, reporting [`Progress`] as each stage advances.
    ///
    /// The order is the CLI's: decode and resample, detect tuning and capo if
    /// asked, generate in order, group by instrument, finger each group,
    /// assemble the document.
    pub fn transcribe_file_with_progress(
        &self,
        path: impl AsRef<Path>,
        mut progress: impl FnMut(Progress),
    ) -> Result<Transcription> {
        progress(Progress {
            stage: Stage::Audio,
            completed: 0,
            total: None,
        });
        let audio = mtab_audio::load_model_input(path.as_ref())?;
        self.run(&audio, &mut progress)
    }

    /// Transcribe already-decoded 16 kHz mono audio, skipping the file hop.
    pub fn transcribe_audio(&self, audio: &mtab_audio::AudioBuffer) -> Result<Transcription> {
        self.run(audio, &mut |_: Progress| {})
    }

    fn run(
        &self,
        audio: &mtab_audio::AudioBuffer,
        progress: &mut impl FnMut(Progress),
    ) -> Result<Transcription> {
        let (tuning, capo) = self.resolve_guitar(audio, progress)?;

        let options = mtab_core::lm::GenOptions {
            max_gen_len: self.decode.max_gen_len,
            beam_size: self.decode.beam_size,
            use_sampling: self.decode.temperature.is_some(),
            temp: self.decode.temperature.unwrap_or(1.0),
            seed: self.decode.seed,
            ..mtab_core::lm::GenOptions::default()
        };
        progress(Progress {
            stage: Stage::Decode,
            completed: 0,
            total: None,
        });
        let result = mtab_audio::transcribe::transcribe(
            &self.model,
            audio,
            &self.decode.instruments,
            &options,
            self.decode.strict_eos,
            |p| {
                progress(Progress {
                    stage: Stage::Decode,
                    completed: p.completed,
                    total: Some(p.total),
                });
            },
        )?;

        let (document, omitted) = self.build_document(&result.notes, &tuning, capo, progress)?;
        Ok(Transcription {
            document,
            notes: result.notes,
            omitted,
            missing_eos: result.missing_eos,
        })
    }

    fn resolve_guitar(
        &self,
        audio: &mtab_audio::AudioBuffer,
        progress: &mut impl FnMut(Progress),
    ) -> Result<(Tuning, Capo)> {
        if let (TuningChoice::Fixed(tuning), CapoChoice::Fixed(capo)) =
            (&self.guitar.tuning, self.guitar.capo)
        {
            return Ok((tuning.clone(), capo));
        }
        progress(Progress {
            stage: Stage::Tuning,
            completed: 0,
            total: None,
        });
        let histogram = mtab_audio::pitch_histogram::pitch_histogram(audio)?;
        let fixed_tuning = match &self.guitar.tuning {
            TuningChoice::Fixed(tuning) => Some(tuning),
            TuningChoice::Auto => None,
        };
        let fixed_capo = match self.guitar.capo {
            CapoChoice::Fixed(capo) => Some(capo),
            CapoChoice::Auto => None,
        };
        let candidates: Vec<_> = mtab_guitar::standard_candidates()
            .into_iter()
            .filter(|c| fixed_tuning.map_or(true, |t| &c.tuning == t))
            .filter(|c| fixed_capo.map_or(true, |capo| c.capo == capo))
            .filter(|c| c.capo.position() <= self.guitar.frets)
            .collect();
        let detection = mtab_guitar::detect_tuning(&histogram.weights, &candidates)?;
        match (detection.sufficient_evidence, detection.best) {
            (true, Some(best)) => Ok((best.candidate.tuning, best.candidate.capo)),
            // Ambiguous evidence falls back exactly as the CLI documents.
            _ => Ok((
                fixed_tuning.cloned().unwrap_or_else(Tuning::standard),
                fixed_capo.unwrap_or(Capo::NONE),
            )),
        }
    }

    fn build_document(
        &self,
        notes: &[NoteEvent],
        tuning: &Tuning,
        capo: Capo,
        progress: &mut impl FnMut(Progress),
    ) -> Result<(TabDocument, Vec<NoteEvent>)> {
        let grouped = group_by_instrument(notes);
        let mut tracks = Vec::with_capacity(grouped.len());
        let mut omitted = Vec::new();

        for (done, (program, indexed)) in grouped.iter().enumerate() {
            progress(Progress {
                stage: Stage::Fingering,
                completed: done,
                total: Some(grouped.len()),
            });
            let track_notes: Vec<NoteEvent> = indexed.iter().map(|(_, n)| *n).collect();
            let options = FingeringOptions {
                max_fret: self.guitar.frets,
                ..FingeringOptions::default()
            };
            let assigned = if let Some(span) = self.guitar.wider_span {
                assign_notes_with_stretch(&track_notes, tuning, capo, options, span)?.notes
            } else if self.guitar.allow_partial {
                match assign_notes(&track_notes, tuning, capo, options) {
                    Ok(notes) => notes,
                    Err(MtabError::UnplayableNote { .. }) => {
                        let partial = assign_notes_partial(&track_notes, tuning, capo, options)?;
                        for &index in &partial.unassigned_indices {
                            omitted.push(track_notes[index]);
                        }
                        partial.notes.into_iter().flatten().collect()
                    }
                    Err(error) => return Err(error),
                }
            } else {
                assign_notes(&track_notes, tuning, capo, options)?
            };
            tracks.push(GuitarTrack {
                tuning: tuning.clone(),
                capo,
                instrument: Some(mtab_core::mt3::instruments::instrument_for_program(
                    *program,
                )),
                notes: assigned,
            });
        }
        progress(Progress {
            stage: Stage::Fingering,
            completed: grouped.len(),
            total: Some(grouped.len()),
        });

        if tracks.is_empty() {
            tracks.push(GuitarTrack {
                tuning: tuning.clone(),
                capo,
                instrument: None,
                notes: Vec::new(),
            });
        }

        Ok((
            TabDocument {
                meta: TrackMeta {
                    title: None,
                    tempo_map: self.tempo.clone(),
                    time_signatures: vec![(0, 4, 4)],
                    frets: self.guitar.frets,
                    source_hash: None,
                },
                tracks,
                chord_symbols: Vec::new(),
            },
            omitted,
        ))
    }
}

fn group_by_instrument(
    notes: &[NoteEvent],
) -> std::collections::BTreeMap<u16, Vec<(usize, NoteEvent)>> {
    let mut grouped: std::collections::BTreeMap<u16, Vec<(usize, NoteEvent)>> =
        std::collections::BTreeMap::new();
    for (index, note) in notes.iter().enumerate() {
        grouped
            .entry(note.instrument)
            .or_default()
            .push((index, *note));
    }
    grouped
}
