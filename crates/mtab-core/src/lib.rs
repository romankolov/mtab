//! `mtab-core` — core of **muscriptor-tab**: MuScriptor model configuration,
//! weights loading, inference, and MT3 decoding.
//!
//! See `PROMPT.md` (§3–§6) for the full architecture. This crate holds the
//! cross-cutting types every other crate builds on:
//! - [`model`]: model sizes/sources/configs (the user contract: size + weights)
//! - [`event`]: raw note events produced by the transcription engine
//! - [`error`]: the shared typed error vocabulary
//!
//! Ported behavior follows the MIT-licensed upstream
//! <https://github.com/muscriptor/muscriptor>; attribution is retained per
//! file where code is ported (NOTICE).

pub mod error;
pub mod event;
pub mod lm;
pub mod model;
pub mod mt3;
pub mod nn;
pub mod safetensors;

pub use error::{MtabError, Result};
pub use event::{NoteEvent, ProgressEvent};
pub use model::{ModelConfig, ModelSize, ModelSource};
pub use safetensors::{TensorInfo, WeightsInfo};
