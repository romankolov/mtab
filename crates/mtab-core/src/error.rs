//! Shared typed errors (PROMPT.md §6.4).
//!
//! Every CLI failure must print an actionable next step — the `Display`
//! implementations below carry those hints; the CLI adds context.

use std::path::PathBuf;

/// Result alias used across the workspace.
pub type Result<T> = std::result::Result<T, MtabError>;

/// Workspace-wide error vocabulary.
#[derive(Debug, thiserror::Error)]
pub enum MtabError {
    #[error(
        "weights not found: {path}\n  hint: pass --weights <file.safetensors>, or run `mtab download --size <small|medium|large>` after accepting the CC BY-NC 4.0 license on the model's Hugging Face page"
    )]
    WeightsNotFound { path: PathBuf },

    #[error(
        "model weights are gated behind the CC BY-NC 4.0 license\n  hint: accept the license at https://huggingface.co/{repo}, then authenticate with `hf auth login` or set HF_TOKEN"
    )]
    LicenseGateRequired { repo: String },

    #[error("weights shape mismatch: expected {expected}, found {found}\n  hint: use weights matching --size (small=103M/14L/768, medium=307M/24L/1024, large=1.4B/48L/1536)")]
    ShapeMismatch { expected: String, found: String },

    #[error("unsupported audio: {0}\n  hint: mtab decodes wav/mp3/flac/ogg/m4a; convert with ffmpeg if needed")]
    UnsupportedAudio(String),

    #[error("decode failure at chunk {chunk_index}: {reason}")]
    DecodeFailure { chunk_index: usize, reason: String },

    #[error("unknown model size '{0}'\n  hint: valid sizes are small, medium, large")]
    UnknownModelSize(String),

    #[error("invalid model source '{0}'\n  hint: pass a <name>.safetensors path, an hf://<org>/<repo> reference, or an http(s):// URL")]
    InvalidModelSource(String),

    #[error("invalid tuning '{0}'\n  hint: run `mtab detect-tuning --list` for supported names")]
    InvalidTuning(String),

    #[error("invalid capo position {0} (must be 0-12)")]
    InvalidCapo(u8),

    #[error("not yet implemented: {0}")]
    NotImplemented(&'static str),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
