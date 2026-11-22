//! Model variants, sources, and configuration.
//!
//! The end user provides exactly two things (PROMPT.md §1): a model **size**
//! (`small` / `medium` / `large`) and its **weights** (safetensors). Everything
//! else is built in.

use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{MtabError, Result};

/// MuScriptor model variants.
///
/// Shapes match the official release table (PROMPT.md §2):
/// small = 103M / 14 layers / dim 768, medium = 307M / 24 / 1024,
/// large = 1.4B / 48 / 1536.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSize {
    Small,
    Medium,
    Large,
}

impl ModelSize {
    /// `(approx_params, layers, dim)` from the official MuScriptor table.
    pub const fn shape(self) -> (u64, usize, usize) {
        match self {
            Self::Small => (103_000_000, 14, 768),
            Self::Medium => (307_000_000, 24, 1024),
            Self::Large => (1_400_000_000, 48, 1536),
        }
    }

    /// Decoder hidden dimension.
    pub const fn dim(self) -> usize {
        self.shape().2
    }

    /// Number of transformer layers.
    pub const fn layers(self) -> usize {
        self.shape().1
    }

    /// Hugging Face repo id hosting the gated CC BY-NC weights.
    pub const fn hf_repo(self) -> &'static str {
        match self {
            Self::Small => "MuScriptor/muscriptor-small",
            Self::Medium => "MuScriptor/muscriptor-medium",
            Self::Large => "MuScriptor/muscriptor-large",
        }
    }

    /// All variants, small → large.
    pub const ALL: [ModelSize; 3] = [Self::Small, Self::Medium, Self::Large];
}

impl fmt::Display for ModelSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        };
        f.write_str(s)
    }
}

impl FromStr for ModelSize {
    type Err = MtabError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            other => Err(MtabError::UnknownModelSize(other.to_owned())),
        }
    }
}

/// Where the weights come from. Only safetensors is accepted (PROMPT.md §6.3,
/// §10: never pickle).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSource {
    /// A local `model.safetensors` file.
    Local(PathBuf),
    /// The crate's own cache of previously fetched weights for this size.
    Cached { size: ModelSize },
    /// A Hugging Face repo reference, e.g. `hf://MuScriptor/muscriptor-medium`.
    HfRepo(String),
    /// An HTTP(S) URL pointing at a safetensors file.
    Url(String),
}

impl FromStr for ModelSource {
    type Err = MtabError;

    fn from_str(s: &str) -> Result<Self> {
        if let Some(rest) = s.strip_prefix("hf://") {
            return Ok(Self::HfRepo(rest.to_owned()));
        }
        if s.starts_with("http://") || s.starts_with("https://") {
            return Ok(Self::Url(s.to_owned()));
        }
        let path = PathBuf::from(s);
        if path.extension().is_some_and(|e| e == "safetensors") {
            return Ok(Self::Local(path));
        }
        Err(MtabError::InvalidModelSource(s.to_owned()))
    }
}

/// Full model configuration handed to the inference engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub size: ModelSize,
    pub source: ModelSource,
}

impl ModelConfig {
    pub fn new(size: ModelSize, source: ModelSource) -> Self {
        Self { size, source }
    }

    /// `(approx_params, layers, dim)` shortcut.
    pub fn shape(&self) -> (u64, usize, usize) {
        self.size.shape()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_shapes_match_reference_table() {
        assert_eq!(ModelSize::Small.shape(), (103_000_000, 14, 768));
        assert_eq!(ModelSize::Medium.shape(), (307_000_000, 24, 1024));
        assert_eq!(ModelSize::Large.shape(), (1_400_000_000, 48, 1536));
    }

    #[test]
    fn model_size_parses_case_insensitively() {
        assert_eq!("small".parse::<ModelSize>().unwrap(), ModelSize::Small);
        assert_eq!("Medium".parse::<ModelSize>().unwrap(), ModelSize::Medium);
        assert_eq!("LARGE".parse::<ModelSize>().unwrap(), ModelSize::Large);
        assert!("huge".parse::<ModelSize>().is_err());
    }

    #[test]
    fn hf_repos_match_upstream_table() {
        assert_eq!(ModelSize::Small.hf_repo(), "MuScriptor/muscriptor-small");
        assert_eq!(ModelSize::Medium.hf_repo(), "MuScriptor/muscriptor-medium");
        assert_eq!(ModelSize::Large.hf_repo(), "MuScriptor/muscriptor-large");
    }

    #[test]
    fn model_source_parses_all_forms() {
        assert_eq!(
            "weights/model.safetensors".parse::<ModelSource>().unwrap(),
            ModelSource::Local(PathBuf::from("weights/model.safetensors"))
        );
        assert_eq!(
            "hf://MuScriptor/muscriptor-medium"
                .parse::<ModelSource>()
                .unwrap(),
            ModelSource::HfRepo("MuScriptor/muscriptor-medium".to_owned())
        );
        assert_eq!(
            "https://example.com/m.safetensors"
                .parse::<ModelSource>()
                .unwrap(),
            ModelSource::Url("https://example.com/m.safetensors".to_owned())
        );
        assert!("weights/model.bin".parse::<ModelSource>().is_err());
    }
}
