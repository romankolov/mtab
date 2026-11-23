//! Model variants, sources, and configuration.
//!
//! The end user provides exactly two things (PROMPT.md §1): a model **size**
//! (`small` / `medium` / `large`) and its **weights** (safetensors). Everything
//! else is built in.

use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
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
    /// Previously cached official Hugging Face weights for this size.
    Cached { size: ModelSize },
    /// A Hugging Face repo reference, e.g. `hf://MuScriptor/muscriptor-medium`.
    HfRepo(String),
    /// An HTTP(S) URL pointing at a safetensors file.
    Url(String),
}

impl FromStr for ModelSource {
    type Err = MtabError;

    fn from_str(s: &str) -> Result<Self> {
        if let Ok(size) = s.parse::<ModelSize>() {
            return Ok(Self::Cached { size });
        }
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

/// Resolve `model.safetensors` from a Hugging Face Hub cache without network.
///
/// `hub` is the Hub cache root, not a repository directory. The repo must be
/// `owner/name`; this function follows only `refs/main`, requires a 40-digit
/// hexadecimal commit, and never chooses an arbitrary stale snapshot. Cached
/// files may normally be symlinks into the Hub's `blobs` directory. This checks
/// file presence, not integrity, model shape, or license acceptance; the loader
/// validates safetensors. There is no download and no token is read.
pub fn resolve_hf_cache(repo: &str, hub: &Path) -> Result<PathBuf> {
    let components: Vec<_> = repo.split('/').collect();
    if components.len() != 2
        || components.iter().any(|part| {
            part.is_empty()
                || *part == "."
                || *part == ".."
                || part.contains("--")
                || !part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        })
    {
        return Err(MtabError::InvalidModelSource(format!(
            "invalid Hugging Face repository: {repo}"
        )));
    }
    let directory = hub.join(format!("models--{}--{}", components[0], components[1]));
    let reference = directory.join("refs/main");
    let file = std::fs::File::open(&reference).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            MtabError::WeightsNotFound {
                path: reference.clone(),
            }
        } else {
            error.into()
        }
    })?;
    let mut bytes = Vec::new();
    file.take(129).read_to_end(&mut bytes)?;
    if bytes.len() > 128 {
        return Err(MtabError::InvalidModelSource(
            "cached refs/main exceeds 128 bytes".into(),
        ));
    }
    let hash = std::str::from_utf8(&bytes)
        .map_err(|_| MtabError::InvalidModelSource("cached refs/main is not UTF-8".into()))?
        .trim();
    if hash.len() != 40 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(MtabError::InvalidModelSource(
            "cached refs/main must be a 40-digit hexadecimal commit".into(),
        ));
    }
    let path = directory
        .join("snapshots")
        .join(hash)
        .join("model.safetensors");
    if !path.is_file() {
        return Err(MtabError::WeightsNotFound { path });
    }
    Ok(path)
}

/// Determine the Hugging Face Hub cache root from the standard environment.
/// `HF_HUB_CACHE` takes precedence over `HF_HOME/hub`, then the user's home
/// cache (using `XDG_CACHE_HOME` if set). Empty variables are ignored. This
/// function never accesses a network.
pub fn hf_hub_cache_dir() -> Result<PathBuf> {
    fn nonempty(name: &str) -> Option<PathBuf> {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    }
    if let Some(path) = nonempty("HF_HUB_CACHE") {
        return Ok(path);
    }
    if let Some(path) = nonempty("HF_HOME") {
        return Ok(path.join("hub"));
    }
    if let Some(path) = nonempty("XDG_CACHE_HOME") {
        return Ok(path.join("huggingface").join("hub"));
    }
    let home = nonempty("HOME")
        .or_else(|| nonempty("USERPROFILE"))
        .ok_or_else(|| {
            MtabError::InvalidModelSource(
                "set HF_HUB_CACHE or HF_HOME to locate the Hugging Face cache".into(),
            )
        })?;
    Ok(home.join(".cache").join("huggingface").join("hub"))
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
    fn resolves_only_valid_hf_main_snapshots_without_network() {
        let dir = std::env::temp_dir().join(format!(
            "mtab-hf-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let repo = dir.join("models--MuScriptor--muscriptor-small");
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        let reference = repo.join("refs/main");
        for name in [
            "", "../x", "a/../x", "a/..", "a/b/c", "a//b", "a\\b/c", "a--b/c", "a/b?x",
        ] {
            assert!(resolve_hf_cache(name, &dir).is_err(), "{name}");
        }
        assert!(resolve_hf_cache("MuScriptor/muscriptor-small", &dir).is_err());
        for invalid in [
            "../../escape".to_owned(),
            "z".repeat(40),
            "f".repeat(41),
            format!("{}{}", "f".repeat(40), " ".repeat(100)),
        ] {
            std::fs::write(&reference, invalid).unwrap();
            assert!(resolve_hf_cache("MuScriptor/muscriptor-small", &dir).is_err());
        }
        let hash = "a".repeat(40);
        std::fs::write(&reference, format!("{hash}\n")).unwrap();
        assert!(resolve_hf_cache("MuScriptor/muscriptor-small", &dir).is_err());
        let snapshot = repo.join("snapshots").join(&hash);
        std::fs::create_dir_all(&snapshot).unwrap();
        let weights = snapshot.join("model.safetensors");
        std::fs::write(&weights, b"test fixture, not model weights").unwrap();
        assert_eq!(
            resolve_hf_cache("MuScriptor/muscriptor-small", &dir).unwrap(),
            weights
        );
        std::fs::remove_file(&reference).unwrap();
        assert!(resolve_hf_cache("MuScriptor/muscriptor-small", &dir).is_err());
        std::fs::remove_dir_all(dir).unwrap();
        assert_eq!(
            "small".parse::<ModelSource>().unwrap(),
            ModelSource::Cached {
                size: ModelSize::Small
            }
        );
    }

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
