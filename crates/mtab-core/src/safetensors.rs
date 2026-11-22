//! Minimal safetensors header reader (PROMPT.md §6.3, §10: safetensors only,
//! never pickle).
//!
//! Format: 8-byte little-endian `u64` header length, then that many bytes of
//! UTF-8 JSON mapping tensor names to `{dtype, shape, data_offsets}`, then the
//! raw tensor data. This module parses only the header — enough for
//! `mtab inspect-weights` and for shape validation before M3 loads tensors.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;

use crate::error::{MtabError, Result};
use crate::model::ModelSize;

/// One tensor's metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TensorInfo {
    /// Element type as written in the file ("F32", "F16", "BF16", ...).
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Byte offsets into the data section: (begin, end).
    pub data_offsets: (usize, usize),
}

impl TensorInfo {
    /// Number of elements.
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Byte width of one element for known dtypes (`None` if unknown).
    pub fn element_size(&self) -> Option<usize> {
        Some(match self.dtype.as_str() {
            "F64" | "I64" | "U64" => 8,
            "F32" | "I32" | "U32" => 4,
            "F16" | "BF16" | "I16" | "U16" => 2,
            "I8" | "U8" | "BOOL" => 1,
            _ => return None,
        })
    }
}

/// Parsed safetensors header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightsInfo {
    /// Tensor name → metadata, sorted (deterministic iteration, §10.7).
    pub tensors: BTreeMap<String, TensorInfo>,
    /// Free-form `__metadata__` entries (often carries license/format hints).
    pub metadata: BTreeMap<String, String>,
}

impl WeightsInfo {
    /// Read and parse the header of a `.safetensors` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|_| MtabError::WeightsNotFound {
            path: path.to_owned(),
        })?;
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)
            .map_err(|e| MtabError::DecodeFailure {
                chunk_index: 0,
                reason: format!("reading safetensors header length: {e}"),
            })?;
        let header_len = u64::from_le_bytes(len_buf) as usize;
        // Sanity bound: the header is JSON metadata; anything larger is corrupt.
        if header_len > 512 * 1024 * 1024 {
            return Err(MtabError::DecodeFailure {
                chunk_index: 0,
                reason: format!("safetensors header implausibly large ({header_len} bytes)"),
            });
        }
        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header)
            .map_err(|e| MtabError::DecodeFailure {
                chunk_index: 0,
                reason: format!("reading safetensors header: {e}"),
            })?;
        Self::parse_header(&header)
    }

    /// Parse raw header JSON bytes.
    pub fn parse_header(header: &[u8]) -> Result<Self> {
        #[derive(Deserialize)]
        struct RawEntry {
            #[serde(default)]
            dtype: String,
            #[serde(default)]
            shape: Vec<usize>,
            #[serde(default)]
            data_offsets: (usize, usize),
        }

        let raw: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(header).map_err(|e| MtabError::DecodeFailure {
                chunk_index: 0,
                reason: format!("invalid safetensors header JSON: {e}"),
            })?;

        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (key, value) in raw {
            if key == "__metadata__" {
                if let Some(map) = value.as_object() {
                    for (k, v) in map {
                        metadata
                            .insert(k.clone(), v.as_str().map(str::to_owned).unwrap_or_default());
                    }
                }
            } else {
                let entry: RawEntry =
                    serde_json::from_value(value).map_err(|e| MtabError::DecodeFailure {
                        chunk_index: 0,
                        reason: format!("invalid tensor entry '{key}': {e}"),
                    })?;
                tensors.insert(
                    key,
                    TensorInfo {
                        dtype: entry.dtype,
                        shape: entry.shape,
                        data_offsets: entry.data_offsets,
                    },
                );
            }
        }
        Ok(Self { tensors, metadata })
    }

    /// Total parameter count across all tensors.
    pub fn total_params(&self) -> usize {
        self.tensors.values().map(TensorInfo::elements).sum()
    }

    /// Tensor count.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Which MuScriptor size these weights most likely are (within 10% of the
    /// nominal parameter count); `None` if nothing matches.
    pub fn likely_size(&self) -> Option<ModelSize> {
        let total = self.total_params() as u64;
        ModelSize::ALL
            .into_iter()
            .min_by_key(|s| {
                let nominal = s.shape().0;
                total.abs_diff(nominal)
            })
            .filter(|s| {
                let nominal = s.shape().0;
                total.abs_diff(nominal) * 10 <= nominal
            })
    }

    /// dtype histogram (dtype → tensor count), sorted by dtype name.
    pub fn dtypes(&self) -> BTreeMap<String, usize> {
        let mut out: BTreeMap<String, usize> = BTreeMap::new();
        for t in self.tensors.values() {
            *out.entry(t.dtype.clone()).or_default() += 1;
        }
        out
    }

    /// The `n` largest tensors by element count (name, elements).
    pub fn largest(&self, n: usize) -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = self
            .tensors
            .iter()
            .map(|(k, t)| (k.clone(), t.elements()))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v.truncate(n);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny in-memory safetensors file and parse it.
    fn fake_weights_file(header_json: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header_json.as_bytes());
        bytes.extend_from_slice(&[0u8; 64]); // dummy data section
        bytes
    }

    #[test]
    fn parses_header_and_counts_params() {
        let json = r#"{
            "layer.weight": {"dtype": "F32", "shape": [8, 4], "data_offsets": [0, 128]},
            "layer.bias":   {"dtype": "F32", "shape": [4],    "data_offsets": [128, 144]},
            "__metadata__": {"license": "CC BY-NC 4.0"}
        }"#;
        let info = WeightsInfo::parse_header(json.as_bytes()).unwrap();
        assert_eq!(info.tensor_count(), 2);
        assert_eq!(info.total_params(), 36);
        assert_eq!(info.metadata.get("license").unwrap(), "CC BY-NC 4.0");
        assert_eq!(info.dtypes().get("F32"), Some(&2));
        assert_eq!(info.largest(1), vec![("layer.weight".to_owned(), 32)]);
        assert_eq!(info.likely_size(), None); // toy params match nothing

        // And the same header through the full file reader.
        let dir = std::env::temp_dir();
        let path = dir.join("mtab-test-weights.safetensors");
        std::fs::write(&path, fake_weights_file(json)).unwrap();
        let from_disk = WeightsInfo::from_file(&path).unwrap();
        assert_eq!(from_disk, info);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn likely_size_classification() {
        // Fabricate a header whose element count is ~103M (small).
        let json = format!(
            r#"{{"w": {{"dtype": "F16", "shape": [{}], "data_offsets": [0, 0]}}}}"#,
            103_000_000
        );
        let info = WeightsInfo::parse_header(json.as_bytes()).unwrap();
        assert_eq!(info.likely_size(), Some(ModelSize::Small));
    }

    #[test]
    fn rejects_garbage_header() {
        assert!(WeightsInfo::parse_header(b"not json at all").is_err());
    }
}
