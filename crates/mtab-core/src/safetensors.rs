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
        self.shape
            .iter()
            .copied()
            .fold(1usize, usize::saturating_mul)
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
    /// Header JSON length in bytes (set by [`WeightsInfo::from_bytes`]).
    pub header_size: usize,
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
        let header_len = usize::try_from(u64::from_le_bytes(len_buf)).map_err(|_| {
            MtabError::SafetensorsCorrupt {
                reason: "header length exceeds platform size".into(),
            }
        })?;
        // Sanity bound: the header is JSON metadata; anything larger is corrupt.
        if header_len > 512 * 1024 * 1024 {
            return Err(MtabError::DecodeFailure {
                chunk_index: 0,
                reason: format!("safetensors header implausibly large ({header_len} bytes)"),
            });
        }
        let file_size = file.metadata()?.len();
        if header_len as u64 > file_size.saturating_sub(8) {
            return Err(MtabError::SafetensorsCorrupt {
                reason: "truncated safetensors header".into(),
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

    /// Parse header + offsets from a full in-memory file image (tests,
    /// [`load_weights_bytes`]).
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < 8 {
            return Err(MtabError::SafetensorsCorrupt {
                reason: "truncated safetensors file (< 8 bytes)".into(),
            });
        }
        let header_len = usize::try_from(u64::from_le_bytes(buf[0..8].try_into().unwrap()))
            .map_err(|_| MtabError::SafetensorsCorrupt {
                reason: "header length exceeds platform size".into(),
            })?;
        if header_len > 512 * 1024 * 1024 || header_len > buf.len() - 8 {
            return Err(MtabError::SafetensorsCorrupt {
                reason: format!(
                    "safetensors header implausibly large or truncated ({header_len} bytes)"
                ),
            });
        }
        let data_start = 8 + header_len;
        let mut info = Self::parse_header(&buf[8..data_start])?;
        info.header_size = header_len;
        let payload_len = buf.len() - data_start;
        let mut spans = Vec::new();
        for (name, ti) in &info.tensors {
            let (start, end) = ti.data_offsets;
            if start > end || end > payload_len {
                return Err(MtabError::SafetensorsCorrupt {
                    reason: format!("tensor '{name}' data out of bounds"),
                });
            }
            let count = ti
                .shape
                .iter()
                .try_fold(1usize, |acc, &n| acc.checked_mul(n))
                .ok_or_else(|| MtabError::SafetensorsCorrupt {
                    reason: format!("tensor '{name}' shape overflows"),
                })?;
            if let Some(width) = ti.element_size() {
                let expected =
                    count
                        .checked_mul(width)
                        .ok_or_else(|| MtabError::SafetensorsCorrupt {
                            reason: format!("tensor '{name}' byte length overflows"),
                        })?;
                if end - start != expected {
                    return Err(MtabError::SafetensorsCorrupt {
                        reason: format!(
                            "tensor '{name}' shape/dtype require {expected} bytes, found {}",
                            end - start
                        ),
                    });
                }
            }
            if start != end {
                spans.push((start, end, name));
            }
        }
        spans.sort_unstable_by_key(|span| span.0);
        for pair in spans.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(MtabError::SafetensorsCorrupt {
                    reason: format!("overlapping tensors '{}' and '{}'", pair[0].2, pair[1].2),
                });
            }
        }
        Ok(info)
    }

    /// Parse raw header JSON bytes.
    pub fn parse_header(header: &[u8]) -> Result<Self> {
        #[derive(Deserialize)]
        struct RawEntry {
            dtype: String,
            shape: Vec<usize>,
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
        Ok(Self {
            tensors,
            metadata,
            header_size: 0,
        })
    }

    /// Total parameter count across all tensors.
    pub fn total_params(&self) -> usize {
        self.tensors
            .values()
            .map(TensorInfo::elements)
            .fold(0usize, usize::saturating_add)
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
                (total.abs_diff(nominal) as u128) * 10 <= nominal as u128
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

/// Decoded tensor payload (floats converted to `f32`; integer dtypes kept
/// raw). Created by [`load_weights`].
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Tensor name as stored in the header.
    pub name: String,
    /// Declared shape.
    pub shape: Vec<usize>,
    /// Decoded elements in row-major order (f32 for F32/F16/BF16).
    pub data: Vec<f32>,
}

impl Tensor {
    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// IEEE 754 binary16 → f32 (handles subnormals, inf, nan).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    let out = match exp {
        0 => {
            if frac == 0 {
                sign << 31 // ±0
            } else {
                // subnormal: normalize
                let mut e = 127 - 15 + 1;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x3FF;
                (sign << 31) | ((e as u32) << 23) | (f << 13)
            }
        }
        0x1F => (sign << 31) | (0xFF << 23) | (frac << 13), // inf/nan
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(out)
}

/// bfloat16 → f32 (top-half truncation).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Load and decode every tensor of a safetensors file into `f32` (PROMPT.md
/// §6.5: real loading, no external crates).
pub fn load_weights(path: &Path) -> Result<BTreeMap<String, Tensor>> {
    let buf = std::fs::read(path)?;
    load_weights_bytes(&buf)
}

/// In-memory variant of [`load_weights`] (tests).
pub fn load_weights_bytes(buf: &[u8]) -> Result<BTreeMap<String, Tensor>> {
    let info = WeightsInfo::from_bytes(buf)?;
    let data_start = 8 + info.header_size;
    let mut out = BTreeMap::new();
    for (name_key, ti) in &info.tensors {
        let s = data_start + ti.data_offsets.0;
        let e = data_start + ti.data_offsets.1;
        let raw = buf.get(s..e).ok_or_else(|| MtabError::SafetensorsCorrupt {
            reason: format!("tensor '{name_key}' data out of bounds"),
        })?;
        let data = match ti.dtype.as_str() {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => {
                return Err(MtabError::SafetensorsCorrupt {
                    reason: format!("tensor '{name_key}': dtype {other} not loadable"),
                })
            }
        };
        out.insert(
            name_key.clone(),
            Tensor {
                name: name_key.clone(),
                shape: ti.shape.clone(),
                data,
            },
        );
    }
    Ok(out)
}

/// Apply upstream `_remap_single_codebook_keys` (`transcription_model.py`):
/// legacy multi-codebook checkpoints map `emb.0.*`→`emb.*` and
/// `linears.0.*`→`linear.*`; a second codebook (`emb.1.*`/`linears.1.*`) is
/// rejected.
pub fn remap_single_codebook_keys(
    state: BTreeMap<String, Tensor>,
) -> Result<BTreeMap<String, Tensor>> {
    if state
        .keys()
        .any(|k| k.starts_with("emb.1.") || k.starts_with("linears.1."))
    {
        return Err(MtabError::SafetensorsCorrupt {
            reason: "checkpoint has more than one codebook (n_q > 1); only single-stream models are supported".into(),
        });
    }
    let mut out = BTreeMap::new();
    for (key, mut tensor) in state {
        let key = if let Some(rest) = key.strip_prefix("emb.0.") {
            format!("emb.{rest}")
        } else if let Some(rest) = key.strip_prefix("linears.0.") {
            format!("linear.{rest}")
        } else {
            key
        };
        tensor.name = key.clone();
        out.insert(key, tensor);
    }
    Ok(out)
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

    /// Build a minimal safetensors image: given (name, dtype, shape, data).
    fn build_st(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
        let mut header = String::from("{");
        let mut offset = 0usize;
        for (i, (name, dtype, shape, data)) in tensors.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{offset},{}]}}",
                offset + data.len()
            ));
            offset += data.len();
        }
        header.push('}');
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        for (_, _, _, data) in tensors {
            buf.extend_from_slice(data);
        }
        buf
    }

    #[test]
    fn loads_f32_f16_bf16() {
        let buf = build_st(&[
            ("a.weight", "F32", vec![1], 1.0f32.to_le_bytes().to_vec()),
            (
                "b.weight",
                "F16",
                vec![3],
                [0x3C00u16, 0xC000, 0x3800]
                    .iter()
                    .flat_map(|u| u.to_le_bytes())
                    .collect(),
            ),
            (
                "c.weight",
                "BF16",
                vec![2],
                [0x3F80u16, 0x4000]
                    .iter()
                    .flat_map(|u| u.to_le_bytes())
                    .collect(),
            ),
        ]);
        let w = load_weights_bytes(&buf).unwrap();
        assert_eq!(w["a.weight"].data, vec![1.0]);
        assert_eq!(w["b.weight"].data, vec![1.0, -2.0, 0.5]); // 0x3C00, 0xC000, 0x3800
        assert_eq!(w["c.weight"].data, vec![1.0, 2.0]);
        assert_eq!(w["b.weight"].shape, vec![3]);
        assert_eq!(w["b.weight"].numel(), 3);
    }

    #[test]
    fn remaps_legacy_codebook_keys() {
        let buf = build_st(&[
            (
                "emb.0.weight",
                "F32",
                vec![1],
                3.0f32.to_le_bytes().to_vec(),
            ),
            (
                "linears.0.weight",
                "F32",
                vec![1],
                4.0f32.to_le_bytes().to_vec(),
            ),
            ("emb.weight", "F32", vec![1], 5.0f32.to_le_bytes().to_vec()),
        ]);
        let w = remap_single_codebook_keys(load_weights_bytes(&buf).unwrap()).unwrap();
        assert_eq!(w["emb.weight"].data, vec![5.0]); // direct key kept
        assert_eq!(w["linear.weight"].data, vec![4.0]);
        assert!(!w.contains_key("emb.0.weight"));

        let bad = build_st(&[(
            "emb.1.weight",
            "F32",
            vec![1],
            1.0f32.to_le_bytes().to_vec(),
        )]);
        let err = remap_single_codebook_keys(load_weights_bytes(&bad).unwrap()).unwrap_err();
        assert!(err.to_string().contains("codebook"));
    }

    #[test]
    fn rejects_out_of_bounds_offsets() {
        let mut buf = build_st(&[("a", "F32", vec![2], 1.0f32.to_le_bytes().to_vec())]);
        buf.truncate(buf.len() - 2); // cut tensor data short
        assert!(load_weights_bytes(&buf).is_err());
    }
    #[test]
    fn rejects_inconsistent_tensor_layouts_without_panicking() {
        let short = build_st(&[("x", "F32", vec![2], 1.0f32.to_le_bytes().to_vec())]);
        assert!(load_weights_bytes(&short).is_err());
        let huge_shape = build_st(&[("x", "F32", vec![usize::MAX, 2], vec![])]);
        assert!(load_weights_bytes(&huge_shape).is_err());
        let overflow = fake_weights_file(&format!(
            r#"{{"x":{{"dtype":"F32","shape":[1],"data_offsets":[{},{}]}}}}"#,
            usize::MAX - 1,
            usize::MAX
        ));
        assert!(load_weights_bytes(&overflow).is_err());
        let overlap = fake_weights_file(
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
        );
        assert!(load_weights_bytes(&overlap).is_err());
        let missing_field = fake_weights_file(r#"{"x":{"dtype":"F32","shape":[0]}}"#);
        assert!(load_weights_bytes(&missing_field).is_err());
    }
}
