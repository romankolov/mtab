//! Guitar stem separation before transcription (PROMPT.md §7.1).
//!
//! Isolating the guitar is the single largest accuracy gain available on a full
//! mix: the transcriber is looking for guitar notes in audio that also contains
//! bass, drums and vocals, and every other instrument is a false-positive
//! candidate. This module runs a Demucs v4 (HTDemucs) ONNX graph over the mix
//! and returns one stem.
//!
//! **Optional by construction.** Everything here sits behind the `separation`
//! feature, and ONNX Runtime is loaded *dynamically* (`load-dynamic`), so the
//! default build links nothing and a binary built without the feature still
//! runs. No model is bundled: the caller passes a path and the file is never
//! downloaded (§10.1). The graph variant and its licence are the caller's
//! choice; the one this was validated against is documented in
//! `docs/separation.md`.
//!
//! **Why the fixed segment length.** HTDemucs is not fully convolutional in
//! time: the graph reshapes to a fixed number of frames and rejects any other
//! input length outright. The `[1, 2, 343980]` sample window (7.8 s at 44.1
//! kHz) is a property of the exported graph, not a choice made here, and the wrapper currently targets this validated six-stem export.

#![cfg(feature = "separation")]

use std::path::Path;

use mtab_core::{MtabError, Result};
use ort::session::builder::GraphOptimizationLevel as OptimizationLevel;

/// Sample rate the Demucs graphs are exported at.
pub const SEPARATION_SAMPLE_RATE: u32 = 44_100;

/// Default segment length in samples. `htdemucs_6s.onnx` declares
/// `mix [1, 2, "samples"]` but reshapes internally to a fixed frame count, so
/// only this length (and nothing else) executes.
pub const DEFAULT_SEGMENT_SAMPLES: usize = 343_980;

/// Where the guitar stem sits in the six-source graph's output order.
///
/// The order is the model's, not ours: `drums, bass, other, vocals, guitar,
/// piano`. A four-source graph has no guitar stem at all, which is why
/// [`Separator::new`] reports the stem count rather than assuming one.
pub const GUITAR_STEM_INDEX: usize = 4;

/// A loaded separation model.
pub struct Separator {
    session: ort::session::Session,
    stems: usize,
    segment: usize,
}

impl std::fmt::Debug for Separator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Separator")
            .field("stems", &self.stems)
            .field("segment_samples", &self.segment)
            .finish_non_exhaustive()
    }
}

impl Separator {
    /// Load an ONNX HTDemucs graph from `path`.
    ///
    /// The file is supplied by the caller and is never fetched. It is not
    /// bundled with this crate: see `docs/separation.md` for the variant this
    /// was tested against, its hash and its licence.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(MtabError::InvalidModelSource(format!(
                "separation model {} does not exist; pass a local ONNX file (it is never downloaded)",
                path.display()
            )));
        }
        let environment = ort::environment::init().build().map_err(|e| {
            MtabError::InvalidModelSource(format!(
                "ONNX Runtime could not start for separation: {e}. It is loaded at run time: \
                 point LD_LIBRARY_PATH (Linux), DYLD_LIBRARY_PATH (macOS) or PATH (Windows) at \
                 the onnxruntime shared library"
            ))
        })?;
        let inits = ort::session::Session::builder(&environment)
            .map_err(|e| MtabError::InvalidModelSource(format!("separation session: {e}")))?;
        let mut builder = inits
            .with_optimization_level(OptimizationLevel::Level1)
            .map_err(|e| MtabError::InvalidModelSource(format!("separation session: {e}")))?;
        let session = builder.commit_from_file(path).map_err(|e| {
            MtabError::InvalidModelSource(format!(
                "cannot load separation model {}: {e}. ONNX Runtime is loaded at run time: \
                     point LD_LIBRARY_PATH (Linux), DYLD_LIBRARY_PATH (macOS) or PATH (Windows) at \
                     the onnxruntime shared library",
                path.display()
            ))
        })?;

        // Output is [1, stems, 2, samples]; trust the graph over a constant.
        let stems = session
            .outputs()
            .first()
            .and_then(|o| match o.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => shape.get(1).copied(),
                _ => None,
            })
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(6);
        if stems <= GUITAR_STEM_INDEX {
            return Err(MtabError::InvalidModelSource(format!(
                "separation model has {stems} stems, so it has no guitar stem at index {GUITAR_STEM_INDEX}; \
                 use a six-source graph (drums, bass, other, vocals, guitar, piano)"
            )));
        }
        Ok(Self {
            session,
            stems,
            segment: DEFAULT_SEGMENT_SAMPLES,
        })
    }

    /// Number of stems this graph produces.
    #[must_use]
    pub fn stems(&self) -> usize {
        self.stems
    }

    /// Segment length the graph executes on, in samples at 44.1 kHz.
    #[must_use]
    pub fn segment_samples(&self) -> usize {
        self.segment
    }

    /// Separate `audio` and return only the guitar stem, resampled to the
    /// transcriber's 16 kHz mono input.
    ///
    /// `audio` may be mono or stereo at any rate; it is resampled to 44.1 kHz
    /// and split into fixed windows. The windows are concatenated without
    /// overlap, so a note crossing a boundary loses its tail in one window and
    /// its attack in the next; that is a real limitation, not a rounding note.
    pub fn guitar_stem(&mut self, audio: &super::AudioBuffer) -> Result<super::AudioBuffer> {
        let stereo = to_stereo_44k(audio)?;
        let frames = stereo.len() / 2;
        if frames == 0 {
            return Err(MtabError::InvalidModelSource(
                "separation needs at least one audio sample".into(),
            ));
        }
        let mut guitar = vec![0.0f32; frames];
        let mut covered = 0usize;
        while covered < frames {
            let take = self.segment.min(frames - covered);
            // The graph rejects any other length, so the tail is zero-padded
            // and the padding is trimmed again after inference.
            let mut window = vec![0.0f32; self.segment * 2];
            // `stereo` and the model tensor are planar: [left samples, right
            // samples]. Copy each channel into its own fixed-size plane.
            window[..take].copy_from_slice(&stereo[covered..covered + take]);
            window[self.segment..self.segment + take]
                .copy_from_slice(&stereo[frames + covered..frames + covered + take]);
            let input = ort::value::Tensor::from_array(([1usize, 2, self.segment], window))
                .map_err(|e| MtabError::InvalidModelSource(format!("separation input: {e}")))?;
            let outputs = self
                .session
                .run(ort::inputs!["mix" => input])
                .map_err(|e| MtabError::InvalidModelSource(format!("separation failed: {e}")))?;
            let tensor = outputs.get("sources").ok_or_else(|| {
                MtabError::InvalidModelSource("separation produced no output".into())
            })?;
            let (shape, data) = tensor
                .try_extract_tensor::<f32>()
                .map_err(|e| MtabError::InvalidModelSource(format!("separation output: {e}")))?;
            let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            let samples = *dims.last().unwrap_or(&self.segment);
            let plane = samples * 2;
            let base = GUITAR_STEM_INDEX * plane;
            for i in 0..take {
                // Mean the two channels: the transcriber takes mono anyway.
                let left = data[base + i];
                let right = data[base + plane / 2 + i];
                guitar[covered + i] = 0.5 * (left + right);
            }
            covered += take;
        }
        super::resample::resample(
            &super::AudioBuffer {
                samples: guitar,
                sample_rate: SEPARATION_SAMPLE_RATE,
                channels: 1,
            },
            super::MODEL_SAMPLE_RATE,
        )
    }
}

fn to_stereo_44k(audio: &super::AudioBuffer) -> Result<Vec<f32>> {
    if audio.channels == 0 || audio.channels > 2 {
        return Err(MtabError::InvalidModelSource(format!(
            "separation takes mono or stereo audio, got {} channels",
            audio.channels
        )));
    }
    let resampled = if audio.sample_rate == SEPARATION_SAMPLE_RATE {
        audio.clone()
    } else {
        super::resample::resample(audio, SEPARATION_SAMPLE_RATE)?
    };
    let frames = resampled.frames();
    // The graph takes planar stereo, `[1, 2, samples]`: all of the left
    // channel, then all of the right.
    let mut planar = Vec::with_capacity(frames * 2);
    for channel in 0..2usize {
        for frame in 0..frames {
            let index = if resampled.channels == 1 {
                frame
            } else {
                frame * 2 + channel
            };
            planar.push(resampled.samples[index]);
        }
    }
    Ok(planar)
}

/// One plucked string: white noise in a delay line, low-passed on each pass.
#[cfg(test)]
fn karplus_strong(f0: f32, n: usize, decay: f32, state: &mut u32) -> Vec<f32> {
    let period = (SEPARATION_SAMPLE_RATE as f32 / f0).round().max(2.0) as usize;
    let mut ring: Vec<f32> = (0..period)
        .map(|_| {
            // xorshift32: small, deterministic, and good enough to excite a
            // string. Tests must never depend on the host's RNG.
            *state ^= *state << 13;
            *state ^= *state >> 17;
            *state ^= *state << 5;
            (*state as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    let mut out = Vec::with_capacity(n);
    let mut index = 0usize;
    for _ in 0..n {
        let value = ring[index];
        out.push(value);
        let next = (index + 1) % period;
        ring[index] = decay * 0.5 * (value + ring[next]);
        index = next;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_conversion_is_planar_and_duplicates_mono() {
        let stereo = super::super::AudioBuffer {
            samples: vec![1.0, 2.0, 3.0, 4.0],
            sample_rate: SEPARATION_SAMPLE_RATE,
            channels: 2,
        };
        assert_eq!(to_stereo_44k(&stereo).unwrap(), vec![1.0, 3.0, 2.0, 4.0]);

        let mono = super::super::AudioBuffer {
            samples: vec![5.0, 6.0],
            sample_rate: SEPARATION_SAMPLE_RATE,
            channels: 1,
        };
        assert_eq!(to_stereo_44k(&mono).unwrap(), vec![5.0, 6.0, 5.0, 6.0]);
    }

    /// Runs the real graph when a checkpoint is supplied out of band; skips
    /// otherwise so CI and contributors without the 246 MB file still pass.
    /// Set `MTAB_SEPARATION_AUDIO` to check a real local recording as well.
    #[test]
    fn guitar_stem_isolates_the_guitar_from_a_guitar_only_recording() {
        let Some(path) = std::env::var_os("MTAB_SEPARATION_MODEL") else {
            eprintln!("MTAB_SEPARATION_MODEL unset; skipping the separation model test");
            return;
        };
        let mut separator = Separator::new(&path).expect("model loads");
        assert_eq!(
            separator.stems(),
            6,
            "the six-source graph has a guitar stem"
        );
        // Two seconds of a strummed open E, synthesised with Karplus-Strong.
        // A bare sine does not survive this test: the graph files a 110 Hz
        // pure tone under bass (measured: 2e-6 of its power reaches the guitar
        // stem), and even a static harmonic stack only reaches 24 %. Real
        // plucked strings are noise-excited, and the graph agrees: this signal
        // puts 92.5 % of its power in the guitar stem, real GuitarSet audio
        // 102 %.
        let rate = SEPARATION_SAMPLE_RATE;
        let samples = {
            let n = rate as usize * 2;
            let mut buf = vec![0.0f32; n];
            let mut state = 0x2545_f491u32;
            for (index, f0) in [82.41f32, 110.0, 146.83, 196.0, 246.94, 329.63]
                .into_iter()
                .enumerate()
            {
                let delay = (index as f32 * 0.02 * rate as f32) as usize;
                let pluck = karplus_strong(f0, n - delay, 0.996, &mut state);
                for (i, v) in pluck.into_iter().enumerate() {
                    let t = i as f32 / rate as f32;
                    buf[delay + i] += v * (-2.0 * t).exp();
                }
            }
            let peak = buf.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for v in &mut buf {
                *v = *v / peak * 0.5;
            }
            buf
        };
        let input = if let Some(path) = std::env::var_os("MTAB_SEPARATION_AUDIO") {
            let mut audio =
                super::super::load_model_input(Path::new(&path)).expect("real audio decodes");
            // Keep the optional real-audio smoke test to one inference window,
            // both to test the fixed-length contract and to cap peak memory.
            let limit = DEFAULT_SEGMENT_SAMPLES * super::super::MODEL_SAMPLE_RATE as usize
                / SEPARATION_SAMPLE_RATE as usize;
            audio.samples.truncate(limit);
            audio
        } else {
            super::super::AudioBuffer {
                samples,
                sample_rate: rate,
                channels: 1,
            }
        };
        let stem = separator.guitar_stem(&input).expect("separation runs");
        assert_eq!(stem.sample_rate, super::super::MODEL_SAMPLE_RATE);
        assert_eq!(stem.channels, 1);
        // The output length follows the decoded input duration.
        let expected = (input.frames() * super::super::MODEL_SAMPLE_RATE as usize)
            / input.sample_rate as usize;
        assert!(
            stem.samples.len().abs_diff(expected) < expected / 10,
            "expected ~{expected} samples, got {}",
            stem.samples.len()
        );
        // A guitar-only input must come back out of the guitar stem almost
        // unchanged. Compare mean power rather than absolute energy so the
        // resample's length change does not matter.
        let input_power: f32 =
            input.samples.iter().map(|s| s * s).sum::<f32>() / input.samples.len() as f32;
        let stem_power: f32 =
            stem.samples.iter().map(|s| s * s).sum::<f32>() / stem.samples.len() as f32;
        let ratio = stem_power / input_power;
        assert!(
            (0.5..=1.5).contains(&ratio),
            "the guitar stem should carry most of a guitar-only input's power, got {ratio:.2} \
             ({stem_power:.6} against {input_power:.6})"
        );
    }
}
