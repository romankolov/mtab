//! High-quality sample-rate conversion via `rubato` (PROMPT.md §6.1).
//! Everything is converted to the model rate (16 kHz upstream) here.

use mtab_core::{MtabError, Result};
use rubato::{
    Resampler as _, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::AudioBuffer;

const CHUNK: usize = 1024;

/// Resample to `target_rate` Hz using a windowed-sinc resampler.
pub fn resample(buf: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
    if buf.sample_rate == target_rate {
        return Ok(buf.clone());
    }
    if buf.samples.is_empty() {
        return Ok(AudioBuffer {
            samples: Vec::new(),
            sample_rate: target_rate,
            channels: buf.channels,
        });
    }

    let channels = buf.channels as usize;
    let ratio = target_rate as f64 / buf.sample_rate as f64;

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 160,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(ratio, 1.0, params, CHUNK, channels)
        .map_err(|e| MtabError::UnsupportedAudio(format!("resampler init: {e}")))?;

    // De-interleave into per-channel planes.
    let frames = buf.frames();
    let mut planes: Vec<Vec<f32>> = vec![Vec::with_capacity(frames + CHUNK); channels];
    for f in 0..frames {
        for (c, plane) in planes.iter_mut().enumerate() {
            plane.push(buf.samples[f * channels + c]);
        }
    }

    let mut out_planes: Vec<Vec<f32>> = vec![Vec::new(); channels];
    let mut pos = 0usize;
    while pos < frames {
        let end = (pos + CHUNK).min(frames);
        let chunk_len = end - pos;
        let mut input_refs: Vec<&[f32]> = Vec::with_capacity(channels);
        for plane in &planes {
            input_refs.push(&plane[pos..end]);
        }
        // Pad the final partial chunk with zeros; excess output is trimmed below.
        if chunk_len < CHUNK {
            let mut padded: Vec<Vec<f32>> = planes.iter().map(|p| p[pos..end].to_vec()).collect();
            for p in &mut padded {
                p.resize(CHUNK, 0.0);
            }
            let input_refs: Vec<&[f32]> = padded.iter().map(|p| p.as_slice()).collect();
            let out = resampler
                .process(&input_refs, None)
                .map_err(|e| MtabError::UnsupportedAudio(format!("resample: {e}")))?;
            for (o, plane) in out.into_iter().zip(out_planes.iter_mut()) {
                plane.extend_from_slice(&o);
            }
        } else {
            let out = resampler
                .process(&input_refs, None)
                .map_err(|e| MtabError::UnsupportedAudio(format!("resample: {e}")))?;
            for (o, plane) in out.into_iter().zip(out_planes.iter_mut()) {
                plane.extend_from_slice(&o);
            }
        }
        pos = end;
    }

    // Expected output length from the nominal ratio; trim resampler tail/padding.
    let expected = ((frames as f64 * ratio).round() as usize).max(1);
    let out_frames = out_planes[0].len().min(expected);

    let mut samples = Vec::with_capacity(out_frames * channels);
    for f in 0..out_frames {
        for plane in &out_planes {
            samples.push(plane[f]);
        }
    }

    Ok(AudioBuffer {
        samples,
        sample_rate: target_rate,
        channels: buf.channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(sr: u32, secs: f32, freq: f32) -> AudioBuffer {
        let n = (sr as f32 * secs) as usize;
        AudioBuffer {
            samples: (0..n)
                .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
                .collect(),
            sample_rate: sr,
            channels: 1,
        }
    }

    #[test]
    fn no_op_on_matching_rate() {
        let buf = sine(16_000, 0.1, 440.0);
        let out = resample(&buf, 16_000).unwrap();
        assert_eq!(out.samples, buf.samples);
    }

    #[test]
    fn converts_48k_to_16k() {
        let buf = sine(48_000, 1.0, 440.0);
        let out = resample(&buf, 16_000).unwrap();
        assert_eq!(out.sample_rate, 16_000);
        // Length ≈ 1/3 of input (allow resampler group delay trimming).
        assert!(
            (out.frames() as i64 - 16_000).abs() <= 64,
            "frames={}",
            out.frames()
        );
        // Energy survives the conversion.
        let rms = (out.samples.iter().map(|s| (*s as f64).powi(2)).sum::<f64>()
            / out.samples.len() as f64)
            .sqrt();
        assert!(rms > 0.3, "rms={rms}");
    }

    #[test]
    fn handles_partial_final_chunk() {
        let buf = sine(44_100, 0.13, 220.0); // awkward length
        let out = resample(&buf, 16_000).unwrap();
        assert!(out.frames() > 2_000 && out.frames() < 2_200);
    }
}
