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
    if buf.sample_rate == 0
        || target_rate == 0
        || buf.channels == 0
        || buf.samples.len() % buf.channels as usize != 0
    {
        return Err(MtabError::UnsupportedAudio(
            "invalid sample rate, channel count, or interleaved sample count".into(),
        ));
    }
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

    let frames = buf.frames();
    let expected = ((frames as f64 * ratio).round() as usize).max(1);
    let mut samples = Vec::with_capacity(expected * channels);
    let mut pos = 0usize;

    // De-interleave one bounded window at a time, then flush zeros until the
    // filter tail has been emitted. SincFixedIn buffers its initial delay by
    // producing a shorter first block, so output indices are already aligned
    // (verified using impulses at the beginning, middle and end).
    while pos < frames || samples.len() / channels < expected {
        let end = (pos + CHUNK).min(frames);
        let mut block = vec![vec![0.0f32; CHUNK]; channels];
        for (f, frame) in buf.samples[pos * channels..end * channels]
            .chunks_exact(channels)
            .enumerate()
        {
            for (ch, &sample) in frame.iter().enumerate() {
                block[ch][f] = sample;
            }
        }
        let output = resampler
            .process(&block, None)
            .map_err(|e| MtabError::UnsupportedAudio(format!("resample: {e}")))?;
        for f in 0..output[0].len() {
            if samples.len() / channels == expected {
                break;
            }
            for channel in &output {
                samples.push(channel[f]);
            }
        }
        pos = end;
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
        // Tail flushing produces the full nominal output length.
        assert_eq!(out.frames(), 16_000);
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
    #[test]
    fn preserves_impulse_time_and_tail() {
        for &(rate, frames, at) in &[
            (48_000, 2048, 0),
            (48_000, 2048, 128),
            (48_000, 2048, 1900),
            (44_100, 4096, 2500),
            (8_000, 1024, 900),
        ] {
            let mut samples = vec![0.0; frames];
            samples[at] = 1.0;
            let out = resample(
                &AudioBuffer {
                    samples,
                    sample_rate: rate,
                    channels: 1,
                },
                16_000,
            )
            .unwrap();
            assert_eq!(
                out.frames(),
                (frames as f64 * 16_000.0 / rate as f64).round() as usize
            );
            let peak = out
                .samples
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap()
                .0;
            let target = (at as f64 * 16_000.0 / rate as f64).round() as isize;
            assert!(
                (peak as isize - target).abs() <= 1,
                "rate {rate} peak {peak}, expected {target}"
            );
            assert!(out.samples[peak].abs() > 0.05);
        }
    }

    #[test]
    fn stereo_remains_interleaved_and_bad_buffers_are_rejected() {
        let mut samples = vec![0.0; 2048 * 2];
        samples[1000 * 2] = 1.0;
        let out = resample(
            &AudioBuffer {
                samples,
                sample_rate: 48_000,
                channels: 2,
            },
            16_000,
        )
        .unwrap();
        assert!(out.samples.chunks_exact(2).any(|f| f[0].abs() > 0.05));
        assert!(out.samples.chunks_exact(2).all(|f| f[1] == 0.0));
        for bad in [
            AudioBuffer {
                samples: vec![1.0],
                sample_rate: 16_000,
                channels: 0,
            },
            AudioBuffer {
                samples: vec![1.0],
                sample_rate: 16_000,
                channels: 2,
            },
            AudioBuffer {
                samples: vec![1.0],
                sample_rate: 0,
                channels: 1,
            },
        ] {
            assert!(resample(&bad, 16_000).is_err());
        }
    }
}
