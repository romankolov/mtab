//! Audio file decoding via `symphonia` (PROMPT.md §6.1): wav/mp3/flac/ogg/m4a
//! → interleaved [`AudioBuffer`].

use std::fs::File;
use std::path::Path;

use symphonia::core::audio::{AudioBuffer, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use mtab_core::{MtabError, Result};

use crate::AudioBuffer as MtabBuffer;

/// Decode an audio file to interleaved f32 samples at the file's native rate.
pub fn load(path: &Path) -> Result<MtabBuffer> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| MtabError::UnsupportedAudio(format!("{}: {e}", path.display())))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| {
            MtabError::UnsupportedAudio(format!("{}: no decodable track", path.display()))
        })?;

    let track_id = track.id;
    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(44_100);
    let mut channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| MtabError::UnsupportedAudio(format!("decoder: {e}")))?;

    let mut samples: Vec<f32> = Vec::new();
    let mut chunk_index = 0usize;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(MtabError::DecodeFailure {
                    chunk_index,
                    reason: "stream reset not supported".to_owned(),
                })
            }
            Err(e) => {
                return Err(MtabError::DecodeFailure {
                    chunk_index,
                    reason: e.to_string(),
                })
            }
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::DecodeError(_)) => {
                // Skip malformed packets, per symphonia guidance.
                chunk_index += 1;
                continue;
            }
            Err(e) => {
                return Err(MtabError::DecodeFailure {
                    chunk_index,
                    reason: e.to_string(),
                })
            }
        };

        let spec = *decoded.spec();
        sample_rate = spec.rate;
        channels = spec.channels.count() as u16;

        let mut buf = AudioBuffer::<f32>::new(decoded.capacity() as u64, spec);
        decoded.convert(&mut buf);

        let planes = buf.planes();
        let planes = planes.planes();
        let frames = buf.frames();
        samples.reserve(frames * channels as usize);
        for f in 0..frames {
            for plane in planes {
                samples.push(plane[f]);
            }
        }
        chunk_index += 1;
    }

    if samples.is_empty() {
        return Err(MtabError::UnsupportedAudio(format!(
            "{}: decoded zero samples",
            path.display()
        )));
    }

    Ok(MtabBuffer {
        samples,
        sample_rate,
        channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::write_wav;
    use std::path::PathBuf;

    fn sine_wav(sr: u32, secs: f32, freq: f32) -> PathBuf {
        let n = (sr as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();
        write_wav(&samples, sr)
    }

    #[test]
    fn decodes_wav_with_native_rate() {
        let path = sine_wav(48_000, 0.5, 440.0);
        let buf = load(&path).unwrap();
        assert_eq!(buf.sample_rate, 48_000);
        assert_eq!(buf.channels, 1);
        assert_eq!(buf.frames(), 24_000);
        assert!(buf.samples.iter().any(|s| *s > 0.5));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_file_is_weights_not_found_style_error() {
        assert!(load(Path::new("/nonexistent/nope.wav")).is_err());
    }
}
