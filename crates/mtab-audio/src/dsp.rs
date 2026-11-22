//! Basic DSP steps (PROMPT.md §6.1): mono mixdown, DC removal, peak
//! normalization. Order per spec: mixdown → DC → normalize → resample.

use crate::AudioBuffer;

/// Average all channels to mono (§6.1 documented rule).
pub fn to_mono(buf: &AudioBuffer) -> AudioBuffer {
    if buf.channels <= 1 {
        return buf.clone();
    }
    let ch = buf.channels as usize;
    let frames = buf.frames();
    let mut samples = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut acc = 0.0f32;
        for c in 0..ch {
            acc += buf.samples[f * ch + c];
        }
        samples.push(acc / ch as f32);
    }
    AudioBuffer {
        samples,
        sample_rate: buf.sample_rate,
        channels: 1,
    }
}

/// Remove the DC component (subtract the mean).
pub fn remove_dc(buf: &mut AudioBuffer) {
    if buf.samples.is_empty() {
        return;
    }
    let mean = buf.samples.iter().map(|s| *s as f64).sum::<f64>() / buf.samples.len() as f64;
    for s in &mut buf.samples {
        *s -= mean as f32;
    }
}

/// Scale so the peak amplitude is `peak_dbfs` (default −1 dBFS per §6.1).
/// No-op on digital silence.
pub fn peak_normalize(buf: &mut AudioBuffer, peak_dbfs: f32) {
    let peak = buf.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak <= f32::EPSILON {
        return;
    }
    let target = 10f32.powf(peak_dbfs / 20.0);
    let gain = target / peak;
    for s in &mut buf.samples {
        *s *= gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo(l: f32, r: f32) -> AudioBuffer {
        AudioBuffer {
            samples: vec![l, r, l, r],
            sample_rate: 16_000,
            channels: 2,
        }
    }

    #[test]
    fn mono_mixdown_averages_channels() {
        let mono = to_mono(&stereo(0.5, -0.5));
        assert_eq!(mono.channels, 1);
        assert!(mono.samples.iter().all(|s| s.abs() < 1e-6));
        let mono = to_mono(&stereo(0.2, 0.4));
        assert!((mono.samples[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn dc_removal_zeroes_constant() {
        let mut buf = stereo(0.25, 0.25);
        buf.channels = 1;
        remove_dc(&mut buf);
        assert!(buf.samples.iter().all(|s| s.abs() < 1e-6));
    }

    #[test]
    fn normalize_hits_target_peak() {
        let mut buf = AudioBuffer {
            samples: vec![0.1, -0.2, 0.05],
            sample_rate: 16_000,
            channels: 1,
        };
        peak_normalize(&mut buf, -1.0);
        let peak = buf.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!((peak - 10f32.powf(-1.0 / 20.0)).abs() < 1e-5);
    }

    #[test]
    fn silence_untouched() {
        let mut buf = AudioBuffer {
            samples: vec![0.0; 8],
            sample_rate: 16_000,
            channels: 1,
        };
        peak_normalize(&mut buf, -1.0);
        assert!(buf.samples.iter().all(|s| *s == 0.0));
    }
}
