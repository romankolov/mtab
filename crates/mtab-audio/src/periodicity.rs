//! Experimental autocorrelation evidence from an onset-strength envelope.
//!
//! This is not beat, downbeat, meter or tempo detection. Peaks can reflect
//! subdivisions or multiples of a musical beat and are not confidence scores.
use crate::onset::{OnsetEnvelope, MAX_ONSET_FRAMES, ONSET_HOP};
use crate::MODEL_SAMPLE_RATE;
use mtab_core::{MtabError, Result};

/// Smallest analyzed lag (25 frames = 0.25 s = 240 BPM).
pub const MIN_LAG: usize = 25;
/// Largest analyzed lag (150 frames = 1.5 s = 40 BPM).
pub const MAX_LAG: usize = 150;
/// Minimum number of 10 ms frames required for periodicity analysis.
pub const MIN_PERIODICITY_FRAMES: usize = 300;

/// One discrete autocorrelation lag; values do not indicate a probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LagEvidence {
    /// Lag in 10 ms frames.
    pub lag: usize,
    /// `6000 / lag` BPM, without interpolation or metrical interpretation.
    pub bpm: f64,
    /// Global-mean-centered, overlap-energy-normalized autocorrelation.
    pub correlation: f64,
    /// A positive local maximum within the searched lag range.
    pub local_maximum: bool,
}

/// Scores for integer lags corresponding to 40–240 BPM, in ascending lag order.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicityEvidence {
    /// Empty when fewer than 300 frames are available or the envelope is flat.
    pub lags: Vec<LagEvidence>,
}

impl PeriodicityEvidence {
    /// Positive local maxima ordered by descending correlation, then shortest
    /// lag. This ordering is not a calibrated ranking of musical beat tempi.
    pub fn candidates(&self) -> Vec<LagEvidence> {
        let mut peaks: Vec<_> = self
            .lags
            .iter()
            .filter(|e| e.local_maximum)
            .copied()
            .collect();
        peaks.sort_by(|a, b| {
            b.correlation
                .total_cmp(&a.correlation)
                .then(a.lag.cmp(&b.lag))
        });
        peaks
    }
}

/// Compute a bounded autocorrelation curve for a 10 ms onset envelope.
///
/// Normalize by the envelope's peak, subtract the **global** mean, then for
/// each lag 25..=150 compute `dot(x[..n-lag], x[lag..])` divided by the
/// product of the two segment L2 norms. A zero denominator maps to zero; scores
/// are clamped to [-1, 1] for floating-point rounding. Flat or short inputs
/// return no evidence. The maximum acceptable envelope has 30,000 frames.
///
/// The output is evidence at discrete lags, with resolution depending on
/// tempo. It has no tempo prior, metrical interpretation, phase, beat times,
/// uncertainty calibration, tempo-change tracking, or downbeat estimate.
/// A positive local peak can be caused by noise, a subdivision, or any regular
/// activity, even in non-musical audio. The optional experimental CLI tracker
/// uses the strongest peak, with known substantial errors.
pub fn autocorrelation(envelope: &OnsetEnvelope) -> Result<PeriodicityEvidence> {
    let x = &envelope.strength;
    if x.len() > MAX_ONSET_FRAMES || x.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(MtabError::UnsupportedAudio(
            "periodicity requires a finite nonnegative onset envelope of at most 30000 frames"
                .into(),
        ));
    }
    let empty = || PeriodicityEvidence { lags: Vec::new() };
    if x.len() < MIN_PERIODICITY_FRAMES {
        return Ok(empty());
    }
    let peak = x.iter().copied().fold(0.0_f64, f64::max);
    if peak == 0.0 {
        return Ok(empty());
    }
    let mean = x.iter().map(|v| *v / peak).sum::<f64>() / x.len() as f64;
    let centered: Vec<f64> = x.iter().map(|v| *v / peak - mean).collect();
    if centered.iter().all(|&v| v == 0.0) {
        return Ok(empty());
    }
    let mut lags = Vec::with_capacity(MAX_LAG - MIN_LAG + 1);
    for lag in MIN_LAG..=MAX_LAG {
        let mut cross = 0.0;
        let mut left_sq = 0.0;
        let mut right_sq = 0.0;
        for (&left, &right) in centered[..centered.len() - lag]
            .iter()
            .zip(&centered[lag..])
        {
            cross += left * right;
            left_sq += left * left;
            right_sq += right * right;
        }
        let denominator = left_sq.sqrt() * right_sq.sqrt();
        let correlation = if denominator == 0.0 {
            0.0
        } else {
            (cross / denominator).clamp(-1.0, 1.0)
        };
        lags.push(LagEvidence {
            lag,
            bpm: 60.0 * f64::from(MODEL_SAMPLE_RATE) / (ONSET_HOP * lag) as f64,
            correlation,
            local_maximum: false,
        });
    }
    for i in 0..lags.len() {
        let value = lags[i].correlation;
        let left_ok = i == 0 || value > lags[i - 1].correlation;
        let right_ok = i + 1 == lags.len() || value >= lags[i + 1].correlation;
        lags[i].local_maximum = value > 0.0 && left_ok && right_ok;
    }
    Ok(PeriodicityEvidence { lags })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_click_envelopes_expose_harmonic_ambiguity() {
        let mut x = vec![0.0; 1000];
        for i in (10..x.len()).step_by(50) {
            x[i] = 1.0;
        }
        let evidence = autocorrelation(&OnsetEnvelope { strength: x }).unwrap();
        let peaks = evidence.candidates();
        for lag in [50, 100, 150] {
            let peak = peaks.iter().find(|item| item.lag == lag).unwrap();
            assert!((peak.correlation - 1.0).abs() < 1e-12);
        }
        assert!((peaks.iter().find(|p| p.lag == 50).unwrap().bpm - 120.0).abs() < 1e-12);
    }

    #[test]
    fn empty_flat_and_invalid_envelopes() {
        for strength in [vec![], vec![0.0; 1000], vec![4.0; 1000], vec![1.0; 299]] {
            assert!(autocorrelation(&OnsetEnvelope { strength })
                .unwrap()
                .lags
                .is_empty());
        }
        for v in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(autocorrelation(&OnsetEnvelope { strength: vec![v] }).is_err());
        }
        assert!(autocorrelation(&OnsetEnvelope {
            strength: vec![0.0; MAX_ONSET_FRAMES + 1]
        })
        .is_err());
    }

    #[test]
    fn constant_gain_does_not_change_scores() {
        let strength: Vec<f64> = (0..500).map(|i| ((i * 71) % 137) as f64 / 137.0).collect();
        let original = autocorrelation(&OnsetEnvelope {
            strength: strength.clone(),
        })
        .unwrap();
        let scaled = autocorrelation(&OnsetEnvelope {
            strength: strength.iter().map(|v| v * 1000.0).collect(),
        })
        .unwrap();
        assert_eq!(original.lags.len(), 126);
        for (a, b) in original.lags.iter().zip(scaled.lags.iter()) {
            assert!((a.correlation - b.correlation).abs() < 1e-12);
            assert_eq!(a.local_maximum, b.local_maximum);
        }
    }
}
