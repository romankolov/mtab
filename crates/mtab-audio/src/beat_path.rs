//! Experimental tempo-conditioned beat-path search.
//!
//! The dynamic-programming transition penalty is based on the formulation in
//! D. P. W. Ellis, "Beat Tracking by Dynamic Programming", JNMR 2007,
//! <https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf>.
//! This is a simplified, independently implemented objective, not a port of
//! any complete tracker and not a claim of parity with MuScriptor or librosa.
use crate::onset::{OnsetEnvelope, MAX_ONSET_FRAMES, ONSET_HOP};
use crate::MODEL_SAMPLE_RATE;
use mtab_core::{MtabError, Result};

/// A tempo-conditioned path of frame positions, not a confidence assessment.
#[derive(Debug, Clone, PartialEq)]
pub struct BeatPath {
    /// Indices into the input onset envelope, in increasing order.
    pub frames: Vec<usize>,
    /// Center times corresponding to `frames`, in seconds.
    pub times: Vec<f64>,
    /// Target period after rounding to an integer number of 10 ms frames.
    pub target_period_frames: usize,
    /// Objective value (sum of normalized strengths minus gap penalties).
    /// This is length-dependent and is not a probability or confidence.
    pub objective: f64,
}

/// Search for a beat path given an explicit, constant target tempo.
///
/// Valid tempo is 40–240 BPM and `tightness` is finite in 0.01–1000.0. The
/// period is `round(6000 / BPM)` frames and allowed gaps are from half (rounded
/// up) to twice that period. Input onset strengths are divided by their maximum.
/// The score of a path is the sum of its normalized onset strengths minus
/// `tightness * ln(gap / period)^2` for each adjacent pair. The highest-scoring
/// nonempty path is found by dynamic programming; equal scores prefer the
/// earliest terminal frame and, for positive predecessor ties, the smallest
/// gap. A predecessor whose score after penalty is zero is not linked.
///
/// Silence, constant envelopes and input shorter than two target periods yield
/// an empty path. Peaks may represent subdivisions or non-musical noise, and
/// the objective may insert zero-strength beats between strong peaks; the
/// output is not confidence-calibrated. This function does not estimate tempo,
/// downbeats or meter. It is not wired to CLI output, and assumes a fixed
/// global period rather than tracking tempo changes. Timestamps have 10 ms
/// resolution and refer to window centers, not corrected attack times.
pub fn track_with_tempo(envelope: &OnsetEnvelope, bpm: f64, tightness: f64) -> Result<BeatPath> {
    if !bpm.is_finite() || !(40.0..=240.0).contains(&bpm) {
        return Err(MtabError::InvalidTempo(
            "target tempo must be 40–240 BPM".into(),
        ));
    }
    if !tightness.is_finite() || !(0.01..=1000.0).contains(&tightness) {
        return Err(MtabError::InvalidTempo(
            "beat-path tightness must be 0.01–1000".into(),
        ));
    }
    let input = &envelope.strength;
    if input.len() > MAX_ONSET_FRAMES || input.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(MtabError::UnsupportedAudio(
            "beat-path input must be a finite nonnegative onset envelope of at most 30000 frames"
                .into(),
        ));
    }
    let period = (60.0 * f64::from(MODEL_SAMPLE_RATE) / (ONSET_HOP as f64 * bpm)).round() as usize;
    let empty = || BeatPath {
        frames: vec![],
        times: vec![],
        target_period_frames: period,
        objective: 0.0,
    };
    if input.len() < 2 * period || input.iter().all(|&v| v == input[0]) {
        return Ok(empty());
    }
    let max = input.iter().copied().fold(0.0_f64, f64::max);
    if max == 0.0 {
        return Ok(empty());
    }
    let signal: Vec<f64> = input.iter().map(|v| *v / max).collect();
    let (frames, objective) = best_path(&signal, period, tightness);
    let times = frames
        .iter()
        .map(|&i| envelope.frame_time(i).expect("validated frame"))
        .collect();
    Ok(BeatPath {
        frames,
        times,
        target_period_frames: period,
        objective,
    })
}

fn best_path(signal: &[f64], period: usize, tightness: f64) -> (Vec<usize>, f64) {
    let min_gap = period.div_ceil(2);
    let max_gap = period * 2;
    let penalties: Vec<f64> = (min_gap..=max_gap)
        .map(|gap| tightness * ((gap as f64 / period as f64).ln()).powi(2))
        .collect();
    let mut scores = vec![0.0; signal.len()];
    let mut links = vec![None; signal.len()];
    let mut best_end = 0;
    for i in 0..signal.len() {
        let mut best_previous = 0.0;
        let mut best_index = None;
        for gap in min_gap..=max_gap.min(i) {
            let j = i - gap;
            let score = scores[j] - penalties[gap - min_gap];
            if score > best_previous {
                best_previous = score;
                best_index = Some(j);
            }
        }
        scores[i] = signal[i] + best_previous;
        links[i] = best_index;
        if scores[i] > scores[best_end] {
            best_end = i;
        }
    }
    let mut frames = Vec::new();
    let mut current = Some(best_end);
    while let Some(i) = current {
        frames.push(i);
        current = links[i];
    }
    frames.reverse();
    (frames, scores[best_end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_clear_periodic_envelopes_and_keeps_alignment() {
        for period in [30, 50, 100, 150] {
            let mut strength = vec![0.0; 1100];
            let expected: Vec<usize> = (20..strength.len()).step_by(period).collect();
            for &i in &expected {
                strength[i] = 1.0;
            }
            let env = OnsetEnvelope { strength };
            let path = track_with_tempo(&env, 6000.0 / period as f64, 3.0).unwrap();
            assert_eq!(path.frames, expected, "period={period}");
            assert_eq!(path.times[0], 0.232);
            assert!((path.objective - expected.len() as f64).abs() < 1e-9);
        }
    }

    #[test]
    fn handles_empty_flat_short_invalid_and_ambiguous() {
        for strength in [vec![], vec![0.0; 500], vec![1.0; 500], vec![1.0; 20]] {
            assert!(track_with_tempo(&OnsetEnvelope { strength }, 120.0, 3.0)
                .unwrap()
                .frames
                .is_empty());
        }
        let env = OnsetEnvelope {
            strength: vec![0.0; 400],
        };
        for bpm in [0.0, 39.0, 241.0, f64::NAN, f64::INFINITY] {
            assert!(track_with_tempo(&env, bpm, 3.0).is_err());
        }
        for weight in [0.0, -1.0, 1001.0, f64::NAN] {
            assert!(track_with_tempo(&env, 120.0, weight).is_err());
        }
        for value in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(track_with_tempo(
                &OnsetEnvelope {
                    strength: vec![value]
                },
                120.0,
                3.0
            )
            .is_err());
        }
        assert!(track_with_tempo(
            &OnsetEnvelope {
                strength: vec![0.0; MAX_ONSET_FRAMES + 1]
            },
            120.0,
            3.0
        )
        .is_err());
        // One isolated transient does not justify inferring a full beat grid.
        let mut single = vec![0.0; 500];
        single[100] = 1.0;
        assert_eq!(
            track_with_tempo(&OnsetEnvelope { strength: single }, 120.0, 3.0)
                .unwrap()
                .frames,
            [100]
        );
    }

    #[test]
    fn matches_exhaustive_search_on_short_sequences() {
        let mut seed = 0x3141_5926_u64;
        for _ in 0..500 {
            let mut strength = vec![0.0; 9];
            for value in &mut strength {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *value = ((seed >> 32) % 10) as f64 / 10.0;
            }
            let period = 2 + (seed % 3) as usize;
            let tightness = 0.7;
            let (path, score) = best_path(&strength, period, tightness);
            let mut exhaustive = 0.0_f64;
            for mask in 1..(1 << strength.len()) {
                let indices: Vec<usize> = (0..strength.len())
                    .filter(|i| mask & (1 << i) != 0)
                    .collect();
                if indices.windows(2).any(|pair| {
                    let gap = pair[1] - pair[0];
                    gap < period.div_ceil(2) || gap > 2 * period
                }) {
                    continue;
                }
                let mut candidate = indices.iter().map(|&i| strength[i]).sum::<f64>();
                for pair in indices.windows(2) {
                    candidate -=
                        tightness * (((pair[1] - pair[0]) as f64 / period as f64).ln()).powi(2);
                }
                exhaustive = exhaustive.max(candidate);
            }
            assert!((score - exhaustive).abs() < 1e-12, "{strength:?} {path:?}");
            let mut reconstructed = path.iter().map(|&i| strength[i]).sum::<f64>();
            for pair in path.windows(2) {
                reconstructed -=
                    tightness * (((pair[1] - pair[0]) as f64 / period as f64).ln()).powi(2);
            }
            assert!((score - reconstructed).abs() < 1e-12);
        }
    }
}
