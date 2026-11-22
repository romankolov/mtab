//! Conservative tuning-template scoring over absolute-pitch evidence.
//!
//! This scores the hypothesis that observed pitches include open strings; it
//! cannot prove tuning or capo. In particular, fretted notes can masquerade as
//! open strings and different tuning/capo combinations can sound identical.
//! Confidence thresholds are heuristics, not calibrated probabilities.

use mtab_core::{MtabError, Result};

use crate::{Capo, Tuning};

/// A tuning and capo hypothesis. Candidates are supplied by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningCandidate {
    pub tuning: Tuning,
    pub capo: Capo,
}

/// One candidate and the fraction of evidence matching an effective open pitch.
#[derive(Debug, Clone, PartialEq)]
pub struct TuningScore {
    pub candidate: TuningCandidate,
    /// Fraction of all observed pitch weight on effective open strings, 0–1.
    pub open_fraction: f64,
    /// Number of distinct open pitches with at least half a frame of evidence.
    pub matched_open_pitches: usize,
}

/// Ranked hypotheses and a conservative, uncalibrated sufficiency decision.
#[derive(Debug, Clone, PartialEq)]
pub struct TuningDetection {
    /// Highest-scoring candidate, if any evidence exists.
    pub best: Option<TuningScore>,
    /// Next candidate, including acoustically equivalent alternatives.
    pub runner_up: Option<TuningScore>,
    /// Best-minus-runner-up open fraction (0 for ties); not a probability.
    pub margin: f64,
    /// Meets minimum evidence and separation thresholds; still only a guess.
    pub sufficient_evidence: bool,
}

/// Cross product of the built-in tuning library with capo positions 0–12.
///
/// Templates with different names or capo positions are intentionally kept
/// separate even when their effective open pitches are identical, so ambiguity
/// cannot accidentally be reported as high confidence.
pub fn standard_candidates() -> Vec<TuningCandidate> {
    Tuning::library()
        .into_iter()
        .flat_map(|tuning| {
            (0..=12).map(move |n| TuningCandidate {
                tuning: tuning.clone(),
                capo: Capo::new(n).expect("capo within range"),
            })
        })
        .collect()
}

/// Rank candidates against weighted MIDI-pitch evidence from an audio frontend.
///
/// A candidate scores the fraction of observed weight at its sounding open
/// string pitches. `sufficient_evidence` requires >=8 total weight, >=4
/// matched open pitches, >=75% open fraction, at least two candidates and a
/// >=0.10 lead over the next candidate. These are conservative but **uncalibrated** thresholds and assume
/// the recorded phrase actually includes open strings. Silence and equivalent
/// or near-tied templates abstain. Arbitrary spectra, chords and fretted notes
/// can fool this heuristic even when it passes the threshold.
pub fn detect_tuning(
    weights: &[f64; 128],
    candidates: &[TuningCandidate],
) -> Result<TuningDetection> {
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(MtabError::InvalidTuning(
            "pitch evidence must be finite and nonnegative".into(),
        ));
    }
    for candidate in candidates {
        if !(Tuning::MIN_STRINGS..=Tuning::MAX_STRINGS).contains(&candidate.tuning.strings.len())
            || !candidate.tuning.strings.windows(2).all(|w| w[0] < w[1])
            || candidate
                .tuning
                .strings
                .iter()
                .any(|&p| u16::from(p) + u16::from(candidate.capo.position()) > 127)
        {
            return Err(MtabError::InvalidTuning(
                "candidate needs 6–8 ascending MIDI strings within range after capo".into(),
            ));
        }
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite() {
        return Err(MtabError::InvalidTuning(
            "pitch evidence total overflowed".into(),
        ));
    }
    if total == 0.0 || candidates.is_empty() {
        return Ok(TuningDetection {
            best: None,
            runner_up: None,
            margin: 0.0,
            sufficient_evidence: false,
        });
    }
    let mut scored: Vec<TuningScore> = candidates
        .iter()
        .map(|candidate| {
            let open: Vec<f64> = candidate
                .tuning
                .strings
                .iter()
                .map(|&pitch| weights[usize::from(pitch) + usize::from(candidate.capo.position())])
                .collect();
            TuningScore {
                candidate: candidate.clone(),
                open_fraction: open.iter().sum::<f64>() / total,
                matched_open_pitches: open.iter().filter(|&&w| w >= 0.5).count(),
            }
        })
        .collect();
    // Stable ordering retains caller template order on an exact score tie.
    scored.sort_by(|a, b| b.open_fraction.total_cmp(&a.open_fraction));
    let best = scored.first().cloned();
    let runner_up = scored.get(1).cloned();
    let margin =
        best.as_ref().unwrap().open_fraction - runner_up.as_ref().map_or(0.0, |s| s.open_fraction);
    let sufficient_evidence = runner_up.is_some()
        && total >= 8.0
        && best.as_ref().unwrap().matched_open_pitches >= 4
        && best.as_ref().unwrap().open_fraction >= 0.75
        && margin >= 0.10;
    Ok(TuningDetection {
        best,
        runner_up,
        margin,
        sufficient_evidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn candidate(tuning: Tuning, capo: u8) -> TuningCandidate {
        TuningCandidate {
            tuning,
            capo: Capo::new(capo).unwrap(),
        }
    }
    fn evidence(pitches: &[u8]) -> [f64; 128] {
        let mut w = [0.0; 128];
        for &p in pitches {
            w[usize::from(p)] += 3.0;
        }
        w
    }

    #[test]
    fn empty_sparse_and_invalid_evidence_abstain() {
        let c = standard_candidates();
        assert!(!detect_tuning(&[0.0; 128], &c).unwrap().sufficient_evidence);
        assert!(
            !detect_tuning(&evidence(&[40, 45]), &c)
                .unwrap()
                .sufficient_evidence
        );
        assert!(
            !detect_tuning(
                &evidence(&Tuning::standard().strings),
                &[candidate(Tuning::standard(), 0)]
            )
            .unwrap()
            .sufficient_evidence
        );
        let mut invalid = [0.0; 128];
        invalid[0] = f64::NAN;
        assert!(detect_tuning(&invalid, &c).is_err());
        invalid[0] = -1.0;
        assert!(detect_tuning(&invalid, &c).is_err());
    }
    #[test]
    fn shared_strings_and_equivalent_capo_are_ambiguous() {
        let w = evidence(&Tuning::standard().strings);
        let alternatives = [
            candidate(Tuning::standard(), 0),
            candidate(Tuning::seven_string(), 0),
            candidate(Tuning::eight_string(), 0),
        ];
        let d = detect_tuning(&w, &alternatives).unwrap();
        assert_eq!(d.margin, 0.0);
        assert!(!d.sufficient_evidence);
        let w = evidence(&[40, 45, 50, 55, 59, 64]);
        let alternatives = [
            candidate(Tuning::standard(), 0),
            candidate(Tuning::half_step_down(), 1),
        ];
        let d = detect_tuning(&w, &alternatives).unwrap();
        assert_eq!(d.margin, 0.0);
        assert!(!d.sufficient_evidence);
    }
    #[test]
    fn ranks_restricted_known_tuning_and_capo() {
        let w = evidence(&[42, 47, 52, 57, 61, 66]);
        let options = [
            candidate(Tuning::standard(), 0),
            candidate(Tuning::standard(), 1),
            candidate(Tuning::standard(), 2),
        ];
        let d = detect_tuning(&w, &options).unwrap();
        assert!(d.sufficient_evidence);
        assert_eq!(d.best.unwrap().candidate.capo.position(), 2);
    }
    #[test]
    fn detects_distinct_low_string_evidence_with_full_library() {
        let w = evidence(&Tuning::eight_string().strings);
        let d = detect_tuning(&w, &standard_candidates()).unwrap();
        assert!(d.sufficient_evidence, "{d:?}");
        let best = d.best.unwrap().candidate;
        assert_eq!(best.tuning, Tuning::eight_string());
        assert_eq!(best.capo, Capo::NONE);
    }
    #[test]
    fn impossible_template_is_rejected_without_panicking() {
        let mut c = candidate(Tuning::standard(), 0);
        c.tuning.strings[5] = 200;
        assert!(detect_tuning(&evidence(&[40]), &[c]).is_err());
    }
}
