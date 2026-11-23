//! Experimental fixed-tempo automatic beat hypothesis.
//!
//! This combines [`crate::periodicity::autocorrelation`] and
//! [`crate::beat_path::track_with_tempo`]. It is not a reliable or calibrated
//! tracker; see the documented GuitarSet diagnostic and offbeat failure.
use crate::{
    beat_path::{track_with_tempo, BeatPath},
    onset::{spectral_flux, OnsetEnvelope},
    periodicity::{autocorrelation, LagEvidence},
    AudioBuffer,
};
use mtab_core::Result;

/// Transition tightness selected on the development set described in
/// `benches/m4-beat-tightness-results.md`; not a confidence score.
pub const EXPERIMENTAL_TIGHTNESS: f64 = 30.0;

/// A single automatic hypothesis, with no confidence or meter estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct AutomaticBeatHypothesis {
    /// Strongest positive autocorrelation local maximum; may be a subdivision.
    pub tempo_evidence: LagEvidence,
    /// Dynamic-programming path conditioned on that one global tempo.
    pub path: BeatPath,
}

/// Analyze a mono 16 kHz signal with the bounded onset frontend, then select
/// one automatic beat hypothesis. Inputs above 30,000 frames are rejected by
/// the onset frontend rather than truncated; this does not bound the input
/// audio buffer itself. See [`automatic_beats_from_envelope`] for limitations.
pub fn automatic_beats(audio: &AudioBuffer) -> Result<Option<AutomaticBeatHypothesis>> {
    automatic_beats_from_envelope(&spectral_flux(audio)?)
}

/// Choose the strongest autocorrelation candidate and search its beat path.
///
/// Returns `None` if there is no candidate or fewer than two detected beats.
/// Non-silent noise, musical subdivisions and stronger offbeats may still
/// generate an incorrect hypothesis, even when this returns `Some`. The
/// selected period is global, so it does not track tempo changes reliably;
/// beat times can deviate locally because of the path search. No downbeats,
/// meter, confidence, silence detector or abstention calibration are provided.
/// Candidate and path scores are not probabilities. This is opt-in and should
/// be manually reviewed; it is not equivalent to upstream MuScriptor tracking.
pub fn automatic_beats_from_envelope(
    envelope: &OnsetEnvelope,
) -> Result<Option<AutomaticBeatHypothesis>> {
    let Some(tempo_evidence) = autocorrelation(envelope)?.candidates().into_iter().next() else {
        return Ok(None);
    };
    let path = track_with_tempo(envelope, tempo_evidence.bpm, EXPERIMENTAL_TIGHTNESS)?;
    if path.times.len() < 2 {
        return Ok(None);
    }
    Ok(Some(AutomaticBeatHypothesis {
        tempo_evidence,
        path,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypothesis_matches_composition_and_is_deterministic() {
        let mut envelope = OnsetEnvelope {
            strength: vec![0.0; 1000],
        };
        for i in (20..1000).step_by(50) {
            envelope.strength[i] = 1.0;
        }
        let result = automatic_beats_from_envelope(&envelope).unwrap().unwrap();
        let candidate = autocorrelation(&envelope).unwrap().candidates()[0];
        assert_eq!(result.tempo_evidence, candidate);
        assert_eq!(
            result.path,
            track_with_tempo(&envelope, candidate.bpm, 30.0).unwrap()
        );
        assert_eq!(
            Some(result),
            automatic_beats_from_envelope(&envelope).unwrap()
        );
    }

    #[test]
    fn absence_of_evidence_does_not_invent_beats() {
        for strength in [vec![], vec![0.0; 1000], vec![1.0; 1000], vec![0.0; 299]] {
            assert_eq!(
                automatic_beats_from_envelope(&OnsetEnvelope { strength }).unwrap(),
                None
            );
        }
        assert!(automatic_beats_from_envelope(&OnsetEnvelope {
            strength: vec![f64::NAN; 400]
        })
        .is_err());
    }

    #[test]
    fn audio_frontend_rejects_wrong_rate_and_silence() {
        let mut audio = AudioBuffer {
            samples: vec![0.0; 16_000 * 4],
            sample_rate: 16_000,
            channels: 1,
        };
        assert_eq!(automatic_beats(&audio).unwrap(), None);
        audio.sample_rate = 44_100;
        assert!(automatic_beats(&audio).is_err());
    }
}
