//! Validated, piecewise-constant tempo timelines and supplied beat times.
//!
//! This module does not detect beats, downbeats, time signatures or confidence
//! from audio. A tempo map defines seconds per quarter note, not note timing.
use crate::{TempoChange, TempoMap};
use mtab_core::{MtabError, Result};

const MAX_BEAT_TIMES: usize = 100_001;

fn invalid(reason: &str) -> MtabError {
    MtabError::InvalidTempo(reason.into())
}

impl TempoMap {
    /// Build one piecewise-constant tempo per supplied quarter-note interval.
    ///
    /// At least two and at most 100,001 finite, strictly increasing times are
    /// required. The first beat must be exactly at time zero; no first-downbeat
    /// offset, pickup, meter, or subdivision inference is performed. Each
    /// interval must be MIDI-representable (1–16,777,215 µs/quarter). Tempo
    /// from the last interval continues after the last beat. The supplied times
    /// must already be trusted quarter beats, not eighth notes or downbeats.
    /// This is a conversion utility, not automatic beat detection.
    pub fn from_quarter_beat_times(times: &[f64]) -> Result<Self> {
        if !(2..=MAX_BEAT_TIMES).contains(&times.len()) || times[0] != 0.0 {
            return Err(invalid(
                "provide 2–100001 quarter-beat times beginning at zero seconds",
            ));
        }
        if times.iter().any(|t| !t.is_finite()) {
            return Err(invalid("beat times must be finite"));
        }
        let mut changes = Vec::with_capacity(times.len() - 1);
        for pair in times.windows(2) {
            let delta = pair[1] - pair[0];
            let micros = delta * 1_000_000.0;
            if !(1.0..=16_777_215.0).contains(&micros) {
                return Err(invalid(
                    "quarter beats must be strictly increasing with MIDI-representable intervals (1–16777215 microseconds)",
                ));
            }
            changes.push(TempoChange {
                time: pair[0],
                bpm: 60.0 / delta,
            });
        }
        Ok(Self { changes })
    }
}

#[derive(Debug, Clone, Copy)]
struct Segment {
    seconds: f64,
    beats: f64,
    bpm: f64,
}

/// A validated tempo map with logarithmic-time seconds ↔ quarter-beat lookup.
///
/// The map is piecewise constant. Construction checks it begins at zero, has
/// ordered finite change times and strictly positive finite BPM, and remains
/// invertible in `f64`. MIDI encoding imposes additional tempo and tick limits.
#[derive(Debug, Clone)]
pub struct TempoTimeline {
    segments: Vec<Segment>,
}

impl TempoTimeline {
    /// Validate and index a tempo map; the input can be dropped afterward.
    pub fn new(map: &TempoMap) -> Result<Self> {
        if map.changes.is_empty() || map.changes[0].time != 0.0 {
            return Err(invalid("tempo map must begin at zero seconds"));
        }
        let mut segments: Vec<Segment> = Vec::with_capacity(map.changes.len());
        for change in &map.changes {
            if !change.time.is_finite() || !change.bpm.is_finite() || change.bpm <= 0.0 {
                return Err(invalid("tempo times and positive BPM must be finite"));
            }
            let beats_per_second = change.bpm / 60.0;
            let seconds_per_beat = 60.0 / change.bpm;
            if !beats_per_second.is_finite()
                || beats_per_second <= 0.0
                || !seconds_per_beat.is_finite()
                || seconds_per_beat <= 0.0
            {
                return Err(invalid("tempo rate exceeds f64 precision or range"));
            }
            let beats = if let Some(previous) = segments.last() {
                if change.time <= previous.seconds {
                    return Err(invalid("tempo changes must be strictly ordered"));
                }
                let value =
                    previous.beats + (change.time - previous.seconds) * (previous.bpm / 60.0);
                if !value.is_finite() || value <= previous.beats {
                    return Err(invalid("tempo timeline exceeds f64 precision or range"));
                }
                value
            } else {
                0.0
            };
            segments.push(Segment {
                seconds: change.time,
                beats,
                bpm: change.bpm,
            });
        }
        Ok(Self { segments })
    }

    /// Convert nonnegative seconds to quarter-note beats from time zero.
    pub fn seconds_to_beats(&self, seconds: f64) -> Result<f64> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(invalid("timeline seconds must be finite and nonnegative"));
        }
        let index = self
            .segments
            .partition_point(|segment| segment.seconds <= seconds)
            - 1;
        let segment = self.segments[index];
        let result = segment.beats + (seconds - segment.seconds) * (segment.bpm / 60.0);
        if !result.is_finite() {
            return Err(invalid("beat position exceeds f64 range"));
        }
        Ok(result)
    }

    /// Convert nonnegative quarter-note beats from time zero into seconds.
    pub fn beats_to_seconds(&self, beats: f64) -> Result<f64> {
        if !beats.is_finite() || beats < 0.0 {
            return Err(invalid("timeline beats must be finite and nonnegative"));
        }
        let index = self
            .segments
            .partition_point(|segment| segment.beats <= beats)
            - 1;
        let segment = self.segments[index];
        let result = segment.seconds + (beats - segment.beats) * (60.0 / segment.bpm);
        if !result.is_finite() {
            return Err(invalid("time position exceeds f64 range"));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supplied_beats_define_exact_variable_tempo_grid() {
        let times = [0.0, 0.5, 1.1, 1.8, 2.3];
        let map = TempoMap::from_quarter_beat_times(&times).unwrap();
        assert_eq!(map.changes.len(), 4);
        let timeline = TempoTimeline::new(&map).unwrap();
        for (beat, time) in times.into_iter().enumerate() {
            assert!((timeline.seconds_to_beats(time).unwrap() - beat as f64).abs() < 1e-12);
            assert!((timeline.beats_to_seconds(beat as f64).unwrap() - time).abs() < 1e-12);
        }
        for i in 0..500 {
            let time = i as f64 / 100.0;
            let beats = timeline.seconds_to_beats(time).unwrap();
            assert!((timeline.beats_to_seconds(beats).unwrap() - time).abs() < 1e-12);
        }
        assert!((timeline.seconds_to_beats(3.3).unwrap() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn reject_invalid_or_unrepresentable_beats() {
        for times in [
            vec![],
            vec![0.0],
            vec![1.0, 2.0],
            vec![0.0, 0.0],
            vec![0.0, -1.0],
            vec![0.0, f64::NAN],
            vec![0.0, f64::INFINITY],
            vec![0.0, 0.000_000_1],
            vec![0.0, 17.0],
        ] {
            assert!(
                TempoMap::from_quarter_beat_times(&times).is_err(),
                "{times:?}"
            );
        }
        assert!(TempoMap::from_quarter_beat_times(&vec![0.0; MAX_BEAT_TIMES + 1]).is_err());
    }

    #[test]
    fn timeline_rejects_bad_maps_and_queries() {
        let map = |changes| TempoMap { changes };
        let change = |time, bpm| TempoChange { time, bpm };
        for changes in [
            vec![],
            vec![change(1.0, 120.0)],
            vec![change(0.0, 0.0)],
            vec![change(0.0, f64::NAN)],
            vec![change(0.0, f64::from_bits(1))],
            vec![change(0.0, 120.0), change(0.0, 90.0)],
            vec![change(0.0, 120.0), change(f64::INFINITY, 90.0)],
        ] {
            assert!(TempoTimeline::new(&map(changes)).is_err());
        }
        let timeline = TempoTimeline::new(&TempoMap::constant(120.0)).unwrap();
        for value in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(timeline.seconds_to_beats(value).is_err());
            assert!(timeline.beats_to_seconds(value).is_err());
        }
    }
}
