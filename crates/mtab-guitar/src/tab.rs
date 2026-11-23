//! The `TabDocument` IR (PROMPT.md §5).
//!
//! Every pipeline stage reads and writes this structure; `--emit-ir` dumps it
//! as JSON for debugging and golden tests.

use serde::{Deserialize, Serialize};

use crate::tuning::{Capo, Tuning};

/// Root of the in-memory tablature representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TabDocument {
    pub meta: TrackMeta,
    pub tracks: Vec<GuitarTrack>,
    pub chord_symbols: Vec<ChordSymbol>,
}

/// Document-level metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackMeta {
    pub title: Option<String>,
    pub tempo_map: TempoMap,
    /// `(bar_index, numerator, denominator)` changes.
    pub time_signatures: Vec<(u32, u32, u32)>,
    /// Default fretboard size for playability constraints (§6.6).
    pub frets: u8,
    /// Hash of the source audio (provenance for goldens/regeneration).
    pub source_hash: Option<String>,
}

/// Piecewise-constant tempo map: `bpm_at(t)` uses the most recent change.
/// The representation permits one change per beat, but no beat tracking or
/// automatic unsteady-tempo inference is implemented by the CLI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TempoMap {
    pub changes: Vec<TempoChange>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TempoChange {
    /// Time in seconds.
    pub time: f64,
    /// Beats per minute.
    pub bpm: f64,
}

impl TempoMap {
    /// Constant tempo from `t = 0`.
    pub fn constant(bpm: f64) -> Self {
        Self {
            changes: vec![TempoChange { time: 0.0, bpm }],
        }
    }

    /// BPM at a given time (last change at or before `t`; first change if
    /// before the map starts).
    pub fn bpm_at(&self, t: f64) -> f64 {
        self.changes
            .iter()
            .rev()
            .find(|c| c.time <= t)
            .or_else(|| self.changes.first())
            .map(|c| c.bpm)
            .unwrap_or(120.0)
    }
}

/// One guitar track: a fretted performance on one instrument.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuitarTrack {
    pub tuning: Tuning,
    pub capo: Capo,
    pub notes: Vec<TabNote>,
    /// Upstream instrument name (e.g. "electric_guitar_distorted").
    pub instrument: Option<String>,
}

/// A fretted note (PROMPT.md §5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TabNote {
    /// Onset time in seconds.
    pub start: f64,
    /// Offset time in seconds.
    pub end: f64,
    /// MIDI pitch (with capo applied).
    pub midi: u8,
    /// String index, low string = 0.
    pub string: u8,
    /// Fret number (0 = open; capo not included — frets are relative to capo).
    pub fret: u8,
    /// MIDI velocity 30–127 with real dynamics (§6.8; NOT fixed).
    pub velocity: u8,
    /// Pitch bend / vibrato contour (§6.8).
    pub bend: Option<PitchBend>,
    pub technique: Vec<Technique>,
    /// True if this note is tied from the previous (same string, legato).
    pub tie_from_prev: bool,
}

/// Pitch-bend contour for a note (§6.8): cents deviation over time.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PitchBend {
    /// Bend start time (seconds), usually the note onset.
    pub start: f64,
    /// Time of maximum deviation (seconds).
    pub peak: f64,
    /// Maximum deviation in cents (positive = up).
    pub depth_cents: f32,
    pub kind: BendKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BendKind {
    /// Unison bend up.
    Bend,
    /// Release down from a pre-bent pitch.
    Release,
    /// Bend up and release within the note.
    BendRelease,
}

/// Guitar articulations (§6.8). Export support varies by technique and format.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Technique {
    Slide { to_fret: u8, kind: SlideKind },
    HammerOn,
    PullOff,
    Vibrato { rate_hz: f32, depth_cents: f32 },
    PalmMute,
    DeadNote,
    Harmonic,
    LetRing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlideKind {
    /// Legato slide (picked only at the start).
    Legato,
    /// Shift slide (re-picked at the target).
    Shift,
    /// Picked slide into the note from below/above.
    Picked,
}

/// Chord symbol above the staff (§6.9).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChordSymbol {
    pub start: f64,
    pub end: f64,
    /// Display label, e.g. "Em7", "D/F#".
    pub label: String,
}

impl TabDocument {
    /// Serialize to pretty JSON (`--emit-ir`).
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_doc() -> TabDocument {
        TabDocument {
            meta: TrackMeta {
                title: Some("Test Riff".into()),
                tempo_map: TempoMap::constant(120.0),
                time_signatures: vec![(0, 4, 4)],
                frets: 22,
                source_hash: Some("deadbeef".into()),
            },
            tracks: vec![GuitarTrack {
                tuning: Tuning::standard(),
                capo: Capo::new(2).unwrap(),
                instrument: Some("electric_guitar_distorted".into()),
                notes: vec![
                    TabNote {
                        start: 0.0,
                        end: 0.5,
                        midi: 64,
                        string: 2,
                        fret: 2,
                        velocity: 100,
                        bend: Some(PitchBend {
                            start: 0.2,
                            peak: 0.35,
                            depth_cents: 150.0,
                            kind: BendKind::BendRelease,
                        }),
                        technique: vec![
                            Technique::Slide {
                                to_fret: 5,
                                kind: SlideKind::Legato,
                            },
                            Technique::Vibrato {
                                rate_hz: 5.5,
                                depth_cents: 30.0,
                            },
                        ],
                        tie_from_prev: false,
                    },
                    TabNote {
                        start: 0.5,
                        end: 0.6,
                        midi: 40,
                        string: 0,
                        fret: 0,
                        velocity: 60,
                        bend: None,
                        technique: vec![Technique::PalmMute, Technique::DeadNote],
                        tie_from_prev: true,
                    },
                ],
            }],
            chord_symbols: vec![ChordSymbol {
                start: 0.0,
                end: 1.0,
                label: "Em".into(),
            }],
        }
    }

    #[test]
    fn json_round_trip() {
        let doc = sample_doc();
        let json = doc.to_json().unwrap();
        let back = TabDocument::from_json(&json).unwrap();
        assert_eq!(doc, back);
    }

    #[test]
    fn tempo_map_interpolates() {
        let tm = TempoMap {
            changes: vec![
                TempoChange {
                    time: 0.0,
                    bpm: 100.0,
                },
                TempoChange {
                    time: 2.0,
                    bpm: 140.0,
                },
            ],
        };
        assert_eq!(tm.bpm_at(0.5), 100.0);
        assert_eq!(tm.bpm_at(2.5), 140.0);
        assert_eq!(tm.bpm_at(2.0), 140.0);
    }
}
