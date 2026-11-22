//! Raw transcription events emitted by the MuScriptor engine (pre-fretting).
//!
//! These mirror the upstream event stream (`muscriptor.events`:
//! NoteStartEvent/NoteEndEvent/ProgressEvent, MIT-licensed, Kyutai & Mirelo)
//! in a Rust-friendly flattened form. Guitar-specific state (string, fret,
//! techniques) lives in `mtab-guitar::tab` and is derived later in the
//! pipeline (PROMPT.md §4).

use serde::{Deserialize, Serialize};

/// A single transcribed note, before string/fret assignment.
///
/// One note = one onset/offset pair. `instrument` is the upstream instrument
/// id (program number); guitar-family filtering happens at decode time via
/// token masking (PROMPT.md §4, §6.3).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NoteEvent {
    /// Onset time in seconds from the start of the audio.
    pub start: f64,
    /// Offset time in seconds from the start of the audio.
    pub end: f64,
    /// MIDI pitch (0-127).
    pub midi: u8,
    /// MIDI velocity (1-127; 0 is never emitted — upstream uses velocity 0 as
    /// a note-off marker in the token stream).
    pub velocity: u8,
    /// Upstream instrument/program id.
    pub instrument: u16,
}

impl NoteEvent {
    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}

/// Coarse transcription progress (PROMPT.md §6.4): `completed` of `total`
/// fixed-size chunks have been transcribed. Mirrors upstream
/// `events.ProgressEvent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgressEvent {
    pub completed: usize,
    pub total: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_duration_is_non_negative() {
        let n = NoteEvent {
            start: 1.5,
            end: 2.25,
            midi: 64,
            velocity: 100,
            instrument: 27,
        };
        assert!((n.duration() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn note_event_serde_round_trip() {
        let n = NoteEvent {
            start: 0.0,
            end: 0.5,
            midi: 40,
            velocity: 90,
            instrument: 25,
        };
        let json = serde_json::to_string(&n).unwrap();
        let back: NoteEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(n, back);
    }
}
