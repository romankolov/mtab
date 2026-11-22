//! `mtab-guitar` — guitar domain logic (PROMPT.md §5, §6.5–§6.9).
//!
//! - [`tab`]: the `TabDocument` IR every pipeline stage reads/writes
//! - [`tuning`]: tuning library + detection input types (detection lands M5)
//! - [`fingering`]: preliminary bounded global string/fret search
//!
//! Tuning detection, chord-shape constraints, evaluation and technique
//! detection remain future work.

pub mod fingering;
pub mod rhythm;
pub mod tab;
pub mod tuning;

pub use fingering::{assign_notes, FingeringOptions};
pub use rhythm::Grid;
pub use tab::{
    BendKind, ChordSymbol, GuitarTrack, PitchBend, SlideKind, TabDocument, TabNote, Technique,
    TempoChange, TempoMap, TrackMeta,
};
pub use tuning::{Capo, Tuning};
