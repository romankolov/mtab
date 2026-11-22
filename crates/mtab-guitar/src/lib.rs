//! `mtab-guitar` — guitar domain logic (PROMPT.md §5, §6.5–§6.9).
//!
//! - [`tab`]: the `TabDocument` IR every pipeline stage reads/writes
//! - [`tuning`]: tuning library + detection input types (detection lands M5)
//!
//! String/fret assignment (the "tab brain", §6.6) and technique detection
//! (§6.8) land in later milestones and extend these types.

pub mod rhythm;
pub mod tab;
pub mod tuning;

pub use rhythm::Grid;
pub use tab::{
    BendKind, ChordSymbol, GuitarTrack, PitchBend, SlideKind, TabDocument, TabNote, Technique,
    TempoChange, TempoMap, TrackMeta,
};
pub use tuning::{Capo, Tuning};
