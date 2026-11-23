//! `mtab-guitar` — guitar domain logic (PROMPT.md §5, §6.5–§6.9).
//!
//! - [`tab`]: the `TabDocument` IR every pipeline stage reads/writes
//! - [`tuning`]: tuning and capo library
//! - [`detection`]: conservative tuning-template scoring
//! - [`fingering`]: preliminary bounded global string/fret search
//!
//! Tuning detection is preliminary; chord-shape constraints, evaluation and
//! technique detection remain future work.

pub mod detection;
pub mod fingering;
pub mod rhythm;
pub mod tab;
pub mod tuning;

pub use detection::{
    detect_tuning, standard_candidates, TuningCandidate, TuningDetection, TuningScore,
};
pub use fingering::{
    assign_notes, assign_notes_partial, assign_notes_with_stretch, FingeringAssignment,
    FingeringOptions, PartialFingering,
};
pub use rhythm::Grid;
pub use tab::{
    BendKind, ChordSymbol, GuitarTrack, PitchBend, SlideKind, TabDocument, TabNote, Technique,
    TempoChange, TempoMap, TrackMeta,
};
pub use tuning::{Capo, Tuning};
