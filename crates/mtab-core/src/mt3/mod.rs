//! MT3 tokenizer vocabulary and the chunk-decode state machine.
//!
//! Exact ports of upstream (MIT, Kyutai & Mirelo; adapted there from YourMT3+):
//! - `tokenizer/notes.py::build_event_vocab` → [`build_event_vocab`]
//! - `tokenizer/mt3.py::MT3Tokenizer` → [`MT3Tokenizer`] (tie prologue,
//!   forbidden-token masking)
//! - `events.py::OpenNoteTracker` + `decode_model_tokens` → [`tracker`]
//!
//! Vocabulary layout (1393 tokens, indices in order):
//! PAD(0) EOS(1) UNK(2) | shift 0..1000 (1001) | pitch 0..127 | velocity 0..1
//! | tie | program 0..129 | drum 0..127.
//!
//! Model cards: small = 1393 (== vocab); medium/large = 1395 — upstream masks
//! `logits[1393:] = -inf` (reserved/OOV), so decode ignores those rows.

pub mod instruments;
pub mod tracker;

use std::collections::HashMap;

pub use tracker::{ChunkBoundary, DecodeAction, OpenNoteTracker};

/// Upstream `max_shift_steps` used for the released models.
pub const MAX_SHIFT_STEPS: usize = 1001;
/// Upstream `frame_rate` (tokens/shift ticks per second).
pub const FRAME_RATE: u32 = 100;
/// Upstream `MINIMUM_NOTE_DURATION_SEC`.
pub const MINIMUM_NOTE_DURATION_SEC: f64 = 0.01;
/// Upstream `DRUM_PROGRAM`.
pub const DRUM_PROGRAM: u16 = 128;
/// Exact vocabulary size of the released tokenizer.
pub const VOCAB_SIZE: usize = 3 + MAX_SHIFT_STEPS + 128 + 2 + 1 + 130 + 128; // 1393

/// Token event kinds (upstream `Event.type`, with specials split out).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    Pad,
    Eos,
    Unk,
    Shift,
    Pitch,
    Velocity,
    Tie,
    Program,
    Drum,
}

/// One vocabulary entry (upstream `Event`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    pub kind: EventKind,
    pub value: u16,
}

/// Build the token-index → [`Event`] decode table (upstream
/// `notes.py::build_event_vocab`, `max_shift_steps = 1001` for released models).
pub fn build_event_vocab(max_shift_steps: usize) -> Vec<Event> {
    let mut vocab = Vec::with_capacity(3 + max_shift_steps + 128 + 2 + 1 + 130 + 128);
    for kind in [EventKind::Pad, EventKind::Eos, EventKind::Unk] {
        vocab.push(Event { kind, value: 0 });
    }
    for v in 0..max_shift_steps {
        vocab.push(Event {
            kind: EventKind::Shift,
            value: v as u16,
        });
    }
    for v in 0..128u16 {
        vocab.push(Event {
            kind: EventKind::Pitch,
            value: v,
        });
    }
    for v in 0..2u16 {
        vocab.push(Event {
            kind: EventKind::Velocity,
            value: v,
        });
    }
    vocab.push(Event {
        kind: EventKind::Tie,
        value: 0,
    });
    for v in 0..130u16 {
        vocab.push(Event {
            kind: EventKind::Program,
            value: v,
        });
    }
    for v in 0..128u16 {
        vocab.push(Event {
            kind: EventKind::Drum,
            value: v,
        });
    }
    vocab
}

/// The released tokenizer (upstream `MT3Tokenizer(instrument_vocabulary=
/// "MT3_FULL_PLUS", max_shift_steps=1001, frame_rate=100)`).
pub struct MT3Tokenizer {
    pub vocab: Vec<Event>,
    token_index: HashMap<(EventKind, u16), usize>,
    pub frame_rate: u32,
}

impl MT3Tokenizer {
    /// The released-model tokenizer configuration.
    pub fn released() -> Self {
        Self::new(MAX_SHIFT_STEPS, FRAME_RATE)
    }

    pub fn new(max_shift_steps: usize, frame_rate: u32) -> Self {
        let vocab = build_event_vocab(max_shift_steps);
        let mut token_index = HashMap::with_capacity(vocab.len());
        for (i, e) in vocab.iter().enumerate() {
            token_index.insert((e.kind, e.value), i);
        }
        Self {
            vocab,
            token_index,
            frame_rate,
        }
    }

    /// Token id for an event (panics on events outside the vocabulary).
    pub fn id(&self, kind: EventKind, value: u16) -> usize {
        self.token_index[&(kind, value)]
    }

    /// Upstream `eos_id = SPECIAL_TOKENS.index("EOS")` = 1.
    pub fn eos_id(&self) -> usize {
        1
    }

    /// Initial/BOS token id (= card, an extra embedding row — upstream
    /// `LMModel.initial_token_id`).
    pub fn initial_token_id(&self, card: usize) -> usize {
        card
    }

    /// Encode a tie prologue declaring `open_note_keys` as sustained
    /// (upstream `tie_section_token_ids`): pairs sorted by (program, pitch),
    /// one program token per run of pitches, terminated by the `tie` token.
    pub fn tie_section_token_ids(&self, open_note_keys: &[(u16, u16)]) -> Vec<usize> {
        let mut keys = open_note_keys.to_vec();
        keys.sort_unstable();
        let mut tokens = Vec::new();
        let mut program_state: Option<u16> = None;
        for (program, pitch) in keys {
            if Some(program) != program_state {
                tokens.push(self.id(EventKind::Program, program));
                program_state = Some(program);
            }
            tokens.push(self.id(EventKind::Pitch, pitch));
        }
        tokens.push(self.id(EventKind::Tie, 0));
        tokens
    }

    /// Token ids that must never be sampled when only `instruments` may appear
    /// (upstream `forbidden_token_ids`): `program` tokens are forbidden unless
    /// they are the representative (first) program of an allowed group;
    /// `drum` tokens are forbidden unless "drums" is allowed. Timing, pitch,
    /// velocity, tie and special tokens are never forbidden.
    pub fn forbidden_token_ids(&self, instruments: &[String]) -> Vec<usize> {
        use self::instruments as inst;
        let allow_drums = instruments.iter().any(|n| n == "drums");
        let mut allowed_programs = Vec::new();
        for name in instruments {
            if name == "drums" {
                continue;
            }
            if let Some(gid) = inst::group_id(name) {
                if let Some(rep) = inst::representative_program(gid) {
                    allowed_programs.push(rep);
                }
            }
        }
        let mut forbidden = Vec::new();
        for (token_id, event) in self.vocab.iter().enumerate() {
            match event.kind {
                EventKind::Program if !allowed_programs.contains(&event.value) => {
                    forbidden.push(token_id)
                }
                EventKind::Drum if !allow_drums => forbidden.push(token_id),
                _ => {}
            }
        }
        forbidden
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_layout_matches_upstream() {
        let t = MT3Tokenizer::released();
        assert_eq!(t.vocab.len(), VOCAB_SIZE);
        assert_eq!(t.vocab.len(), 1393);
        let e = |i: usize| t.vocab[i];
        assert_eq!(
            e(0),
            Event {
                kind: EventKind::Pad,
                value: 0
            }
        );
        assert_eq!(
            e(1),
            Event {
                kind: EventKind::Eos,
                value: 0
            }
        );
        assert_eq!(
            e(2),
            Event {
                kind: EventKind::Unk,
                value: 0
            }
        );
        assert_eq!(
            e(3),
            Event {
                kind: EventKind::Shift,
                value: 0
            }
        );
        assert_eq!(
            e(1003),
            Event {
                kind: EventKind::Shift,
                value: 1000
            }
        );
        assert_eq!(
            e(1004),
            Event {
                kind: EventKind::Pitch,
                value: 0
            }
        );
        assert_eq!(
            e(1131),
            Event {
                kind: EventKind::Pitch,
                value: 127
            }
        );
        assert_eq!(
            e(1132),
            Event {
                kind: EventKind::Velocity,
                value: 0
            }
        );
        assert_eq!(
            e(1133),
            Event {
                kind: EventKind::Velocity,
                value: 1
            }
        );
        assert_eq!(
            e(1134),
            Event {
                kind: EventKind::Tie,
                value: 0
            }
        );
        assert_eq!(
            e(1135),
            Event {
                kind: EventKind::Program,
                value: 0
            }
        );
        assert_eq!(
            e(1264),
            Event {
                kind: EventKind::Program,
                value: 129
            }
        );
        assert_eq!(
            e(1265),
            Event {
                kind: EventKind::Drum,
                value: 0
            }
        );
        assert_eq!(
            e(1392),
            Event {
                kind: EventKind::Drum,
                value: 127
            }
        );
    }

    #[test]
    fn tie_prologue_encodes_program_runs() {
        let t = MT3Tokenizer::released();
        let ids = t.tie_section_token_ids(&[(26, 64), (26, 60), (24, 40)]);
        assert_eq!(
            ids,
            vec![
                t.id(EventKind::Program, 24),
                t.id(EventKind::Pitch, 40),
                t.id(EventKind::Program, 26),
                t.id(EventKind::Pitch, 60),
                t.id(EventKind::Pitch, 64),
                t.id(EventKind::Tie, 0),
            ]
        );
    }

    #[test]
    fn forbidden_tokens_for_guitar_family() {
        let t = MT3Tokenizer::released();
        let names: Vec<String> = [
            "acoustic_guitar",
            "clean_electric_guitar",
            "distorted_electric_guitar",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let forbidden = t.forbidden_token_ids(&names);
        // All program tokens except representatives {24, 26, 29}, plus all
        // drums: (130 - 3) + 128 = 255.
        assert_eq!(forbidden.len(), 255);
        assert!(!forbidden.contains(&t.id(EventKind::Program, 24)));
        assert!(!forbidden.contains(&t.id(EventKind::Program, 26)));
        assert!(!forbidden.contains(&t.id(EventKind::Program, 29)));
        assert!(forbidden.contains(&t.id(EventKind::Program, 25)));
        assert!(forbidden.contains(&t.id(EventKind::Drum, 60)));
        // Timing/pitch/velocity/tie never forbidden.
        for id in [
            t.id(EventKind::Shift, 0),
            t.id(EventKind::Pitch, 60),
            t.id(EventKind::Velocity, 1),
            t.id(EventKind::Tie, 0),
        ] {
            assert!(!forbidden.contains(&id));
        }
        // With drums allowed, drum tokens are not forbidden.
        let mut with_drums = names.clone();
        with_drums.push("drums".to_owned());
        let f2 = t.forbidden_token_ids(&with_drums);
        assert_eq!(f2.len(), 127); // 130-3 program, drums pass
    }
}
