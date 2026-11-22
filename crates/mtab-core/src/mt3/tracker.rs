//! Chunk-decode state machine — exact port of upstream `events.py`
//! (`OpenNoteTracker` + `decode_model_tokens`), MIT, Kyutai & Mirelo.
//!
//! Consumes the interleaved stream of [`ChunkBoundary`] markers and token ids;
//! every chunk begins with a *tie prologue* ((program, pitch) pairs sustained
//! from the previous chunk, terminated by a `tie` token). Open notes absent
//! from the tie set close at the chunk boundary. The same state machine feeds
//! prelude forcing (its [`OpenNoteTracker::open_keys`] is what the next
//! chunk's prologue must declare).

use std::collections::BTreeSet;

use super::{Event, EventKind, MINIMUM_NOTE_DURATION_SEC};
use crate::event::NoteEvent;

/// Marks the start of a new model-output chunk (upstream `ChunkBoundary`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkBoundary {
    /// Chunk start time in seconds.
    pub seek_time: f64,
    /// Next chunk start (`None` for the last chunk); events at or past it are
    /// dropped.
    pub next_seek_time: Option<f64>,
}

/// Note actions implied by the token stream (upstream `_NoteAction`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecodeAction {
    StartNote { program: u16, pitch: u16, time: f64 },
    EndNote { program: u16, pitch: u16, time: f64 },
    DrumHit { pitch: u16, time: f64 },
}

/// Feed item: a chunk boundary or a raw token id.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StreamItem {
    Boundary(ChunkBoundary),
    Token(usize),
}

type Key = (u16, u16);

/// The decode state machine (upstream `OpenNoteTracker`).
pub struct OpenNoteTracker {
    vocab: Vec<Event>,
    frame_rate: f64,
    /// Open notes as (key, onset), insertion-ordered: end-of-stream closes
    /// replay in onset order (upstream dict ordering).
    open: Vec<(Key, f64)>,
    seek_time: f64,
    next_seek_time: Option<f64>,
    start_tick: i64,
    tick_state: i64,
    program: Option<u16>,
    velocity: Option<u16>,
    in_prologue: bool,
    skip_rest: bool,
    tie_set: BTreeSet<Key>,
    chunk_started: bool,
}

impl OpenNoteTracker {
    pub fn new(vocab: Vec<Event>, frame_rate: u32) -> Self {
        Self {
            vocab,
            frame_rate: frame_rate as f64,
            open: Vec::new(),
            seek_time: 0.0,
            next_seek_time: None,
            start_tick: 0,
            tick_state: 0,
            program: None,
            velocity: None,
            in_prologue: true,
            skip_rest: false,
            tie_set: BTreeSet::new(),
            chunk_started: false,
        }
    }

    /// Sorted `(program, pitch)` pairs currently held open — the set the next
    /// chunk's tie prologue must declare (prelude forcing reads this).
    pub fn open_keys(&self) -> Vec<Key> {
        let mut keys: Vec<Key> = self.open.iter().map(|(k, _)| *k).collect();
        keys.sort_unstable();
        keys
    }

    pub fn feed(&mut self, item: StreamItem) -> Vec<DecodeAction> {
        match item {
            StreamItem::Boundary(b) => {
                let mut actions = Vec::new();
                // Previous chunk never closed its tie prologue (malformed):
                // treat its tie set as empty so every still-open note ends at
                // that chunk's boundary.
                if self.chunk_started && self.in_prologue {
                    actions = self.end_all(self.seek_time);
                }
                self.seek_time = b.seek_time;
                self.next_seek_time = b.next_seek_time;
                // Python `round()` is half-to-even.
                self.start_tick = py_round(b.seek_time * self.frame_rate);
                self.tick_state = self.start_tick;
                self.program = None;
                self.velocity = None;
                self.in_prologue = true;
                self.skip_rest = false;
                self.tie_set.clear();
                self.chunk_started = true;
                actions
            }
            StreamItem::Token(id) => {
                let event = self.vocab[id];

                if self.in_prologue {
                    return match event.kind {
                        EventKind::Tie => {
                            // End of tie section: close prior notes not sustained here.
                            self.in_prologue = false;
                            self.velocity = None;
                            let ended: Vec<Key> = self
                                .open
                                .iter()
                                .map(|(k, _)| *k)
                                .filter(|k| !self.tie_set.contains(k))
                                .collect();
                            let mut actions = Vec::with_capacity(ended.len());
                            for key in ended {
                                self.remove_open(key);
                                actions.push(DecodeAction::EndNote {
                                    program: key.0,
                                    pitch: key.1,
                                    time: self.seek_time,
                                });
                            }
                            actions
                        }
                        EventKind::Shift => {
                            // No tie token: malformed. Close all open notes at
                            // the boundary and drop the rest of the chunk.
                            self.in_prologue = false;
                            self.skip_rest = true;
                            self.end_all(self.seek_time)
                        }
                        EventKind::Program => {
                            self.program = Some(event.value);
                            Vec::new()
                        }
                        EventKind::Pitch => {
                            if let Some(program) = self.program {
                                self.tie_set.insert((program, event.value));
                            }
                            Vec::new()
                        }
                        _ => Vec::new(),
                    };
                }

                if self.skip_rest {
                    return Vec::new();
                }

                match event.kind {
                    EventKind::Shift => {
                        if event.value > 0 {
                            self.tick_state = self.start_tick + event.value as i64;
                        }
                        Vec::new()
                    }
                    EventKind::Program => {
                        self.program = Some(event.value);
                        Vec::new()
                    }
                    EventKind::Velocity => {
                        self.velocity = Some(event.value);
                        Vec::new()
                    }
                    EventKind::Drum => {
                        let time = self.tick_state as f64 / self.frame_rate;
                        if self.next_seek_time.map_or(true, |next| time < next) {
                            vec![DecodeAction::DrumHit {
                                pitch: event.value,
                                time,
                            }]
                        } else {
                            Vec::new()
                        }
                    }
                    EventKind::Pitch => {
                        let (Some(program), Some(velocity)) = (self.program, self.velocity) else {
                            return Vec::new();
                        };
                        let time = self.tick_state as f64 / self.frame_rate;
                        if self.next_seek_time.is_some_and(|next| time >= next) {
                            return Vec::new();
                        }
                        let key = (program, event.value);
                        let mut actions = Vec::new();
                        if self.remove_open(key).is_some() {
                            actions.push(DecodeAction::EndNote {
                                program,
                                pitch: event.value,
                                time,
                            });
                        }
                        if velocity > 0 {
                            self.open.push((key, time));
                            actions.push(DecodeAction::StartNote {
                                program,
                                pitch: event.value,
                                time,
                            });
                        }
                        actions
                    }
                    _ => Vec::new(),
                }
            }
        }
    }

    /// End of stream: close anything still open. A well-formed final chunk
    /// uses the minimum-duration fallback; one that ended mid-prologue closes
    /// at its boundary.
    pub fn finish(&mut self) -> Vec<DecodeAction> {
        if self.chunk_started && self.in_prologue {
            return self.end_all(self.seek_time);
        }
        let actions: Vec<DecodeAction> = self
            .open
            .iter()
            .map(|(key, onset)| DecodeAction::EndNote {
                program: key.0,
                pitch: key.1,
                time: onset + MINIMUM_NOTE_DURATION_SEC,
            })
            .collect();
        self.open.clear();
        actions
    }

    fn remove_open(&mut self, key: Key) -> Option<f64> {
        let pos = self.open.iter().position(|(k, _)| *k == key)?;
        Some(self.open.remove(pos).1)
    }

    fn end_all(&mut self, time: f64) -> Vec<DecodeAction> {
        let actions: Vec<DecodeAction> = self
            .open
            .iter()
            .map(|(key, _)| DecodeAction::EndNote {
                program: key.0,
                pitch: key.1,
                time,
            })
            .collect();
        self.open.clear();
        actions
    }
}

/// Python `round()` (half-to-even) on f64.
fn py_round(x: f64) -> i64 {
    let floor = x.floor();
    let diff = x - floor;
    if (diff - 0.5).abs() < f64::EPSILON {
        // Exactly .5: round to even.
        let f = floor as i64;
        if f % 2 == 0 {
            f
        } else {
            f + 1
        }
    } else {
        x.round() as i64
    }
}

/// Turn a token/boundary stream into note events (upstream
/// `decode_model_tokens`): every note gets exactly one start and one matching
/// end; drum hits become instantaneous notes of [`MINIMUM_NOTE_DURATION_SEC`].
///
/// Upstream emits no per-note dynamics (velocity tokens are on/off markers),
/// so decoded notes carry the upstream MIDI convention velocity = 100.
/// Pitch bends and true dynamics are reconstructed later (M7).
pub fn decode_stream(items: &[StreamItem], vocab: Vec<Event>, frame_rate: u32) -> Vec<NoteEvent> {
    let mut tracker = OpenNoteTracker::new(vocab, frame_rate);
    let mut notes = Vec::new();
    // (program, pitch) -> index into `notes` of the open start.
    let mut open: Vec<(Key, usize)> = Vec::new();

    let apply =
        |actions: Vec<DecodeAction>, open: &mut Vec<(Key, usize)>, notes: &mut Vec<NoteEvent>| {
            for action in actions {
                match action {
                    DecodeAction::StartNote {
                        program,
                        pitch,
                        time,
                    } => {
                        open.push(((program, pitch), notes.len()));
                        notes.push(NoteEvent {
                            start: time,
                            end: f64::NAN, // filled at matching EndNote
                            midi: pitch as u8,
                            velocity: 100,
                            instrument: program,
                        });
                    }
                    DecodeAction::EndNote {
                        program,
                        pitch,
                        time,
                    } => {
                        if let Some(pos) = open.iter().position(|(k, _)| *k == (program, pitch)) {
                            let (_, idx) = open.remove(pos);
                            notes[idx].end = time;
                        }
                    }
                    DecodeAction::DrumHit { pitch, time } => {
                        notes.push(NoteEvent {
                            start: time,
                            end: time + MINIMUM_NOTE_DURATION_SEC,
                            midi: pitch as u8,
                            velocity: 100,
                            instrument: super::DRUM_PROGRAM,
                        });
                    }
                }
            }
        };

    for item in items {
        apply(tracker.feed(*item), &mut open, &mut notes);
    }
    apply(tracker.finish(), &mut open, &mut notes);

    // Drop pathological unclosed notes (defensive; well-formed streams end all).
    notes.retain(|n| n.end.is_finite());
    notes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mt3::{build_event_vocab, EventKind, MT3Tokenizer, DRUM_PROGRAM};

    fn t() -> MT3Tokenizer {
        MT3Tokenizer::released()
    }

    fn items_seq(pairs: &[(&str, u16)], seek: f64, next: Option<f64>) -> Vec<StreamItem> {
        // Helper: boundary + event tokens given (kind-name, value).
        let tk = t();
        let mut v = vec![StreamItem::Boundary(ChunkBoundary {
            seek_time: seek,
            next_seek_time: next,
        })];
        for (kind, value) in pairs {
            let k = match *kind {
                "shift" => EventKind::Shift,
                "pitch" => EventKind::Pitch,
                "velocity" => EventKind::Velocity,
                "tie" => EventKind::Tie,
                "program" => EventKind::Program,
                "drum" => EventKind::Drum,
                _ => panic!("bad kind {kind}"),
            };
            v.push(StreamItem::Token(tk.id(k, *value)));
        }
        v
    }

    #[test]
    fn simple_note_lifecycle() {
        let tk = t();
        let stream = items_seq(
            &[
                ("tie", 0),
                ("program", 26),
                ("velocity", 1),
                ("shift", 100), // 1.0 s
                ("pitch", 64),
            ],
            0.0,
            None,
        );
        let mut all: Vec<StreamItem> = stream;
        all.push(StreamItem::Token(tk.eos_id()));
        let notes = decode_stream(&all, tk.vocab, 100);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].midi, 64);
        assert_eq!(notes[0].instrument, 26);
        assert_eq!(notes[0].start, 1.0);
        assert_eq!(notes[0].end, 1.0 + MINIMUM_NOTE_DURATION_SEC); // finish() fallback
    }

    #[test]
    fn retrigger_closes_then_reopens() {
        let tk = t();
        // NOTE: shift values are absolute-in-chunk (upstream assigns
        // tick_state = start_tick + value), so 50 -> 0.5 s, 100 -> 1.0 s.
        let all = items_seq(
            &[
                ("tie", 0),
                ("program", 26),
                ("velocity", 1),
                ("shift", 50),
                ("pitch", 64),
                ("shift", 100),
                ("pitch", 64), // retrigger: close at 1.0, reopen
            ],
            0.0,
            None,
        );
        let notes = decode_stream(&all, tk.vocab, 100);
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].start, 0.5);
        assert_eq!(notes[0].end, 1.0);
        assert_eq!(notes[1].start, 1.0);
    }

    #[test]
    fn tie_prologue_keeps_sustained_notes_open() {
        let tk = t();
        let mut all = items_seq(
            &[
                ("tie", 0),
                ("program", 26),
                ("velocity", 1),
                ("shift", 50),
                ("pitch", 64),
            ],
            0.0,
            Some(2.0),
        );
        // Chunk 2 at t=2.0 declares the note as tied, then closes it (velocity
        // 0 pitch = pure close) at 2.5 without retriggering.
        all.extend(items_seq(
            &[
                ("program", 26),
                ("pitch", 64),
                ("tie", 0),
                ("velocity", 0),
                ("shift", 50),
                ("pitch", 64),
            ],
            2.0,
            None,
        ));
        let notes = decode_stream(&all, tk.vocab, 100);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].start, 0.5);
        assert_eq!(notes[0].end, 2.5);
    }

    #[test]
    fn untied_notes_close_at_boundary() {
        let tk = t();
        let mut all = items_seq(
            &[
                ("tie", 0),
                ("program", 26),
                ("velocity", 1),
                ("shift", 50),
                ("pitch", 64),
            ],
            0.0,
            Some(2.0),
        );
        // Chunk 2 ties NOTHING -> note closes at the boundary (2.0).
        all.extend(items_seq(&[("tie", 0)], 2.0, None));
        let notes = decode_stream(&all, tk.vocab, 100);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].end, 2.0);
    }

    #[test]
    fn malformed_chunk_skips_rest() {
        let tk = t();
        let mut all = items_seq(
            &[
                ("tie", 0),
                ("program", 26),
                ("velocity", 1),
                ("shift", 50),
                ("pitch", 64),
            ],
            0.0,
            Some(2.0),
        );
        // Chunk 2: shift BEFORE tie -> malformed: close all at 2.0, skip rest.
        all.extend(items_seq(
            &[
                ("shift", 10),
                ("program", 26),
                ("velocity", 1),
                ("pitch", 60),
            ],
            2.0,
            None,
        ));
        let notes = decode_stream(&all, tk.vocab, 100);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].end, 2.0);
    }

    #[test]
    fn next_seek_time_filters_late_events() {
        let tk = t();
        let all = items_seq(
            &[
                ("tie", 0),
                ("program", 26),
                ("velocity", 1),
                ("shift", 300), // 3.0 s >= next_seek_time 2.0 -> dropped
                ("pitch", 64),
                ("shift", 50),
                ("pitch", 62), // 0.5 s -> kept
            ],
            0.0,
            Some(2.0),
        );
        let notes = decode_stream(&all, tk.vocab, 100);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].midi, 62);
        assert_eq!(notes[0].start, 0.5);
    }

    #[test]
    fn velocity_zero_closes_without_reopen() {
        let tk = t();
        let all = items_seq(
            &[
                ("tie", 0),
                ("program", 26),
                ("velocity", 1),
                ("shift", 50),
                ("pitch", 64),
                ("velocity", 0),
                ("shift", 100),
                ("pitch", 64),
            ],
            0.0,
            None,
        );
        let notes = decode_stream(&all, tk.vocab, 100);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].start, 0.5);
        assert_eq!(notes[0].end, 1.0);
    }

    #[test]
    fn drum_hits_are_instantaneous() {
        let tk = t();
        let all = items_seq(
            &[("tie", 0), ("drum", 36), ("shift", 50), ("drum", 38)],
            0.0,
            None,
        );
        let notes = decode_stream(&all, tk.vocab, 100);
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].instrument, DRUM_PROGRAM);
        assert_eq!(notes[0].start, 0.0);
        assert_eq!(notes[1].start, 0.5);
        assert_eq!(notes[1].end, 0.5 + MINIMUM_NOTE_DURATION_SEC);
    }

    #[test]
    fn vocab_default_matches_release() {
        assert_eq!(build_event_vocab(1001).len(), 1393);
    }
}
