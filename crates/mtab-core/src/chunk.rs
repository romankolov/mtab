//! Ordered five-second chunk generation and MT3 prelude forcing.
//!
//! The caller supplies a generator so audio features and the model backend can
//! remain independent of this orchestration layer. Generator output must begin
//! with the supplied prompt and may include EOS. This module does not infer
//! guitar dynamics; the upstream MT3 decoder currently uses velocity 100.

use crate::event::{NoteEvent, ProgressEvent};
use crate::mt3::tracker::{decode_stream, StreamItem};
use crate::mt3::{ChunkBoundary, MT3Tokenizer, OpenNoteTracker};
use crate::{MtabError, Result};

/// Model input is 5 seconds at 16 kHz.
pub const CHUNK_SAMPLES: usize = 80_000;
const SAMPLE_RATE: f64 = 16_000.0;

/// Final notes and indexes of chunks that did not produce EOS.
pub struct ChunkResult {
    pub notes: Vec<NoteEvent>,
    pub missing_eos: Vec<usize>,
}

/// Process 16 kHz mono samples in order. The callback receives the chunk index,
/// a zero-padded five-second chunk, and the forced tie prompt (`None` for the
/// first chunk). It returns tokens including that prompt and, normally, EOS.
/// Progress is reported before generation and after each successful chunk.
///
/// When `strict_eos` is false, missing EOS is recorded rather than fatal.
pub fn transcribe_chunks<G, P>(
    samples: &[f32],
    strict_eos: bool,
    mut generate: G,
    mut progress: P,
) -> Result<ChunkResult>
where
    G: FnMut(usize, &[f32], Option<&[u32]>) -> Result<Vec<u32>>,
    P: FnMut(ProgressEvent),
{
    let tokenizer = MT3Tokenizer::released();
    let count = samples.len() / CHUNK_SAMPLES + usize::from(samples.len() % CHUNK_SAMPLES != 0);
    let mut tracker = OpenNoteTracker::new(tokenizer.vocab.clone(), tokenizer.frame_rate);
    let mut stream = Vec::new();
    let mut missing_eos = Vec::new();
    progress(ProgressEvent {
        completed: 0,
        total: count,
    });

    for (index, chunk) in samples.chunks(CHUNK_SAMPLES).enumerate() {
        let seek = index as f64 * CHUNK_SAMPLES as f64 / SAMPLE_RATE;
        let boundary = ChunkBoundary {
            seek_time: seek,
            next_seek_time: (index + 1 < count)
                .then_some(seek + CHUNK_SAMPLES as f64 / SAMPLE_RATE),
        };
        tracker.feed(StreamItem::Boundary(boundary));
        stream.push(StreamItem::Boundary(boundary));
        let prompt: Option<Vec<u32>> = (index > 0).then(|| {
            tokenizer
                .tie_section_token_ids(&tracker.open_keys())
                .into_iter()
                .map(|id| id as u32)
                .collect()
        });
        let mut padded = vec![0.0; CHUNK_SAMPLES];
        padded[..chunk.len()].copy_from_slice(chunk);
        let tokens = generate(index, &padded, prompt.as_deref())?;
        if let Some(prompt) = &prompt {
            if !tokens.starts_with(prompt) {
                return Err(MtabError::DecodeFailure {
                    chunk_index: index,
                    reason: "generated output does not start with the forced tie prologue".into(),
                });
            }
        }
        let eos = tokenizer.eos_id() as u32;
        let eos_pos = tokens.iter().position(|&id| id == eos);
        if eos_pos.is_none() {
            if strict_eos {
                return Err(MtabError::DecodeFailure { chunk_index: index, reason: "generation reached its limit without EOS; retry without strict EOS or increase the limit".into() });
            }
            missing_eos.push(index);
        }
        for &id in &tokens[..eos_pos.unwrap_or(tokens.len())] {
            if id as usize >= tokenizer.vocab.len() {
                return Err(MtabError::DecodeFailure {
                    chunk_index: index,
                    reason: format!("invalid MT3 token {id}"),
                });
            }
            let item = StreamItem::Token(id as usize);
            tracker.feed(item);
            stream.push(item);
        }
        progress(ProgressEvent {
            completed: index + 1,
            total: count,
        });
    }
    let notes = decode_stream(&stream, tokenizer.vocab, tokenizer.frame_rate);
    Ok(ChunkResult { notes, missing_eos })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mt3::EventKind;

    #[test]
    fn padding_progress_and_forced_ties_across_chunks() {
        let tokenizer = MT3Tokenizer::released();
        let id = |kind, value| tokenizer.id(kind, value) as u32;
        let data = vec![0.25; CHUNK_SAMPLES + 17];
        let mut progress = Vec::new();
        let out = transcribe_chunks(
            &data,
            true,
            |index, chunk, prompt| {
                assert_eq!(chunk.len(), CHUNK_SAMPLES);
                if index == 0 {
                    assert!(prompt.is_none());
                    assert!(chunk.iter().all(|&v| v == 0.25));
                    Ok(vec![
                        id(EventKind::Tie, 0),
                        id(EventKind::Program, 26),
                        id(EventKind::Velocity, 1),
                        id(EventKind::Shift, 50),
                        id(EventKind::Pitch, 64),
                        1,
                    ])
                } else {
                    assert_eq!(&chunk[..17], &[0.25; 17]);
                    assert!(chunk[17..].iter().all(|&v| v == 0.0));
                    let expected = vec![
                        id(EventKind::Program, 26),
                        id(EventKind::Pitch, 64),
                        id(EventKind::Tie, 0),
                    ];
                    assert_eq!(prompt.unwrap(), expected);
                    let mut result = expected;
                    result.extend([
                        id(EventKind::Velocity, 0),
                        id(EventKind::Shift, 25),
                        id(EventKind::Pitch, 64),
                        1,
                    ]);
                    Ok(result)
                }
            },
            |p| progress.push(p),
        )
        .unwrap();
        assert!(out.missing_eos.is_empty());
        assert_eq!(out.notes.len(), 1);
        assert_eq!(
            (out.notes[0].start, out.notes[0].end, out.notes[0].midi),
            (0.5, 5.25, 64)
        );
        assert_eq!(
            progress,
            vec![
                ProgressEvent {
                    completed: 0,
                    total: 2
                },
                ProgressEvent {
                    completed: 1,
                    total: 2
                },
                ProgressEvent {
                    completed: 2,
                    total: 2
                }
            ]
        );
    }

    #[test]
    fn eos_policy_and_invalid_tokens() {
        let t = MT3Tokenizer::released();
        let tie = t.id(EventKind::Tie, 0) as u32;
        let ok = transcribe_chunks(&[0.0], false, |_, _, _| Ok(vec![tie]), |_| {}).unwrap();
        assert_eq!(ok.missing_eos, vec![0]);
        assert!(matches!(
            transcribe_chunks(&[0.0], true, |_, _, _| Ok(vec![tie]), |_| {}),
            Err(MtabError::DecodeFailure { chunk_index: 0, .. })
        ));
        assert!(matches!(
            transcribe_chunks(&[0.0], false, |_, _, _| Ok(vec![99999, 1]), |_| {}),
            Err(MtabError::DecodeFailure { chunk_index: 0, .. })
        ));
    }

    #[test]
    fn empty_input_reports_empty_progress_without_generating() {
        let mut progress = Vec::new();
        let out = transcribe_chunks(
            &[],
            false,
            |_, _, _| panic!("no chunk expected"),
            |p| progress.push(p),
        )
        .unwrap();
        assert!(out.notes.is_empty());
        assert_eq!(
            progress,
            vec![ProgressEvent {
                completed: 0,
                total: 0
            }]
        );
    }
}
