# Architecture

`mtab` is a pure-Rust workspace that turns audio into a guitar tablature
document and then into playable files. No Python, PyTorch or ONNX runtime is
needed at run time; Python appears only in tests, as an independent reader for
the formats we write.

## Crates

| Crate | Responsibility | Key modules |
| --- | --- | --- |
| `mtab-core` | Container, math and model plumbing: errors, safetensors loading, the weight/`LmConfig` mapping, the decoder-only transformer, MT3 tokenizer and token semantics, chunked decoding, MIDI-independent IR types, progress events | `error`, `safetensors`, `lm`, `mt3`, `chunk`, `model`, `event` |
| `mtab-audio` | Everything that touches samples: decode (Symphonia), mono mixdown, resample, the upstream-exact mel frontend, model-driven transcription, and the experimental analyses (beat, onset, periodicity, pitch tracking, bends, dynamics, articulation) | `decode`, `dsp`, `resample`, `mel`, `transcribe`, `automatic_beat`, `pitch_track`, `bend_analysis`, `dynamics`, `articulation` |
| `mtab-guitar` | Guitar domain logic: tunings and capo, tuning/capo scoring, the global string/fret assignment, rhythm/quantization helpers, technique types, chord symbols | `tuning`, `detection`, `fingering`, `rhythm`, `tempo`, `tab`, `chords` |
| `mtab-export` | Writers and the notation plan: MIDI with real tempo maps and bends, the restricted GP5 plan and encoder, MusicXML 3.1, ASCII tab | `midi`, `gp5_plan`, `gp5`, `musicxml`, `ascii` |
| `muscriptor-tab` | The §6.11 facade: `Transcriber::builder()` composing the four crates below into one entry point, with `Progress` callbacks and `Transcription::export_*` | `lib` |
| `mtab-cli` | The `mtab` binary: argument validation, the transcription pipeline, `detect-tuning`, `eval`, `inspect-weights` | `main`, `guitarset*` |

The CLI predates the facade and still calls the stage crates directly, which is
why it can reach options (`--partial-report`, `--emit-ir`, bend and
articulation inference) that the facade deliberately leaves out. The facade is
the supported library surface; the CLI is a superset of it.

## Pipeline

```
audio file ──decode──▶ mixed down, resampled to 16 kHz ──mel──▶ model input
                                                                    │
                          transformer + MT3 decode (chunked, forced ties)
                                                                    │
                                                              NoteEvents
                                                                    │
                       ┌────────── Tuning/Capo detection (optional) ─┘
                       ▼
              string/fret assignment (global DP over playable positions)
                       │
                       ▼
                 TabDocument (JSON IR)
                       │
        ┌──────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼              ▼
      MIDI            GP5          MusicXML        ASCII tab
```

Two properties matter more than the rest:

1. **The frontend is a port, not an approximation.** The mel filterbank, window,
   padding mode and power convention match upstream MuScriptor bit-for-bit
   within rounding, and golden tests pin them. A small mismatch silently
   degrades every downstream number, so it is checked first.
2. **Guessing is explicit.** Anything the pipeline cannot represent is either
   rejected with a reason (for example an unsupported bend) or attached with a
   warning and marked experimental (tuning, bends, dynamics, articulation,
   chords). The notation plan is shared by all three notation writers, so a
   document that cannot be notated fails before any file is written.

## The notation plan

`gp5_plan` quantizes a `TabDocument` into bars of a 1/32 grid at a constant
integer tempo in 4/4, splitting every note into tied beats. GP5, ASCII and
MusicXML all render from that one plan, so the three files always agree about
bar boundaries, tie splits, technique placement and chord anchors. The plan is
deliberately narrow: variable tempo, other meters, bends that do not fit the
supported contour, and unsupported techniques are errors rather than silent
data loss.

## Experimental analyses

These modules are opt-in and never change the model's decoded notes unless the
user asks for them:

- **Tuning and capo** (`mtab-guitar::detection`) scores tuning templates against
  open-string pitch evidence.
- **Bends** (`mtab-audio::bend_analysis`) tracks F0 inside isolated note spans.
- **Dynamics** (`mtab-audio::dynamics`) compares attack energy to local RMS.
- **Articulation** (`mtab-audio::articulation`) compares attack/tail energy and
  spectral centroid to suggest dead notes and palm mutes.
- **Chords** (`mtab-guitar::chords`) labels simultaneous pitch-class sets using
  a fixed vocabulary.
- **Rhythm** (`mtab-audio::automatic_beat`, `beat_path`) supplies a beat
  hypothesis for MIDI only; notation exports still require a constant tempo.

Each documents its own abstention behaviour and calibration status in its
module header. None of them is validated against a held-out corpus.

## Testing strategy

- Golden/parity tests against the upstream Python reference for the mel
  frontend, the transformer block and the full token stream.
- Round-trip tests through independent readers: PyGuitarPro for GP5,
  `mido` for MIDI, the W3C MusicXML 3.1 XSD plus `music21` for MusicXML.
- Property and exhaustive tests for the fingering search and the notation plan.
- Fuzz targets for the safetensors loader and the audio decoder.
- GuitarSet evaluation harnesses; see the README table for what has and has not
  been measured.
