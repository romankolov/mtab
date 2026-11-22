# mtab — muscriptor-tab

[![CI](https://github.com/romankolov/mtab/actions/workflows/ci.yml/badge.svg)](https://github.com/romankolov/mtab/actions/workflows/ci.yml)
[![Release](https://github.com/romankolov/mtab/actions/workflows/release.yml/badge.svg)](https://github.com/romankolov/mtab/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-green.svg)](LICENSE)

**Audio → guitar tablature in pure Rust**, powered by [MuScriptor](https://github.com/muscriptor/muscriptor)
(Kyutai + Mirelo). Point it at any audio file plus a MuScriptor weights file
(`small` / `medium` / `large`) and produce preliminary **MIDI + Guitar Pro 5**
output. The goal is accurate tuning, fingering, rhythm and techniques; several
of these are not yet implemented or reliably inferred.

> **Status: in development.** The audio frontend, transformer inference,
> chunk decoder, and MIDI writer have preliminary implementations. The CLI
> supports raw-note JSONL and preliminary MIDI export with string/fret assignment.
> Experimental, restricted GP5, MusicXML and ASCII tab export are available. The preliminary
> fingering search has not been compared with tab-labeler-v3 on held-out audio.
> The official small checkpoint has
> been checked against upstream on a synthetic plucked-string fixture and
> tested on one real-guitar excerpt. Corpus-level accuracy and end-to-end
> performance targets remain unverified.

## Current usage

Supply a local MuScriptor safetensors checkpoint matching `--size` (there is no
bundled or automatic download):

```console
mtab transcribe song.wav --size small --weights ./muscriptor-small.safetensors \
    --instruments guitar --format jsonl -o notes.jsonl
```

For preliminary unquantized MIDI, experimental notation exports and an optional JSON tablature IR:

```console
mtab transcribe song.wav --size small --weights ./muscriptor-small.safetensors \
    --instruments guitar --tuning standard --capo 0 --grid off \
    --dynamics --emit-ir song.json -o song.mid,song.gp5,song.musicxml,song.txt
```

Automatic tuning and capo detection are experimental. The detector analyses
bounded monophonic pitch evidence and scores possible open strings; it warns
and falls back to standard tuning and capo 0 when evidence is ambiguous. If
one setting is explicit, the fallback preserves it. Chords, fretted notes,
bass and other instruments can fool the uncalibrated heuristic, so verify the
result or set `--tuning` and `--capo` explicitly. Run
`mtab detect-tuning song.wav` for ranked hypotheses, or
`mtab detect-tuning --list` for tunings.

Beat tracking and quantization are not implemented: MIDI preserves note times
with a fixed 120 BPM / 4/4 map, apart from clipping sequential boundary
roundoff of at most one nanosecond. Fret assignment may
fail on unplayable or beam-pruned polyphony; it is not a quality evaluation.
The optional `--allow-wide-stretch` retries a failed assignment with a maximum
seven-fret span instead of four, without dropping notes or allowing same-string
overlaps. It emits a warning when used: the resulting grips may be physically
uncomfortable or impractical and require review.
GP5 output always uses a fixed 120 BPM, 4/4 and a 1/32-note grid, regardless
of `--grid`; the CLI reports the largest quantization shift. It supports up
to 15 six- or seven-string tracks, titles representable in Windows-1252
(without control characters, up to 255 encoded bytes), and GP5's discrete
velocity levels. Other Unicode titles require renaming the audio file. It can
encode dead-note, palm-mute and restricted bend markings supplied in the library
IR, but does not infer them from audio. It rejects same-string quantization
collisions, unsupported bends, eight-string tracks and other unsupported
notation rather than dropping notes.
Two generated fixtures were opened and GP5-re-exported with TuxGuitar 1.6.6;
their tested musical fields and bend points survived. This is not a general
compatibility or playback guarantee, and real-checkpoint CLI-to-GP5 output has
not yet been validated. MIDI remains
unquantized. Pitch bends are not yet inferred. The MIDI writer isolates bent
overlapping notes onto separate channels and reports an error if its available
15 melodic channels are exhausted. Bend contours use stepped control points
and support up to ±200 cents. MIDI files do not encode tablature fingering; the
optional IR does. No DAW round-trip validation has yet been performed.

Omit `-o` to write JSONL to stdout. Each line is a raw note with `start`, `end`
(seconds), `midi`, `velocity`, and `instrument` (program) fields; progress and
warnings go to stderr. Velocities default to upstream's fixed value 100. The
optional `--dynamics` flag estimates relative velocity (30–127) from the
attack envelope and local RMS, using the mixed audio. This experimental
heuristic is uncalibrated; overlapping notes and other instruments can bias
it, and notes without usable attack energy retain their original velocity.
`--strict-eos` makes an incomplete generated chunk an error, `--beam-size` selects beam search, and `--temperature` enables seeded
sampling (`--seed`). The current reference CPU implementation uses f64 and is
not yet performance-optimized; device acceleration and streaming audio are
not implemented. Local checkpoints only; URL/Hugging Face fetching is pending.

The intended finished product accepts just a model size and weights and emits
fully featured MIDI and Guitar Pro tablature; the remaining features are tracked in
[PROMPT.md](PROMPT.md).

## Weights & licensing

- **Code:** [MIT](LICENSE) — see [NOTICE](NOTICE) for third-party components.
- **GuitarSet excerpt:** CC BY 4.0; see its [attribution and changes](crates/mtab-cli/tests/fixtures/guitarset/README.md).
- **MuScriptor weights:** **CC BY-NC 4.0 (non-commercial), gated on
  [Hugging Face](https://huggingface.co/MuScriptor)**. They are **not** included
  in this repo or in release artifacts. Accept the license on the model page,
  authenticate (`hf auth login` / `HF_TOKEN`), and pass your own weights file.

## Development

The workspace supports Rust 1.75 or later with the checked-in lockfile; CI
also tests that minimum version.

```console
$ cargo test --workspace      # tests (see .github/workflows/ci.yml)
$ cargo run -p mtab-cli -- --models
```

CI runs lint + tests on Linux/macOS/Windows for every push; version tags
(`v*`) trigger cross-platform release builds ([release.yml](.github/workflows/release.yml));
weekly [nightly.yml](.github/workflows/nightly.yml) runs extended tests,
`cargo audit`, and an optional GuitarSet eval.

The build agent commits and pushes after **every** step — see
[PROMPT.md §11](PROMPT.md).

### Optional small-checkpoint smoke test

After accepting the conditions on the [small model page](https://huggingface.co/MuScriptor/muscriptor-small)
and downloading `model.safetensors` yourself, you can run the ignored
end-to-end check against that local file:

```sh
MTAB_SMALL_WEIGHTS=/path/to/model.safetensors \
  cargo test -p mtab-cli --release --test small_checkpoint_smoke -- --ignored --nocapture
```

The test uses a synthetic plucked-string waveform and runs the CLI twice for
byte-identical JSONL, then a third time to verify generated MIDI note events and tablature IR
against the JSONL plus the optional dynamics estimator.
It defaults to 80 generated tokens; override with `MTAB_SMOKE_MAX_GEN_LEN`.
This is a **smoke test**, not an accuracy test. It never downloads or commits
weights.

### GuitarSet oracle-note fingering evaluation

The preliminary evaluator scores string assignment using GuitarSet's **reference
notes**, not transcribed audio. Download the CC BY 4.0 annotation archive and
unzip its `.jams` files, then run:

```sh
mtab eval --corpus guitarset --oracle-notes --annotations /path/to/annotations \
  --report oracle-report.md
```

[Our 360-excerpt report](benches/guitarset-oracle.md) documents a 59.65%
all-note string accuracy for the bounded beam solver, with 40 excerpts left
unassigned. An optional wider-stretch retry recovered 38 of those excerpts and
scored 67.04%; every recovered excerpt contained a span above four frets. The
collision-ignoring per-note greedy baseline scored 45.40%. The same corpus
was used to diagnose the solver, so these are not held-out results. This is **not** end-to-end audio accuracy or a tab-labeler-v3 comparison; the
full acceptance evaluation remains outstanding. The optional `Nightly`
workflow can run this annotation-only check with `run_oracle=true`.

### Real-guitar note smoke and evaluation

The official small model was also tested on one five-second, CC BY 4.0
GuitarSet guitar excerpt. [The report](benches/guitarset-small-note-smoke.md)
records 82.35% pitch-plus-onset F1 and 52.94% pitch-plus-onset-plus-offset F1
on that **single development excerpt**; neither is a corpus accuracy claim.
The optional ignored test uses a user-provided licensed checkpoint. To score
other note JSONL against the matching JAMS annotation:

```sh
mtab eval --corpus guitarset --annotations recording.jams \
  --predictions notes.jsonl --clip-duration 5
```

This is note-event matching, not full transcription or tablature evaluation.
The Nightly `run_real_small` option performs the fixture check using an
accepted Hugging Face account's `HF_TOKEN` repository secret; no model
weights are uploaded as artifacts.

### Experimental ASCII library exporter

`mtab_export::ascii::ascii_tab` returns text and its notation plan; the
`ExportFormat::AsciiTab` dispatcher writes `.txt`/`.tab` files. It shares
the GP5 planner's constant integer tempo, 4/4, six/seven-string and 1/32-grid
constraints. Each bar shows beat durations above aligned strings. The legend
explains ties, dead notes, palm mutes and the supported bend notation; velocities
are omitted. It is a readable quantized view, not a lossless performance format.
The CLI supports `.txt` and `.tab` outputs, alone or alongside other formats.

### Experimental MusicXML library exporter

`mtab_export::musicxml::musicxml` returns MusicXML 3.1 and the quantized
notation plan; `ExportFormat::MusicXml` writes `.musicxml` or `.xml`. It emits
standard concert-pitch and tablature staves, tuning, capo, bars, durations,
chords, ties, string/fret, dead-note heads and palm-mute annotations. It uses
the restricted GP5 planner and rejects timed bends rather than discarding
them. Velocities are not encoded; the two notation staves duplicate notes and
may double playback in applications. It has not been checked in a notation
application. CI validates the W3C 3.1 schema and reads a fixture with
music21; Python is test-only. The CLI supports `.musicxml` and `.xml`, alone
or alongside other formats.

### Experimental GP5 library encoder

`mtab_export::gp5::gp5_bytes` encodes the currently restricted GP5 5.00 subset:
constant integer tempo, 4/4, six or seven strings, and a 1/32-note grid. It
preserves explicit dead-note and palm-mute marks and a restricted set of bends,
but rejects unsupported contours, other techniques, explicit source ties, chord
symbols, variable tempo/meter and quantization collisions rather than silently
discarding them. A bend must start at the note attack, peak at an interior
multiple of 1/12 of its original duration, rise by exactly 100 or 200 cents,
and fit one notated beat without a tie or an intervening note boundary. Bent
dead notes are rejected. The peak's relative position is preserved on the
quantized beat, so its absolute time shifts with the note boundaries. GP5
stores interpolated bend points; the MIDI exporter currently emits stepped
pitch-wheel events at those points, so their playback can differ. Velocity is
rounded to GP5 dynamic levels. Two small fixtures have been visually checked and re-exported in TuxGuitar
1.6.6; see `benches/m6-tuxguitar.md` for scope and limitations. Guitar Pro
itself has not yet been checked.

An independent PyGuitarPro round-trip fixture check is required in CI. To run
it locally, install `PyGuitarPro==0.10.1` in a Python environment and use:

```sh
MTAB_PYGUITARPRO_PYTHON=/path/to/python cargo test --locked -p mtab-export \
  --test gp5_roundtrip -- --ignored --nocapture
```

Python and PyGuitarPro are test-only; the Rust application does not need them.

### Optional independent MIDI reader check

Produce JSONL and MIDI from the same input and generation settings, then use
`python -m pip install mido==1.3.3` and run
`python tools/check_midi_mido.py notes.jsonl song.mid`. The development-only
checker compares note events and the fixed tempo/meter using Mido; it is not a
DAW playback or round-trip test. Python is not required by the Rust application.

### Optional small-checkpoint upstream parity test

With a local upstream MuScriptor checkout and its Python dependencies (`torch`,
`numpy`, `einops`, `safetensors`), generate a temporary PyTorch reference and
compare it with the Rust implementation. This test uses the same synthetic
fixture and does not embed or distribute model-derived goldens:

```sh
python tools/gen_real_small_reference.py \
  --muscriptor-root /path/to/muscriptor \
  --weights /path/to/model.safetensors --output /tmp/mtab-small-reference.json
MTAB_SMALL_WEIGHTS=/path/to/model.safetensors \
MTAB_SMALL_REFERENCE=/tmp/mtab-small-reference.json \
  cargo test -p mtab-core --release --test real_small_parity -- --ignored --nocapture
```

The test compares every generated token and every masked/unmasked logit row
for greedy and forced-prompt decoding. The official small checkpoint passed on
the fixture; see [benches/report.md](benches/report.md) for the scope and
measured errors. The reference is generated with upstream PyTorch float32;
the Rust inference path uses f64.
