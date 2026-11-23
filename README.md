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

Optional experimental analysis flags: `--techniques` suggests dead notes and
palm mutes, and `--chords` adds chord symbols from the detected notes. Both are
uncalibrated heuristics (see below), and both are rejected with `--format
jsonl`, where there is no tablature IR to annotate.


Automatic tuning and capo detection are experimental. The detector analyses
bounded monophonic pitch evidence and scores possible open strings; it warns
and falls back to standard tuning and capo 0 when evidence is ambiguous. If
one setting is explicit, the fallback preserves it. Chords, fretted notes,
bass and other instruments can fool the uncalibrated heuristic, so verify the
result or set `--tuning` and `--capo` explicitly. Run
`mtab detect-tuning song.wav` for ranked hypotheses, or
`mtab detect-tuning --list` for tunings.

The audio library has experimental bounded spectral-flux onset and
periodicity-evidence APIs, plus an experimental beat-path search conditioned
on a supplied tempo ([details and independent checks](benches/m4-onset-envelope.md)).
These are not a complete automatic beat tracker and do not change CLI output.
A limited [50-recording diagnostic](benches/m4-guitarset-beat-path.md) found
substantial beat-phase and tempo errors, even when reference tempo was supplied.
A subsequent [predeclared tightness experiment](benches/m4-beat-tightness-results.md)
reduced some errors but did not make the tracker production-ready.

No reliable automatic beat tracking is enabled by default: MIDI preserves note times with a fixed
120 BPM / 4/4 map by default, apart from clipping sequential boundary
roundoff of at most one nanosecond. Use `--bpm 96` to specify a constant
integer tempo (4–255 BPM) instead; it sets the musical grid and metadata but
does not change the recorded timing or detect beats. Tempo overrides are not
applied to raw JSONL. MIDI is subject to tick and tempo-event rounding.

For MIDI only, `--beat-times beats.json` instead accepts manually supplied
quarter-beat times in seconds, for example `[0.0, 0.5, 1.1, 1.8, 2.3]`.
A first beat after zero is supported using a **synthetic pre-roll tempo** to
align it to a quarter-note boundary: the pre-roll beat count is the offset
divided by the first interval, rounded to the nearest integer and at least one.
This does not infer earlier beats, a downbeat, meter, or pickup measures; 4/4
bars still start at audio time zero. The CLI warns about this convention and
uses the final interval's tempo beyond the last beat. There is no automatic beat detection in this mode. Provide 2–100,001 finite, nonnegative, strictly increasing
times, with intervals from 1 to 16,777,215 microseconds. The synthetic pre-roll
interval has the same bounds and is limited to 100,000 beats; the UTF-8 JSON
file must be at most 2 MiB. The file is
validated and read before model loading. Preflight rejects paths and existing
hard links that alias inputs or other outputs, but cannot protect against files
changed concurrently after the check. This option conflicts with `--bpm`, raw JSONL, GP5, MusicXML and ASCII outputs.

```console
mtab transcribe song.wav --size small --weights ./muscriptor-small.safetensors \
    --tuning standard --capo 0 --beat-times beats.json --emit-ir song.json -o song.mid
```

An opt-in `--experimental-beat-tracking` estimates beats from the audio for
MIDI output only. It is **not production quality**: on a 50-recording GuitarSet
second-take diagnostic it achieved 53.31% beat micro F1, and a synthetic louder-
offbeat control scored 0%. It can select half/double tempo or offbeats, reports
no confidence, does not detect downbeats or meter, and uses a global tempo
candidate. It analyzes at most 30,000 onset frames (about five minutes), fails
rather than producing a tempo if fewer than two beats are found, and assumes
4/4 with a synthetic pre-roll for an offset first beat. Note playback retains
raw seconds subject to MIDI rounding. This option conflicts with `--bpm` and
`--beat-times`, and is not available for JSONL, GP5, MusicXML or ASCII. Review
the MIDI grid manually; see [the diagnostic](benches/m4-beat-tightness-results.md).

Fret assignment may fail on unplayable or beam-pruned polyphony; it is not a quality evaluation.
The optional `--allow-wide-stretch` retries a failed assignment with a maximum
seven-fret span instead of four, without dropping notes or allowing same-string
overlaps. It emits a warning when used: the resulting grips may be physically
uncomfortable or impractical and require review.

Alternatively, `--partial-report omitted.json` opts into a strict search followed,
on unplayable tracks, by a four-fret-span partial search. This may omit notes
from **every output, including MIDI**. The JSON report records each omitted
raw note and its zero-based input index, even when different instruments form
separate tracks; it is written with zero omissions when no retry is needed.
The CLI warns when notes are omitted. Review the report and do not treat the
result as a complete transcription. The partial beam is approximate and may
omit more notes than necessary; it is not compatible with `--allow-wide-stretch`
or raw JSONL mode. Like other output files, report writes are not transactional
if a filesystem error occurs.

```console
mtab transcribe song.wav --size small --weights ./muscriptor-small.safetensors \
    --tuning standard --capo 0 --partial-report omitted.json \
    -o partial.mid,partial.gp5
```

GP5, MusicXML and ASCII use the selected constant BPM (120 by default), 4/4
and a 1/32-note grid, regardless of `--grid`; the CLI reports the largest quantization shift. It supports up
to 15 six- or seven-string tracks, titles representable in Windows-1252
(without control characters, up to 255 encoded bytes), and GP5's discrete
velocity levels. Other Unicode titles require renaming the audio file. It can
encode dead-note, palm-mute, let-ring and restricted bend markings supplied in the library
IR, but does not infer them from audio. It rejects same-string quantization
collisions, unsupported bends, eight-string tracks and other unsupported
notation rather than dropping notes.
An opt-in real-small CLI-to-GP5/IR test is available in `small_checkpoint_smoke`
(the gated checkpoint and PyGuitarPro are required); it has not yet been run.
Two generated fixtures were opened and GP5-re-exported with TuxGuitar 1.6.6;
their tested musical fields and bend points survived. This is not a general
compatibility or playback guarantee, and real-checkpoint CLI-to-GP5 output has
not yet been validated. MIDI remains
unquantized. Pitch bends are not inferred by default. The MIDI writer isolates bent
overlapping notes onto separate channels and reports an error if its available
15 melodic channels are exhausted. Supplied bend contours are interpolated to
pitch-wheel steps at up to 50 Hz (capped at 2,048 steps per segment) and
support up to ±200 cents. MIDI files do not encode tablature fingering; the
optional IR does. A synthetic bend/release was rendered and measured with
FluidSynth and two soundfonts; see `benches/m7-midi-bend-playback.md`. One
synthetic bend was also imported and played in Ardour 8.12; its saved imported
source preserved wheel events but dropped the pitch-sensitivity RPN. See
`benches/m4-ardour-midi.md`. This is not a general DAW round-trip guarantee.

Omit `-o` to write JSONL to stdout. Each line is a raw note with `start`, `end`
(seconds), `midi`, `velocity`, and `instrument` (program) fields; progress and
warnings go to stderr. Velocities default to upstream's fixed value 100. The
optional `--dynamics` flag estimates relative velocity (30–127) from the
attack envelope and local RMS, using the mixed audio. This experimental
heuristic is uncalibrated; overlapping notes and other instruments can bias
it, and notes without usable attack energy retain their original velocity. A
controlled FluidSynth test checks ordering on one synthetic velocity sequence,
not velocity calibration or real-guitar accuracy; see
`benches/m7-dynamics-playback.md`.
`--strict-eos` makes an incomplete generated chunk an error, `--beam-size` selects beam search, and `--temperature` enables seeded
sampling (`--seed`). The current reference CPU implementation uses f64 and is
not yet performance-optimized; device acceleration and streaming audio are
not implemented. Local checkpoints only; URL/Hugging Face fetching is pending.

The intended finished product accepts just a model size and weights and emits
fully featured MIDI and Guitar Pro tablature; the remaining features are tracked in
[PROMPT.md](PROMPT.md).

## Offline Hugging Face cache

In addition to a local safetensors file, `--weights small` (or `medium`/`large`)
resolves that variant from an **existing** Hugging Face Hub cache; the keyword
must match `--size`. `--weights hf://MuScriptor/muscriptor-small` resolves an
existing cache entry by repository. These lookups use `refs/main` and its
40-hex-digit snapshot's `model.safetensors`; missing references or files fail
rather than choosing an arbitrary snapshot or downloading. `HF_HUB_CACHE`,
`HF_HOME`, `XDG_CACHE_HOME`, and the home-directory default determine the Hub
cache location, in that order. The loader still validates the checkpoint; a
cache path or hash alone does not prove its contents or grant a license.
Direct HTTP(S) fetching and the `download` helper remain unimplemented. Cache
weights and aliases are protected from output overwrites, subject to concurrent
filesystem changes after checking.

## Weights & licensing

- **Code:** [MIT](LICENSE) — see [NOTICE](NOTICE) for third-party components.
- **GuitarSet excerpt:** CC BY 4.0; see its [attribution and changes](crates/mtab-cli/tests/fixtures/guitarset/README.md).
- **MuScriptor weights:** **CC BY-NC 4.0 (non-commercial), gated on
  [Hugging Face](https://huggingface.co/MuScriptor)**. They are **not** included
  in this repo or in release artifacts. Accept the license on the model page,
  authenticate (`hf auth login` / `HF_TOKEN`), and pass your own weights file.

## Documentation and examples

- [`docs/architecture.md`](docs/architecture.md) — crate layout, pipeline, the
  shared notation plan, and the testing strategy.
- [`docs/tuning-detection.md`](docs/tuning-detection.md) — how tuning/capo
  detection scores hypotheses and why it can be wrong.
- [`docs/techniques.md`](docs/techniques.md) — what the IR carries, what each
  export can represent, what is suggested from audio and what is not inferred.
- [`docs/weights.md`](docs/weights.md) — obtaining and converting MuScriptor
  weights legally, and the auxiliary-model license picture.

Runnable examples live in `crates/mtab-cli/examples/`:

```console
cargo run --release -p mtab-cli --example basic -- song.wav ./weights.safetensors
cargo run --release -p mtab-cli --example streaming -- long.wav ./weights.safetensors
cargo run -p mtab-cli --example custom_guitar
```

`basic` is the happy path (audio in, MIDI/GP5/MusicXML/ASCII out), `streaming`
drives the chunk loop itself so only one five-second chunk is in memory at a
time, and `custom_guitar` builds a tablature document by hand for a
seven-string with a capo and exports it without any model.

## Development

The workspace supports Rust 1.75 or later with the checked-in lockfile; CI
also tests that minimum version.

```console
$ cargo test --workspace      # tests (see .github/workflows/ci.yml)
$ cargo run -p mtab-cli -- --models
```

CI runs lint + tests on Linux/macOS/Windows for every push. Version tags
(`v*`) trigger cross-platform release builds ([release.yml](.github/workflows/release.yml));
manual dispatch runs the same builds, release-mode tests, binary smoke checks,
archive checks and checksums without publishing a release. No weights are
packaged. Weekly [nightly.yml](.github/workflows/nightly.yml) runs extended tests,
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
scored 67.04%; every recovered excerpt contained a span above four frets. A
separate four-fret partial retry scored 68.50% and left 59 notes unassigned,
recovering at least one note in every strict-failure excerpt. This comparison
is affected by strict failures counting the whole excerpt as unassigned; the
partial output is incomplete. These corpus-informed results are not held-out.
The collision-ignoring per-note greedy baseline scored 45.40%. The same corpus
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

For emitted standard-tuning, capo-zero IR, an additional single-recording
framewise evaluation compares note-derived pitch sets and string/fret positions:

```sh
mtab transcribe recording.wav --size small --weights /path/to/model.safetensors \
  --tuning standard --capo 0 -o recording.mid --emit-ir recording.json
mtab eval --annotations recording.jams --tab-ir recording.json \
  --clip-duration 5 --report recording-eval.md
```

To transcribe and score an explicit list of local audio recordings end to end,
use a separate audio manifest, with paths relative to that manifest:

```json
[
  {"annotation": "a.jams", "audio": "a.wav", "duration": 30.0},
  {"annotation": "b.jams", "audio": "b.wav", "duration": 27.0}
]
```

```sh
mtab eval --audio-manifest audio-manifest.json --size small \
  --weights /path/to/model.safetensors --max-gen-len 512 --report eval.md
```

For a locally extracted GuitarSet 1.1.0 mono-mic corpus, the optional
preparation helper pairs `*.jams` with matching `*_mic.wav` files:

```sh
python tools/make_guitarset_audio_manifest.py --annotations /data/guitarset/annotation \
  --audio /data/guitarset/audio_mono-mic --output audio-manifest.json
```

It expects 360 annotations unless `--allow-subset` is explicitly passed,
checks that every matching WAV exists, and uses a duration just below the WAV
end. It writes absolute paths and SHA-256 hashes for each pair. It does not
download the dataset, prove that 360 files are the official set, establish a
held-out split, or replace recording selection and provenance review. Python is a preparation and
test dependency only, not an mtab runtime dependency.

The model is loaded once. The evaluator uses only each audio prefix specified
by its duration, greedy guitar-masked decoding with strict EOS, and standard
tuning/capo zero/default fingering. Pass `--partial-fingering` to evaluate the
opt-in strict-then-partial policy instead. The report lists omitted generated
notes for each successful entry (before window clipping); omitted notes may have
been true or false positives, so their exclusion changes precision and recall.
Omission counts for failed entries are unknown. This evaluation mode is tested
with synthetic notes but has not yet been run with an official checkpoint.
If inference or fingering fails, that entry
is scored as an empty prediction, marked failed in the report, and the command
returns an error after writing the report; such failures contribute false
negatives but have unknown hypothetical false positives. It validates that
all listed paths exist, but cannot verify that audio and annotations are paired
correctly. When both `audio_sha256` and `annotation_sha256` are supplied for an
entry, they are verified before inference; supplying just one is an error.
Hashes identify bytes, not dataset authenticity. The report records manifest
and model-weight hashes. List the intended complete split explicitly and
preserve the manifest and setup before drawing any conclusions. The
Nightly `run_eval` option currently runs only the checked-in five-second
**development** excerpt, not the full corpus. It requires an authorized
`HF_TOKEN` repository secret and has not been run in the current environment.

For several precomputed recordings, make a JSON manifest (paths are relative
to the manifest) and pass `--tab-manifest manifest.json --report eval.md`:

```json
[
  {"annotation": "a.jams", "ir": "a.ir.json", "duration": 30.0},
  {"annotation": "b.jams", "ir": "b.ir.json", "duration": 27.0}
]
```

The manifest scorer micro-aggregates frame counts and includes a per-recording
summary. It requires every listed input, rejects duplicate annotations or IR
files, and fails instead of silently dropping a bad entry. It does not perform
inference or prove that the selection is representative. Record the corpus
split, model, and decoding settings separately.

The IR score uses 10 ms frame centers and fixed MIDI notes (not continuous
pitch contours), reports multi-pitch and tab-position precision/recall/F1,
as well as one-to-one pitch/onset and string-fret/onset note-event scores
(with and without offset matching),
and gives a pitch-matched wrong-string rate plus a TDR-style ratio: correct
string/fret activations divided by correctly detected unique-pitch activations
([Wiggins and Kim 2019, Eq. 8](https://archives.ismir.net/ismir2019/paper/000033.pdf)).
The ratio can exceed 100% for simultaneous unisons, so it is not the complement
of the wrong-string rate. Note-event-derived references and our frame grid do
not reproduce the original TabCNN protocol; these scores do not establish
corpus quality or a tab-labeler-v3 comparison.
The specified duration must match the audio being evaluated. Note-event JSONL
and IR frame metrics use different matching definitions. The Nightly `run_real_small` option performs the fixture check using an
accepted Hugging Face account's `HF_TOKEN` repository secret; no model
weights are uploaded as artifacts.

A transcription with no detected notes produces a warning and one empty guitar
track for notation export; it does not invent notes. This is covered by a
synthetic-model CLI test, not a real-checkpoint quality assessment.

### Experimental monophonic pitch-track API

`mtab_audio::pitch_track::PitchTracker` uses the existing FFT-based YIN-style
periodicity estimator to return frame-centered F0 values for a bounded region
of mono 16 kHz audio. Frames stay within the requested region. Synthetic sine,
sweep, silence and noise tests exercise the implementation. It is **not**
source separation and is not reliable on arbitrary polyphonic mixtures.
`mtab_audio::bend_analysis::suggest_bends` can also suggest simple bend shapes
for non-overlapping notes, with explicit abstention reasons. The optional
`--experimental-bends` flag attaches candidates to MIDI/IR output and does not
detect other effects.
It skips detected overlapping notes across all tracks, short notes, and other
insufficient evidence, but undetected mixture sounds can still produce false
bends. Its output is uncalibrated and requires manual review. It is MIDI-only,
and the reported candidate count is not confidence. Synthetic checks include
a bend through partial fingering and an independent Mido playback-time/wheel
inspection; the tiny-model transcription is not real-checkpoint validation.
One isolated real GuitarSet note provides a small F0 sanity check, not a
bend-quality benchmark; see `benches/m7-pitch-fixture.md`.

`mtab_audio::articulation::suggest_articulations` implements the
dead-note/palm-mute half of `--techniques`. For every supplied note it measures
a 32 ms attack window against a 32 ms tail window 16 ms later, then classifies
by attack-to-tail energy loss and spectral centroid with **fixed thresholds**:
a fast decay with a dark spectrum is a dead note, a milder decay with a
darkening spectrum is a palm mute, and everything else is left open. Decay and
centroid are gain-invariant, but the classifier is not trained, not calibrated
and not per-instrument: drums, bass and other mixture content share the same
windows, window placement follows transcription timing rather than a detected
onset, and short or silent notes are reported rather than guessed. Suggested
marks are attached to the IR (and so reach MIDI and every notation export) with
a warning; review them before trusting a tab. Synthetic checks cover open,
dead, palm-muted, short, silent and out-of-range notes.

`mtab_guitar::suggest_chords` (the `--chords` flag) builds chord symbols from
the detected notes using a fixed vocabulary — the twelve roots C, C#, D, Eb, E,
F, F#, G, Ab, A, Bb, B with `""`, `m`, `dim`, `7`, `maj7` and `m7` — and emits
nothing for notes that do not form exactly one of those chords. Symbols span at
least 0.25 s. On the GuitarSet oracle diagnostic coverage was 15.8% with 87.0%
precision when it did fire (`benches/m7-guitarset-chord-results.md`); it is not
a chord-recognition benchmark and the vocabulary makes it abstain often.

### Experimental ASCII library exporter

`mtab_export::ascii::ascii_tab` returns text and its notation plan; the
`ExportFormat::AsciiTab` dispatcher writes `.txt`/`.tab` files. It shares
the GP5 planner's constant integer tempo, 4/4, six/seven-string and 1/32-grid
constraints. Each bar shows beat durations above aligned strings. The legend
explains ties, dead notes, palm mutes, let-ring and the supported bend notation; velocities
are omitted. It is a readable quantized view, not a lossless performance format.
The CLI supports `.txt` and `.tab` outputs, alone or alongside other formats.

### Experimental MusicXML library exporter

`mtab_export::musicxml::musicxml` returns MusicXML 3.1 and the quantized
notation plan; `ExportFormat::MusicXml` writes `.musicxml` or `.xml`. It emits
standard concert-pitch and tablature staves, tuning, capo, bars, durations,
ties, string/fret, dead-note heads, palm-mute, let-ring, vibrato
(wavy-line start/continue/stop), hammer-on and slide (`notations/slide`
start/stop) annotations, and supplied chord symbols as `<harmony>` root/kind
on the first staff. It uses
the restricted GP5 planner and rejects timed bends rather than discarding
them. Velocities are not encoded; the two notation staves duplicate notes and
may double playback in applications. MuseScore 3.2.3 imported a fixture but changed some string/fret assignments
and dropped capo and palm-mute information on re-export; **do not rely on
this MusicXML for those details in MuseScore**. See `benches/m6-musicxml.md`.
CI validates the W3C 3.1 schema and reads a fixture with music21; Python is
test-only. The CLI supports `.musicxml` and `.xml`, alone
or alongside other formats.

### Experimental GP5 library encoder

`mtab_export::gp5::gp5_bytes` encodes the currently restricted GP5 5.00 subset:
constant integer tempo, 4/4, six or seven strings, and a 1/32-note grid. It
preserves explicit dead-note, palm-mute, let-ring and vibrato marks (GP5 stores
vibrato as a boolean, not rate/depth), hammer-on/pull-off and adjacent-note
slide markers, supplied chord symbols as beat chord diagrams (name plus
root/quality; no fretboard diagram is invented), and a restricted set of bends,
but rejects unsupported contours, other techniques, explicit source ties,
variable tempo/meter and quantization collisions rather than silently
discarding them. A bend must start at the note attack, peak at an interior
multiple of 1/12 of its original duration, rise by exactly 100 or 200 cents,
and fit one notated beat without a tie or an intervening note boundary. Bent
dead notes are rejected. The peak's relative position is preserved on the
quantized beat, so its absolute time shifts with the note boundaries. GP5
stores interpolated bend points; the MIDI exporter samples linear pitch-wheel
ramps, so timing and playback can still differ between applications. Velocity is
rounded to GP5 dynamic levels. Two small fixtures have been visually checked and re-exported in TuxGuitar
1.6.6; see `benches/m6-tuxguitar.md` for scope and limitations. Guitar Pro
itself has not yet been checked. Explicit let-ring and vibrato markings were
separately checked with PyGuitarPro and the MusicXML XSD; notation-app round
trips and playback have not been tested, and nothing infers vibrato from audio
(`--techniques` remains unavailable). MIDI preserves the supplied note-off time and does not
extend it merely because a let-ring mark is present.

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
