# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning: [SemVer](https://semver.org/).

## [0.1.0] - 2026-11-24

First tagged release. Audio to guitar tablature in pure Rust on MuScriptor:
`.mid`, `.gp5`, `.musicxml` and ASCII tab out of any common audio format plus a
local or explicitly downloaded weights file.

Release archives for `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`,
`x86_64-apple-darwin`, `aarch64-apple-darwin` and `x86_64-pc-windows-msvc`, each
with `SHA256SUMS.txt` and no model weights (§10.1).

**Known limitations at this tag** — read these before relying on output:

- No GPU or SIMD path: f64 on CPU only, ~0.37× real time on `small` and ~0.03×
  on `large`. §6.13's targets assume a backend that does not exist yet.
- No stem separation (`--separate` is refused).
- No reliable beat tracker; notation uses a constant tempo and a fixed 1/32 grid.
- Tunings, capo and techniques are detected by uncalibrated heuristics.
- No GuitarSet corpus accuracy figure and no tab-labeler-v3 comparison, so
  fingering quality against the §9 acceptance criterion is unmeasured.

## [Unreleased]

### Changed
- The string/fret solver's open-string bonus dropped from 0.8 to 0.1. On the
  feasible GuitarSet excerpts the old value took the open string where the
  reference frets it 3,671 times; measured accuracy was 68.28% at 0.8, 70.70% at
  0.0 and 65.82% at 2.0. Feasible-excerpt string accuracy is now **70.14%**
  (wrong-string rate 29.86%) against an unchanged 46.24% greedy baseline.
  The exhaustive-search test oracle mirrors the cost function and was updated
  with it.
- The oracle report gained an error anatomy: 0.7% of feasible notes are chordal,
  so chord-shape work is not the next lever.

### Added
- The GuitarSet oracle-note report now decomposes by reference feasibility. 111 of
  360 excerpts (30.8%) contain a physically impossible instant — a fretted span
  over four frets between simultaneous notes, or one string sounding two pitches —
  dominated by the same pitch appearing on two neighbouring strings, which is
  hexaphonic pickup bleed rather than a double stop. On the 249 feasible excerpts
  the strict solver scores 68.28% string accuracy against 46.24% for the per-note
  greedy baseline, both measured over 41,138 notes. All 40 strict failures were
  already infeasible excerpts.
- `mtab download` works. It is the only networked command and runs only when
  invoked. It fetches the size and LFS SHA-256 from the Hub, resumes a
  `.<name>.part` with a `Range` request, verifies the checksum before renaming,
  deletes the file on mismatch, and refuses a file whose parameter count does not
  match `--size`. A 401/403 is reported as the license gate with its URL. Verified
  against the live gated Hub: 411,888,600 bytes, 122 tensors, sha256 match,
  resume from 300 MB, and one deliberately flipped byte rejected.
- `muscriptor-tab` facade crate implementing the §6.11 API: `Transcriber::builder()`
  with `.size()`, `.weights()`, `.model()` (already-loaded), `.guitar()`,
  `.decode()`, `.grid()` and `.tempo()`; `Transcriber::transcribe_file` and
  `transcribe_file_with_progress` emitting `Progress { stage, completed, total }`;
  and a `Transcription` with `export_midi` / `export_guitar_pro` /
  `export_musicxml` / `export_ascii` / `ascii_tab` / `ir_json`. `build()` loads
  the weights once and the transcriber serves any number of inputs. A grid other
  than 1/16 or off is rejected at `build()` rather than silently ignored. Eight
  tests run the facade end to end over the committed tiny checkpoint.
- Three runnable examples: `basic`, `custom_guitar`, `streaming`.
- `samples/showcase/`: a worked example that runs the whole CLI surface on a
  real recording — every output format, beam search, seeded sampling, fixed
  tuning/capo, unquantized timing, beat tracking, JSONL, a determinism check and
  an error path — and renders them into a browsable gallery. The
  `.github/workflows/sample.yml` workflow runs it on a hosted runner, fetching
  the gated weights into the runner's scratch space with a repository secret
  and failing the job if any `*.safetensors` reaches the upload directory.
- End-to-end goldens for the five §6.14.4 fixtures (`benches/goldens/`, IR JSON
  + GP5 + MIDI) driven by `crates/mtab-cli/tests/end_to_end_goldens.rs`. They
  run the real chunker, forced tie prelude, tracker, string/fret assignment and
  exporters over the committed GuitarSet excerpt and replay its annotated
  onsets as MT3 tokens, because the gated CC BY-NC weights are not available in
  CI. Two chunks per fixture, with notes ringing across the boundary.
- Supplied-IR vibrato marks now survive every notation export: GP5 note-level
  vibrato flags, ASCII `v` markers, and MusicXML wavy-line ornaments with
  start/continue/stop across split notes. Vibrato depth and rate are not
  encodable in GP5, and nothing infers vibrato from audio.
- Independent checks: `tools/check_gp5_vibrato.py` (PyGuitarPro) and
  `tools/check_musicxml_vibrato.py` (W3C XSD plus exact per-staff ornament
  positions) run in the same gated CI jobs as the existing let-ring checks.
- Hammer-ons and pull-offs are now distinct. The notation plan rejects a
  hammer-on that moves to a lower fret, a pull-off that moves to a higher fret,
  and a note marked as both; previously both marks collapsed into GP5's single
  legato flag and a contradicting direction was silently rewritten. MusicXML
  writes `<pull-off>`, which it previously omitted, and ASCII prints `o` for a
  pull-off instead of reusing `h` for both.
- `--techniques` now suggests dead notes and palm mutes from audio
  (`mtab_audio::articulation`), and `--chords` now derives chord symbols from
  the detected notes (`mtab_guitar::suggest_chords`) instead of both being
  rejected as unimplemented. Both are uncalibrated heuristics with fixed
  thresholds/vocabulary, attach to the IR with a warning, and are rejected
  together with `--format jsonl`. `--separate` is still rejected because it
  needs an auxiliary separation model.
- Supplied chord symbols now reach all notation exports. Only the fixed
  `mtab_guitar` root/vocabulary spellings are accepted; each symbol anchors to
  the containing beat of the first track (a trailing symbol to the last beat),
  two symbols on one beat or out-of-order symbols are rejected, and at most
  20,000 symbols of at most one hour each are planned. GP5 writes a chord
  diagram with name, root and quality and no invented fingering; ASCII prints a
  `chd` row; MusicXML writes `<harmony>` with root-step/root-alter and kind.
  `tools/check_gp5_chords.py` and `tools/check_musicxml_chords.py` verify the
  output independently.
- Fixed the vendored GP5 chord writer, which emitted ints where the format uses
  bytes (and wrote six rather than seven string-fret slots), so files with
  chord diagrams were unreadable past the first chord. See
  `vendor/guitarpro/README-mtab.md`.
- Supplied hammer-on/pull-off and slide techniques now reach all notation
  exports. The planner validates them against the adjacent attack on the same
  string (different fret, no dead-note target, exact adjacency) because GP5
  stores only a flag with no target; GP5 gets hammer and legato/shift/picked
  slide flags, ASCII a legend for the same, MusicXML `hammer-on` and
  `notations/slide` start/stop. Independent checker scripts cover the flags and
  the MusicXML XSD order. Nothing infers these techniques from audio.

### Fixed
- The KV cache was stored as f64, doubling the largest allocation in a long
  decode. It is now fp32 and widened only at the point of use, which halves the
  cache: a beam-4 decode that was OOM-killed on a 2 GB machine now peaks at
  757 MB, and a `large` beam-4 run saves roughly 4 GB.
- A pitch closed and re-struck within the same frame decoded to a zero-length
  note, which every downstream consumer rejects; it now gets the same
  `MINIMUM_NOTE_DURATION_SEC` floor as drum hits and unclosed notes. Found by
  the new end-to-end goldens, which replay a real performance that restrikes a
  string on the beat.
- An overlong chord label crashed the notation planner in debug builds and
  produced a silently wrong root/quality in release builds, because root
  scoring subtracted string lengths (`roots - remainder`) and underflowed.
  Found by the new `ir_export` fuzz target; see `fuzz/README.md`.

### Security
- Audio decoding now rejects hostile containers instead of aborting. Bounded
  libFuzzer campaigns against the safetensors loader and the Symphonia decode
  path found five upstream defects: a zero-sample-rate `TimeBase` assert, an
  ADTS frame-count subtraction underflow, up-front multi-gigabyte allocations
  from forged FLAC picture/Vorbis-comment lengths, an FLAC application-block
  underflow, and an out-of-bounds AAC scalefactor-band index. Four Symphonia
  0.5.5 crates are vendored under `vendor/` with minimal hardening patches
  (MPL-2.0, edits documented per crate) and `mtab-audio` enforces absolute
  decoded-sample and duration limits. Regression seeds are tracked in
  `fuzz/seeds/` and two decode tests cover zero-rate and truncated headers.
  A single-core, time-boxed campaign is not proof of memory safety.
- Opt-in `cargo-fuzz` targets for the safetensors loader and audio decoding,
  with synthetic tracked seeds, a seeding helper, ignored corpora, and a
  findings log. They require a nightly toolchain and are not part of CI.

### Added
- A predeclared beat-path tightness grid and separate 50-take validation,
  including all per-recording results and input hashes. Automatic tracking
  remains experimental and is not enabled in the CLI.
- A reproducible 50-recording GuitarSet rhythm diagnostic, with separate
  reference-tempo and uncalibrated autocorrelation baselines and input hashes.
  Results expose substantial beat-path errors; no CLI tracker was enabled.
- Experimental tempo-conditioned beat-path search with synthetic, exhaustive
  and small real-audio evaluations. It is not a complete automatic tracker and
  is not wired to CLI output.
- An experimental autocorrelation evidence API for onset envelopes, returning
  competing 40–240 BPM lag peaks without claiming beat or tempo detection.
- An experimental, bounded spectral-flux onset-envelope library API with an
  independent NumPy FFT check. This is not beat or onset-event detection and
  does not affect the CLI's tempo behavior.
- Transcription output preflight checks existing file identities to reject
  hard-link aliases to inputs or other outputs, as well as canonical paths.
  This is not a defense against concurrent filesystem changes.
- CLI `--beat-times` accepts a bounded JSON list of supplied quarter-beat times
  for variable-tempo MIDI. It does not perform audio beat or meter detection;
  it assumes 4/4 and does not support JSONL or notation exports.
- MIDI now indexes validated tempo maps for logarithmic-time timestamp lookup;
  collapsed tempo-change ticks and unrepresentable timestamps are rejected.
  Added a synthetic variable-tempo fixture and independent Mido timing check.
- Added validated piecewise tempo timelines and conversion from explicitly
  supplied quarter-beat timestamps; this utility does not detect beats.
- Support an explicit constant `--bpm` (integer 4–255) for MIDI and notation
  output. Default remains 120; this is not automatic beat detection. MIDI
  keeps raw note timing, while notation uses its existing 1/32 grid.
- Documented an Ardour 8.12 synthetic bend import/playback smoke check and
  independently checked the saved MIDI source. Wheel events survived, but the
  pitch-sensitivity RPN was absent; other synthesizers may use a different
  range. This is not a general DAW round-trip guarantee.
- Added a synthetic MIDI dynamics playback sanity check using FluidSynth,
  Mido and an equal-velocity negative control. This does not calibrate the
  experimental dynamics estimator or validate real guitar.
- Added `--partial-fingering` to audio-manifest evaluation with per-recording
  omission counts and explicit scoring caveats. Official-weight evaluation of
  this mode is still pending.
- Reran the 360-file GuitarSet oracle-note evaluation with partial recovery;
  this measures reference-note fingering, not audio transcription or v3.
- Add opt-in `--partial-report` recovery: omitted notes are listed in a JSON
  report and warned about; all outputs, including MIDI, are incomplete when
  any notes are omitted. Strict failure remains the default.
- The GuitarSet oracle report now scores an explicit span-four partial retry
  on strict-search failures, counting skipped notes as incorrect.
- Added an experimental partial fingering library API that explicitly reports
  unassigned input indices instead of failing an entire track. A bounded beam
  minimizes skipped notes before ergonomic cost; it may not find a global
  optimum. Strict assignment remains unchanged by default.
- Added a deterministic exhaustive-search oracle for short fingering cases,
  checking score optimality and constraints for a wide beam, with varied
  overlaps, capo and span; this does not measure musical quality.
- Added a synthetic MIDI bend/release playback sanity check with FluidSynth,
  a test-only soundfont, an independent spectral pitch estimate and a flat
  negative control. This is not a DAW round-trip or real-guitar bend test.
- Test the experimental Rust pitch tracker and bend suggestions on that
  synthesized waveform; loosen the endpoint tolerance from 25 to 30 cents.
  Real-recording bend accuracy remains unvalidated.
- GuitarSet IR and audio reports include one-to-one note-level string/fret
  precision, recall and F1 with onset and optional offset matching. The scorer
  is checked against an independent brute-force Python implementation.
- Release workflow now tests and smoke-runs each target, uses the supported
  macOS Intel runner, validates archive contents and produces checksums for
  both manual dry runs and tagged releases. Cross-platform results must still
  be verified on GitHub Actions.
- IR and end-to-end GuitarSet reports now also include note-level pitch/onset
  and pitch/onset/offset precision, recall and F1 with one-to-one matching.
- End-to-end evaluation reports include SHA-256 identifiers for the manifest
  and local weights. Audio manifests can provide hashes for each audio/JAMS
  pair, verified before inference; the preparation helper supplies both.
  Hashes do not establish dataset authenticity or license authorization.
- Added an offline GuitarSet manifest preparation utility that pairs local
  annotations and mono-mic WAVs, defaults to requiring 360 files, and rejects
  missing audio and accidental input overwrites. Subsets require an explicit
  flag; file count alone does not verify corpus identity or held-out status.
- Framewise reports include the Wiggins–Kim TDR-style ratio, with explicit
  unison and protocol caveats, checked against an independent brute-force
  Python frame-set oracle in CI.
- Added local-weight audio-manifest end-to-end GuitarSet evaluation, reusing
  one loaded model with deterministic guitar-masked strict-EOS inference and
  default standard-tuning fingering. Failed recordings remain false negatives,
  are explicitly reported and cause a nonzero exit. Optional Nightly `run_eval`
  now targets one five-second development excerpt; the official checkpoint run
  and full-corpus/baseline comparison remain unverified.
- Added strict JSON-manifest aggregation for multiple precomputed GuitarSet IR
  evaluations, including per-recording rows and micro-aggregated frame counts.
  It does not run inference and refuses missing, malformed or duplicate inputs.
- Score one emitted standard-tuning/capo-zero TabDocument against matching
  GuitarSet JAMS using fixed-pitch 10 ms framewise multi-pitch and tab-position
  precision/recall/F1 and a pitch-matched wrong-string rate. Requires an
  explicit evaluation duration and does not claim published TabCNN TDR or a
  corpus or tab-labeler-v3 comparison.
- MIDI now renders supplied bend contours as bounded-rate linear pitch-wheel
  ramps rather than three abrupt steps. An independent mido parser verifies
  ramps, RPN range, note timing, and velocity; no audio playback validation.
- Added a licensed five-second GuitarSet pitch-contour fixture and a real-audio
  F0 regression on one isolated reference note. It is a small sanity check,
  not corpus-level accuracy or a labeled-bend benchmark.
- Added a conservative, opt-in library API for tentative bends on isolated
  notes. It abstains on overlap, short notes, unvoiced frames, extreme pitch
  deviations and ambiguous contours; it has synthetic tests only and is not
  wired into CLI or transcription results.
- Experimental reusable monophonic F0 frame tracking for note-bounded regions,
  sharing the tuning detector's FFT-accelerated YIN-style estimator. Tests use
  synthetic steady tones, a frequency sweep, silence, noise and window bounds.
  No audio-to-bend inference or polyphonic isolation is claimed.
- Added an opt-in official-small CLI-to-GP5 test comparing independently
  reconstructed quantized notes against the emitted IR. The checker is covered
  by model-free tied-note fixtures in CI; the gated real-checkpoint test has
  not been run in the current environment and requires an authorized weight.
- Tested a synthetic MusicXML fixture with MuseScore 3.2.3. Pitches and ties
  survived re-export, but some string/fret assignments changed and capo and
  palm-mute information was lost; MusicXML compatibility remains experimental.
- Silent/empty transcription results now export as one empty guitar track in
  all four formats, with an explicit warning. A synthetic-model CLI test
  verifies the pipeline and independent GP5 and XML checks run in CI.
- CLI mixed outputs now include experimental MusicXML and ASCII tab alongside
  MIDI and GP5. All requested formats are rendered before any output write;
  explicit format and filename extensions must agree.
- An experimental MusicXML 3.1 library exporter with standard and tab staves,
  validated in CI against the official schema and read independently with
  music21. It uses the restricted notation planner, rejects bends, and does
  not yet encode note velocities or guarantee single-voice playback.
- An experimental ASCII tab library exporter with bar-aligned strings, beat
  durations, ties, dead notes, palm muting and supported bend markings. It uses
  the restricted GP5 plan; velocities are omitted and output is quantized.
- Manually opened and GP5-re-exported two generated fixtures in TuxGuitar
  1.6.6; independent comparisons cover musical fields and bend points. The
  app check exposed and fixed a half-height bend mapping that the parser
  alone did not detect. Audio playback was unavailable in this environment.
- GP5 can encode explicitly supplied bend, prebend-release, and bend-release
  contours on a single unsplit beat, with positive 100/200-cent heights and an
  interior peak on the 1/12-note bend grid. Unsupported contours are errors.
  Peak timing follows the quantized note; GP5 bend playback may differ from
  MIDI pitch-wheel interpolation. No production bend inference yet.
- GP5 writes explicit dead-note and palm-mute marks from the library IR,
  including palm-mute marks on generated ties. These are not inferred from
  audio; unsupported techniques and bends remain errors. Independent
  PyGuitarPro tests check that palm mutes are not miswritten as staccato.
- GP5 titles now preserve accented Western European characters using the
  Windows-1252 code page. Unrepresentable or oversized titles are rejected
  without overwriting existing files; independent parser tests cover round-trip.
- **M6 experimental library encoder:** GP5 5.00 encoding for the restricted
  4/4 constant-tempo notation plan, using a patched MIT `guitarpro` 0.1.0
  fork. An independent PyGuitarPro test checks multi-track notes, durations,
  ties, string/fret, capo, tuning and GP5 dynamic levels. The export dispatcher
  and CLI can write MIDI plus GP5 together; changing tempo/meter, eight strings,
  unsupported bends, most techniques and colliding quantized notes remain unsupported.
  Limited TuxGuitar fixture checks; Guitar Pro itself is not yet checked.
- **M6 groundwork:** a deterministic 4/4, constant-integer-tempo rhythmic
  planning API on a 1/32 grid. It creates rests and tied note segments across
  attacks and bar lines, reports timing displacement, and rejects unsupported
  techniques, source ties, eight-string tuning and same-string quantization
  collisions rather than dropping them. The planner is separate from the GP5 writer.
- **M2 complete: mel frontend with SACRED PARITY GATE GREEN** — exact port of
  upstream `modules/mel_spectrogram.py` + `MelSpectrogramConditioner`
  (HTK fbanks, periodic Hann, reflect-pad STFT, **power=1.0**, `log(mel+1e-6)`,
  n_fft 2048 / hop 160 / 512 mels @ 16 kHz). Verified against the verbatim
  upstream MIT reference (torch float64 goldens): max rel err ≈ 4e-12 on
  tone/chords/mix fixtures (`tests/mel_parity.rs`, regenerate with
  `tools/gen_mel_fixtures.py`). Tolerance is mel-space scaled `1e-4·max(1,|b|)`:
  fp32 torch noise on unnormalized magnitudes (~1e3) forbids a literal 1e-4
  abs bound; goldens are therefore generated in double precision.
- Audio ingest (M2a): symphonia decode (wav/mp3/flac/ogg/m4a), mono mean,
  rubato windowed-sinc resample to 16 kHz. `load_model_input` matches upstream
  `load_audio` semantics EXACTLY (no DC removal / normalization — the released
  models were trained with `normalize_audio=False`; deviation from PROMPT §6.1
  noted deliberately, parity rule §10.2 wins).
- Workspace scaffold: `mtab-core`, `mtab-audio`, `mtab-guitar`, `mtab-export`, `mtab-cli`.
- `TabDocument` IR with JSON round-trip (`--emit-ir` support): tracks, fretted
  notes with dynamics/bends/techniques, tempo map, chord symbols (PROMPT.md §5).
- Tuning library (standard, half-step-down, drop-D/C, DADGAD, open G/D/E,
  7/8-string) + capo type (PROMPT.md §6.5).
- Model types: `ModelSize` (small/medium/large with official shapes),
  `ModelSource` (local safetensors / HF repo / URL), typed error vocabulary
  with actionable hints (PROMPT.md §6.4).
- CI/CD workflows (`ci.yml`, `release.yml`, `nightly.yml`) and the master
  build prompt (`PROMPT.md`).
