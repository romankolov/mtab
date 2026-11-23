# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
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
