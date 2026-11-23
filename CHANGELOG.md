# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
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
