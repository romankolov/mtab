# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- **M6 experimental library encoder:** GP5 5.00 encoding for the restricted
  4/4 constant-tempo notation plan, using a patched MIT `guitarpro` 0.1.0
  fork. An independent PyGuitarPro test checks multi-track notes, durations,
  ties, string/fret, capo, tuning and GP5 dynamic levels. The export dispatcher
  and CLI can write MIDI plus GP5 together; effects, changing tempo/meter,
  eight strings and colliding quantized notes remain unsupported. Not yet manually checked in a notation program.
- **M6 groundwork:** a deterministic 4/4, constant-integer-tempo rhythmic
  planning API on a 1/32 grid. It creates rests and tied note segments across
  attacks and bar lines, reports timing displacement, and rejects unsupported
  techniques, source ties, eight-string tuning and same-string quantization
  collisions rather than dropping them. This is not yet a GP5 file writer.
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
