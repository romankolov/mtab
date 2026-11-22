# ULTIMATE PROMPT — Build **muscriptor-tab**: Audio → Guitar Tablature in Pure Rust on Top of MuScriptor

> This file is the master build prompt for the AI coding agent developing
> [`romankolov/mtab`](https://github.com/romankolov/mtab).

---

You are a world-class Rust systems engineer and music-information-retrieval (MIR) engineer. Your job is to design, implement, test, and document **`muscriptor-tab`**: a production-quality, reusable Rust library + CLI that converts **any audio file into accurate guitar tablature**, using the **MuScriptor** transcription model as its note-recognition engine. Everything must be built into the library — **the only thing the end user provides is the model variant (small / medium / large) and its weights file**. Outputs are **MIDI (`.mid`)** and **Guitar Pro (`.gp5`)**, plus bonus MusicXML and ASCII tab.

Work from the official upstream sources and the prior art listed in §2. Port, don't guess: MuScriptor's inference code is MIT-licensed and you must match its preprocessing, tokenization, and decoding exactly (verify with golden tests). Be rigorous, incremental, and always leave the repository compiling with green tests — and **pushed to GitHub after every step** (§11).

---

## 1. Mission & product definition

**One-liner:** `song.wav` + `muscriptor-medium.safetensors` in → `song.gp5` + `song.mid` out, with playable, human-readable guitar tablature — fretting, tuning, capo, rhythm, and techniques included.

**User contract (non-negotiable):**
1. User picks a model size: `small` (103M, CPU-friendly), `medium` (307M, default), or `large` (1.4B, max accuracy).
2. User supplies the weights (a local `.safetensors` file, an `hf://` reference, or an HTTP(S) URL). **Never bundle, ship, or silently auto-download weights** — they are gated on Hugging Face behind a CC BY-NC 4.0 license acceptance (see §10). Optional explicit download helper is allowed when the user passes `--download` and a valid `HF_TOKEN`.
3. Everything else is built-in and automatic by default: decoding, resampling, feature extraction, chunked inference, note assembly, tuning/capo detection, string/fret assignment, rhythm quantization, technique detection, export.

**Deliverables:**
- A Rust workspace with a reusable library crate (`muscriptor-tab`, lib name `mtab-core` and friends) and a CLI binary (`mtab`).
- No Python runtime at build or run time. No Node. Pure Rust + native deps behind cargo features.
- Cross-platform: Linux, macOS (Metal), Windows; x86-64 and aarch64.
- Docs, examples, tests, benchmarks, CI config.
- **GitHub repo kept in sync continuously:** commit + push after every step to `https://github.com/romankolov/mtab` (§11).
- **CI/CD green:** lint + tests on every push, cross-platform release builds on version tags, nightly extended runs (§12).

---

## 2. Ground truth: read these first (do not invent facts)

**MuScriptor (the engine)** — Kyutai + Mirelo, released 2026:
- Repo: `https://github.com/muscriptor/muscriptor` (code: **MIT**). Paper linked there. Online demo `muscriptor.kyutai.org`.
- Architecture: **decoder-only Transformer** that reads a **mel-spectrogram of short audio segments** and autoregressively emits **MT3-style MIDI tokens** (pitch / time-shift / instrument / velocity). Transcription is framed as language modeling.
- Variants (exact shapes — use these for config structs):

  | size | params | layers | dim | HF repo |
  |---|---|---|---|---|
  | `small` | 103M | 14 | 768 | `MuScriptor/muscriptor-small` |
  | `medium` (default) | 307M | 24 | 1024 | `MuScriptor/muscriptor-medium` |
  | `large` | 1.4B | 48 | 1536 | `MuScriptor/muscriptor-large` |

- Quality: large ≈ **Multi F1 48.2** vs YourMT3+ 21.9 (best open-source transcription model at release).
- Key behaviors you must reproduce in Rust:
  - **Instrument conditioning:** instrument tokens can be masked so only guitar-family instruments can appear (upstream CLI `--instruments`, names from `muscriptor list-instruments`; guitar family includes steel/nylon acoustic, clean/distorted/jazz electric, guitar harmonics — **verify exact name strings in upstream code and use those verbatim**).
  - **Chunked generation with "prelude forcing":** notes sustained across a chunk boundary are teacher-forced into the next chunk's prologue so a chunk can't restart them with the wrong instrument. Chunks must be generated in order (batch size 1) when this is on.
  - **Decoding:** temperature sampling (`-t`) or beam search (`--beam-size`).
  - **Tempo/quantization:** beats, time signature and bar lines are detected from audio and written into the MIDI; unsteady tempo gets a follow-performance tempo map (`--fixed-tempo` writes one average tempo).
  - Weights format: **safetensors**, loaded from local path, HF cache (`~/.cache/muscriptor/` upstream; also support `~/.cache/huggingface/`), `hf://` or HTTP(S) URL in upstream. Model selection accepts a bare size keyword.
- Licenses: **code MIT, weights CC BY-NC 4.0 (gated)** — see §10.

**Prior art (study and surpass):**
- `lepusagri/autotab-rust` — native Rust MuScriptor inference via **OpenVINO** (Intel Arc + CPU), native mel preprocessing and MT3 decoding from official safetensors, Demucs guitar separation, tuning detection, GP5 export via the MIT `guitarpro` Rust library, GUI + CLI, `tab-labeler-v3` fingering. Known weaknesses you must fix: **fixed MIDI velocity (no dynamics), no pitch bends, and a heuristic-only fret assignment**.
- `danielyx-z/tabula` — audio → MuScriptor → MIDI → small T5 → tab events → playability pass (string assignment, capo, collision fixing) → audio onset analysis (dead notes, articulations) → `.gp5/.mid/.wav` + ASCII tab. Their finding: **MuScriptor's own built-in tab output "offers very little practical use"** — a serious guitar-specific post-processing layer is what makes tabs good. They quantize time to a 20-tick grid at PPQ 960.
- `arraypress/scribe-muscriptor` — Apple Core ML conversions of the same weights (proof the weights format-convert cleanly; also confirms license terms).
- `Perlence/PyGuitarPro` (Python) — authoritative GP3/GP4/GP5 format reference (it's a port of AlphaTab ← TuxGuitar). Use it in tests to validate your GP5 files parse correctly.
- Guitar tablature evaluation: **TabCNN** (Wiggins & Kim, 2019) defines tablature metrics (multi-pitch accuracy, tablature accuracy, wrong-string rate, TDR). **GuitarSet** (360 six-string acoustic clips with note + string/fret annotations) is the standard eval corpus.

---

## 3. Architecture

Rust workspace (a minimal scaffold already exists in this repo — extend it):

```
mtab/                            # https://github.com/romankolov/mtab
├── Cargo.toml                   # workspace (exists)
├── Cargo.lock                   # committed (exists)
├── PROMPT.md                    # this file
├── .github/workflows/           # ci.yml, release.yml, nightly.yml (exist — keep green)
├── crates/
│   ├── mtab-core/               # model config, safetensors load, inference backends, MT3 decode
│   ├── mtab-audio/              # decode (symphonia), resample, mel/STFT, chunking, onset/f0 tools
│   ├── mtab-guitar/             # tuning/capo detect, string-fret assignment, playability, techniques, chords
│   ├── mtab-export/             # MIDI, GP5, MusicXML, ASCII tab writers
│   └── mtab-cli/                # `mtab` binary (clap)
├── tests/                       # golden parity + end-to-end
├── benches/
└── examples/
```

Fewer crates is acceptable (single lib + bin) if you keep the module boundaries above clean. Everything must be usable as a library — the CLI is a thin shell over the public API.

**Cargo features (keep the core lean):**
- `backend-candle` (default; CPU + `metal`/`cuda` via candle features) — candle natively loads safetensors.
- `backend-openvino` (Intel Arc/NPU; stateful graphs like autotab-rust).
- `backend-ort` (ONNX Runtime, for auxiliary models: separation, f0).
- `separation` (Demucs/htdemucs guitar stem isolation).
- `techniques` (f0-based bends/vibrato, dead-note/articulation classifiers).
- `chords` (chord-symbol recognition).
- `auralize` (fluidsynth check-mix rendering).

---

## 4. End-to-end pipeline (the library's spine)

```
audio in (any format)
  → decode (symphonia: wav/mp3/flac/ogg/m4a) → mono mixdown → resample to model sample rate
  → [optional] guitar stem separation (Demucs ONNX) → use guitar stem; fall back to mix
  → [optional] pre-analysis: tuning & capo detection, tempo/beat grid (onset envelope + autocorrelation/DP)
  → mel-spectrogram (EXACT upstream frontend: n_fft, hop, n_mels, fmin/fmax, log scaling — port from MIT code)
  → chunked autoregressive inference (KV-cache, prelude forcing, in-order chunks)
  → MT3 token stream → note events (pitch, onset, offset, velocity, instrument)
  → filter to guitar-family instruments (token masking at decode time — do this, not just post-filter)
  → note cleanup: merge duplicates, drop sub-threshold blips, heal chunk-boundary splits
  → [techniques] f0 track → pitch bends / vibrato; onset analysis → dead notes, palm mutes, slides, hammer-on/pull-off hints
  → rhythm quantization: snap onsets/offsets to the beat grid (support triplet/16th grids; tempo map if unsteady)
  → string & fret assignment (DP over the whole performance — see §6.6)
  → playability pass (collision fix, stretch/position constraints, capo shifts)
  → TabDocument (in-memory IR)
  → export: .mid (tempo map + bends), .gp5 (full Guitar Pro 5), .musicxml, ASCII tab
```

---

## 5. The in-memory IR (`TabDocument`)

Design this first — every stage reads/writes it.

```rust
pub struct TabDocument {
    pub meta: TrackMeta,           // title, tempo map, time sigs, tuning, capo, frets, source hash
    pub tracks: Vec<GuitarTrack>,  // one per detected/selected guitar
    pub chord_symbols: Vec<ChordSymbol>,
}
pub struct GuitarTrack {
    pub tuning: Tuning,            // 6 (or 7/8) MIDI pitches of open strings
    pub capo: u8,
    pub notes: Vec<TabNote>,
}
pub struct TabNote {
    pub start: f64, pub end: f64,   // seconds
    pub midi: u8,
    pub string: u8, pub fret: u8,   // assigned by §6.6
    pub velocity: u8,               // DYNAMICS — fix autotab-rust's fixed-velocity weakness
    pub bend: Option<PitchBend>,    // cents + curve — fix their no-bends weakness
    pub technique: Vec<Technique>,  // Slide{..}, HammerOn, PullOff, Vibrato{..}, PalmMute, DeadNote, Harmonic, LetRing
    pub tie_from_prev: bool,
}
```

Serialization to/from JSON for debugging (`--emit-ir`) is required.

---

## 6. Detailed requirements

### 6.1 Audio ingest (`mtab-audio`)
- `symphonia` for decoding (feature-gated codecs); mixdown to mono (documented rule: average, with optional mid/side keep for future stereo models); DC removal; peak-normalize to −1 dBFS before features (keep an un-normalized copy for export metadata).
- High-quality resampling to the model sample rate (read it from upstream config; `rubato` or a windowed-sinc implementation). Streaming resample — never load a 2-hour file fully into RAM twice.
- Memory budget: process in bounded windows; support files ≥ 2 hours with < 4 GB RAM on `small`.

### 6.2 Feature extraction
- Port the **exact** mel frontend from upstream Python (MIT): window, hop, FFT size, mel bins, fmin/fmax, log-mel vs dB, pre-emphasis, padding/centering convention.
- Write golden tests: run the upstream Python once on 3 short fixtures (clean tone, distorted power chords, full mix), dump `.npy` mel arrays into `tests/fixtures/`, and assert your Rust output matches to ≤ 1e-4 max abs error (fp32). This parity is the single most important correctness gate.

### 6.3 Model & weights (`mtab-core`)
- `ModelSize::{Small, Medium, Large}` with the exact hyperparams from §2's table; architecture = decoder-only Transformer, MT3 vocab, tied/untied embeddings per upstream. Port layer-by-layer from upstream code; add unit tests comparing hidden states of one transformer block against fixture tensors exported from upstream (allow ≤ 1e-3 fp32 drift).
- `ModelSource::Local(PathBuf) | HfCache { size } | Url(String) | HfRepo(String)`.
- **safetensors only** (never pickle). Map keys faithfully; handle both the original HF layout and re-packed layouts (probe and adapt, error clearly otherwise).
- KV-cache for incremental decode; batch=1 in-order chunking with **prelude forcing** (default on, `--no-prelude-forcing` to disable); chunk length configurable (default: upstream's).
- Decoding: greedy default; `--beam-size N` (beam search) and `--temperature T` (sampling, seeded RNG for determinism — `--seed`). Respect instrument masking.
- Device selection `Device::Auto` → CUDA if available, else Metal (macOS), else CPU with SIMD (AVX2/NEON) threading (`--threads`).
- Precision: fp32 default on CPU; fp16/bf16 fast paths on GPU where numerically safe; document accuracy deltas.

### 6.4 Error handling & UX
- `thiserror` typed errors: `WeightsNotFound`, `LicenseGateRequired` (with instructions), `ShapeMismatch { expected, found }`, `UnsupportedAudio`, `DecodeFailure { chunk_index, source }`.
- Progress: `tracing` + a callback/stream of `ProgressEvent`s (stage, %, ETA) so GUIs can wrap the lib.
- Every CLI failure prints an actionable next step (e.g., how to accept the HF license and pass `HF_TOKEN`).

### 6.5 Tuning & capo detection (`mtab-guitar`)
- Pitch-histogram / chroma analysis against a library of tunings: Standard, Drop D, Drop C, DADGAD, Open G/D/E, Half-step down, 7-string (B), 8-string — configurable extensibility.
- Capo detection 0–12 by best alignment of observed lowest-pitch clusters to the tuning template.
- CLI overrides: `--tuning standard|drop-d|...|auto`, `--capo N|auto`, `--frets N`. Auto by default; if confidence < threshold, emit a warning and default to Standard/0.

### 6.6 String & fret assignment (the "tab brain") — this is where tabs become good
- Global **dynamic-programming/Viterbi assignment** over note sequences (not per-note greedy): minimize a cost =
  `w1·position_shifts + w2·finger_stretch_violations + w3·string_crossing_jumps + w4·impossible_overlaps − w5·open_string_preference − w6·fretboard_position_continuity`
- Constraints: frets 0..=N, per-string pitch range from detected tuning/capo, max finger span (4 frets default, stretch model), no two simultaneous notes on one string, thumb-over heuristics optional.
- Chord-aware: when ≥ 3 notes onset together, try known chord shapes first (barre shapes, power chords, open shapes) before free DP.
- Multi-track: if the mix contains two guitars, produce two tracks (use instrument tokens + source separation to split); same-string collisions across tracks are fine.
- Deliver `tab-labeler` quality ≥ autotab-rust's `tab-labeler-v3` on a fixture set you build from GuitarSet annotations (GuitarSet has ground-truth string/fret labels — use them).

### 6.7 Rhythm & quantization
- Beat/downbeat tracking built-in (onset envelope + tempo autocorrelation + DP beat tracker; no Python). Emit tempo map + time signatures into MIDI/GP5.
- Quantize note starts/ends to a user-set grid (`--grid 1/8|1/16|1/32|triplet...|off`, default 1/16) with a confidence measure; keep unquantized timing in the IR (`start_raw`) and write MIDI note microtiming where GP5 allows (or rely on the tempo map).
- Support rubato: when the beat tracker is unsure, use the follow-performance tempo map (like upstream) rather than forcing a grid.

### 6.8 Techniques & dynamics (`techniques` feature) — surpass both prior works
- **Pitch bends / vibrato:** run a lightweight f0 tracker (CREPE-small via `ort`, or a well-tested pYIN-style autocorrelation port) **on the guitar stem, masked to each note's time span**; convert f0 deviation > 25 cents into MIDI pitch-bend events and GP5 bend objects (quarter/full bends, releases); vibrato = periodic f0 modulation → GP5 vibrato.
- **Dead notes / palm mutes:** percussive onset detector (high spectral flux, low harmonic energy, short decay) → GP5 dead note (X) / palm-mute marks.
- **Slides / hammer-ons / pull-offs:** onset-to-onset interval analysis: overlapping/legato transitions on adjacent strings/frets → slide (with slide target), hammer-on/pull-off.
- **Velocity/dynamics:** map note energy envelope (relative to local RMS) to MIDI velocity 30–127 instead of a constant — fixes autotab-rust's biggest MIDI weakness.
- Everything above must be individually toggleable and must degrade gracefully to plain notes when the feature is off or the model is `small` on CPU.

### 6.9 Chord symbols (`chords` feature)
- Chroma/HPCP template matching per beat → chord symbols (maj, min, 7, m7, sus4, add9, power chords, slash bass) written above the tab in GP5/MusicXML. Fine for the tab's "chord track".

### 6.10 Export (`mtab-export`)
- **MIDI (`.mid`):** via `midly`. SMF type 1; one track per guitar + tempo/signature track. Include program change (24/25/26/27/30/31 per guitar type), pitch bends, sustain. Deterministic bytes for identical input.
- **Guitar Pro 5 (`.gp5`):** via the MIT `guitarpro` Rust crate (native GP5 encode/read, as used by autotab-rust). If any GP5 detail is missing there (bends, slides, dead notes, palm mutes, chord diagrams, capo, 7/8-string), implement the missing read/write parts in `mtab-export` following PyGuitarPro/TuxGuitar as the format reference. **Validate every exported file by parsing it back with PyGuitarPro in the test suite** (a dev-dependency Python call in tests is fine — it is not a runtime dependency).
- **MusicXML 3.1** (tab staff + standard staff) and **ASCII tab** (aligned to bar lines, with rhythm stems optional) — cheap and high value for users.
- Optional `auralize` feature: render a stereo check-mix (left = original, right = synthesized transcription via fluidsynth) exactly like upstream's `--auralize`.

### 6.11 Public API (must look at least this nice)

```rust
use muscriptor_tab::{Transcriber, ModelSize, ModelSource, GuitarConfig, Grid, Tuning, Capo};

let tr = Transcriber::builder()
    .size(ModelSize::Medium)                                  // architecture
    .weights(ModelSource::Local("weights/muscriptor-medium.safetensors".into()))
    .device(Default::default())                               // Auto
    .guitar(GuitarConfig { tuning: Tuning::Auto, capo: Capo::Auto, frets: 22, ..Default::default() })
    .grid(Grid::Auto)                                         // beat-aware quantization
    .build()?;

let tab = tr.transcribe_file("song.wav")?;                    // streaming progress via .progress(cb)
tab.export_midi("song.mid")?;
tab.export_guitar_pro("song.gp5")?;
tab.export_musicxml("song.musicxml")?;
println!("{}", tab.ascii_tab());
```

### 6.12 CLI (clap)

```
mtab transcribe song.wav \
  --size medium --weights ./weights/muscriptor-medium.safetensors \
  --instruments guitar            # mask tokens to guitar family
  --tuning auto --capo auto --frets 22 \
  --separate                      # Demucs guitar stem first
  --techniques --chords \
  --grid 1/16 --beam-size 4 --seed 0 \
  -o song.mid,song.gp5,song.musicxml,song.txt

mtab inspect-weights model.safetensors     # size, keys, param count, license hint
mtab download --size medium --token-env HF_TOKEN   # explicit, user-invoked fetch only
mtab detect-tuning song.wav                  # standalone tuning/capo report
mtab eval --corpus guitarset --report eval.md      # §6.14 harness (also used by nightly.yml)
```

### 6.13 Performance targets (measure in `benches/` + a `benches/report.md`)
- `small`: real-time factor ≤ 0.1 (i.e., ≥ 10× faster than real time) on a modern 8-core CPU; < 2 GB RAM.
- `medium`: ≥ 3× real time CPU / ≥ 15× GPU (fp16); < 4 GB RAM.
- `large`: ≥ 1× real time on a modern discrete GPU (fp16); < 12 GB RAM.
- Cold-start (weights load) excluded from RTF; report it separately.

### 6.14 Testing & evaluation
1. **Parity tests** (mel frontend, transformer block, full token stream on fixtures) vs upstream Python — §6.2.
2. **GP5 round-trip** via PyGuitarPro + open in TuxGuitar/Guitar Pro for manual spot-checks (document the manual step).
3. **GuitarSet eval harness** (`mtab eval`, §6.12): multi-pitch F1, onset F1, **tablature accuracy, wrong-string rate, TDR** (TabCNN metrics). Track numbers for small/medium/large in the README.
4. **End-to-end goldens:** 5 fixtures (clean acoustic, distorted rhythm, lead with bends, capo+drop-D, full band mix) with checked-in expected `.gp5`/IR JSON (regenerate with `UPDATE_GOLDENS=1`).
5. **Fuzz** the safetensors loader and audio decoder (`cargo-fuzz`).
6. Determinism test: same input + seed ⇒ byte-identical MIDI.

### 6.15 Docs & packaging
- `README.md` (quickstart, license table, benchmark table), `docs/architecture.md`, `docs/tuning-detection.md`, `docs/techniques.md`, `docs/weights.md` (how to obtain/convert weights legally).
- `examples/`: `basic.rs`, `custom_guitar.rs`, `streaming.rs`.
- CI (GitHub Actions): see §12 — the three workflows already exist; keep them passing.

---

## 7. "Better for guitar" toolbox (integrate behind features — this is the competitive edge)

Prioritize in this order; each must be optional so the core stays dependency-light:

1. **Guitar stem separation** before transcription (Demucs/htdemucs ONNX via `ort`, or UVR-MDX-Net): isolating the guitar removes bass/drum bleed and is the single biggest accuracy win on full mixes. Use the stem for transcription + f0; keep the original for auralization.
2. **Instrument-token masking** to the guitar family at decode time (upstream capability — free accuracy).
3. **Tuning + capo detection** (§6.5) so fret numbers are correct by construction.
4. **Global DP string/fret assignment** (§6.6) — tabs live or die here; GuitarSet-supervised.
5. **f0-based bends & vibrato** (CREPE-small ONNX or pYIN port) — MuScriptor emits no bends (fixed velocity, flat pitch); guitar players expect bends.
6. **Dead-note / palm-mute / slide / hammer-on / pull-off detection** from onset + spectral analysis (Tabula proved this matters: their tab correctly shows percussive dead notes others miss).
7. **Beat tracking + musical quantization** (built-in) for readable rhythm; respect triplets and shuffle.
8. **Chord-symbol recognition** above the staff.
9. **Auralize check-mix** (fluidsynth) for instant QA by ear.
10. **Eval on GuitarSet + Slakh2100 guitar stems** so every improvement is measured.

If and only if `separation`/`techniques` features need models beyond MuScriptor, prefer ONNX (`ort`) so everything remains one Rust binary; document each auxiliary model's license. Never make the core depend on them.

---

## 8. Implementation plan (follow this order; keep main green; **commit + push after every step**, §11)

- **M1 — Skeleton & IR:** extend the scaffold: `TabDocument`, crate layout, CLI stub, CI green. ✔ compiles.
- **M2 — Audio + features:** decode/resample/mel with golden parity vs upstream (§6.2). ✔ parity tests pass.
- **M3 — Model + decode:** safetensors load, transformer port, MT3 decode, chunking + prelude forcing, instrument masking; parity on token streams. ✔ `mtab transcribe --format jsonl` works on `small`.
- **M4 — MIDI export + tempo map.** ✔ playable MIDI in a DAW.
- **M5 — Guitar brain:** tuning/capo detect, DP string/fret assignment, playability pass; GuitarSet tablature metrics wired up (`mtab eval`).
- **M6 — GP5 + MusicXML + ASCII export**, PyGuitarPro round-trip tests. ✔ file opens in Guitar Pro 5 / TuxGuitar / MuseScore.
- **M7 — Techniques & dynamics** (bends, dead notes, slides, velocities) + chords. ✔ goldens updated.
- **M8 — Separation feature, benchmarks, docs, examples, eval report in README, v0.1.0 tag + release.** ✔ release-quality.

---

## 9. Acceptance criteria (definition of done)

- [ ] `mtab transcribe song.wav --size medium --weights <file> -o song.mid,song.gp5` produces both files from any common audio format on all three OSes.
- [ ] The user provides **only** size + weights; no other configuration required for the happy path.
- [ ] Mel + token parity tests pass vs upstream Python (exact tolerances in §6.2/§6.3).
- [ ] GP5 output parses in PyGuitarPro and opens correctly in Guitar Pro 5 / TuxGuitar / MuseScore; contains tuning, capo, correct string/fret numbers, bends, and at least dead-note/palm-mute marks on the fixture set.
- [ ] GuitarSet eval: tablature accuracy of the full pipeline ≥ autotab-rust's `tab-labeler-v3` baseline and wrong-string rate below the per-note greedy baseline (report numbers).
- [ ] MIDI has real velocity dynamics and pitch bends (not flat/constant).
- [ ] Deterministic with `--seed`; no runtime Python; `cargo test` green; `clippy -D warnings` clean.
- [ ] Every step of development is pushed to `romankolov/mtab`; CI is green on `main`; the `v0.1.0` release builds for all targets in §12.
- [ ] README documents the license split (MIT code / CC BY-NC 4.0 weights) and the weight-acquisition flow.

---

## 10. Hard rules & pitfalls (read carefully)

1. **Licensing:** MuScriptor **code is MIT** (port freely, retain attribution notices), but **weights are CC BY-NC 4.0 and Hugging-Face-gated**. Never commit weights to the repo, never auto-download without explicit user action, never suggest commercial use of the weights without a license. Put a `NOTICE` file listing all third-party code (`guitarpro` crate MIT, PyGuitarPro LGPL-3.0 test-only, any ONNX aux models with their licenses).
2. **Frontend parity is sacred** — a 2-bin mel mismatch destroys transcription quality silently. Golden tests (§6.2) before anything else.
3. **Prelude forcing** is easy to get wrong; test specifically at chunk boundaries with sustained notes (fixture: held power chord spanning two chunks).
4. **Do not over-quantize**: rubato passages must use the tempo-map path, not a forced grid.
5. **Don't trust the model for fretting** — MuScriptor predicts pitch, not string/fret. The DP assigner (§6.6) owns fretting; the model only supplies notes.
6. **Keep aux models optional.** The library must work fully offline with just weights + audio.
7. **Determinism:** seeded sampling, stable iteration order over hash maps, fixed tie-breaking in DP.
8. **Error messages are part of the product** (§6.4) — especially the HF license-gate flow.
9. Prefer well-maintained crates: `symphonia`, `rubato`, `realfft`/`rustfft`, `candle` (or `ort`), `safetensors`, `midly`, `clap`, `thiserror`, `tracing`, `rayon`. The MIT `guitarpro` crate for GP5 (extend it in-tree if needed).
10. When unsure about an upstream detail, **read the upstream source** (MIT) — do not hallucinate tensor shapes or token vocabularies.
11. **Push or it didn't happen** — after every step, commit and push per §11. The local workspace is disposable; GitHub is the source of truth. Never commit credentials (§11.5).

---

## 11. GitHub persistence protocol (workspace space is limited — push after every step)

**Repository:** `https://github.com/romankolov/mtab` (remote `origin`, default branch `main`) — already bootstrapped with a minimal workspace, this prompt, and the CI workflows. The local workspace may be wiped or truncated at any time, so GitHub is the project's persistent storage.

1. **Commit and push after EVERY completed step** — a step = one coherent unit: a milestone task, a fix, a feature slice, a test suite, a doc update. Never hold more than one step of unpushed work. Big step? Commit and push working intermediate slices (`wip:` commits) as you go.
2. **Commit style:** Conventional Commits — `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `build:`, `perf:`, `refactor:`, `chore:`, `wip:` (pushed intermediate slices only). Imperative mood, ≤72-char subject, milestone ref (M1–M8) in the body. Examples: `feat(M3): port MT3 decoder with prelude forcing`, `test(M2): mel golden parity vs upstream`.
3. **Push discipline:** `git push origin main` after each commit (or a short chain of commits within one step). Never force-push `main`. If CI on `main` breaks, fix it with a follow-up commit before continuing other work. A milestone is not done until its push passes CI (§12).
4. **Review before staging:** run `git status` and `git diff --staged`; confirm no secrets, weights, or junk are staged. `git add -A` is safe ONLY because the committed `.gitignore` excludes `target/`, `*.safetensors`, `*.onnx`, `*.pt`, `.env*`, `credentials*` — never weaken that file.
5. **Credentials:** a GitHub PAT with `repo` scope is provided at runtime via the `GH_TOKEN` environment variable (in this workspace it sits in the git-ignored file `.env`). Authenticate with `gh auth login --with-token` + `gh auth setup-git`, or a git credential helper. **NEVER write the token into any tracked file, commit message, workflow YAML, README, or log**; never embed it in a committed remote URL. If a token ever lands in a file being committed: stop, purge it, and tell the user to rotate it immediately.
6. **What NOT to push:** model weights (`*.safetensors` — CC BY-NC 4.0 + size), any credential, generated audio renders, `target/`, any file > 1 MB except an explicitly needed test fixture. **Do push:** code, tests, small fixtures (`tests/fixtures/`, ≤ 1 MB each), docs, workflow changes, `Cargo.lock`, `CHANGELOG.md`.
7. **Remote states:** if the remote is empty, push `main` with `-u`. If the remote already has commits, `git pull --rebase origin main` before pushing. Keep history linear and honest — no history rewriting after push.
8. **Changelog & versions:** maintain `CHANGELOG.md` (Keep a Changelog) from M1 on. Release = semver tag `v0.y.z` on a green `main` commit → `release.yml` builds the release (§12).

---

## 12. CI/CD — GitHub Actions (workflows are pre-committed in `.github/workflows/` — keep them green and extend them)

Three workflows ship with the scaffold. They are part of the product.

| file | triggers | what it does |
|---|---|---|
| `ci.yml` | every push to `main`, every PR | `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` on Linux + macOS + Windows, release-mode smoke build, `cargo doc` |
| `release.yml` | tags `v*`, manual dispatch | release builds for `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-apple-darwin`, `aarch64-apple-darwin`, `x86_64-pc-windows-msvc`; packages tar.gz/zip with README + LICENSE + NOTICE; SHA256SUMS; GitHub Release with artifacts (manual dispatch = dry-run build, no release) |
| `nightly.yml` | weekly cron + manual | extended tests in release mode, `cargo audit` (RustSec), optional GuitarSet eval job (runs only with `run_eval=true` and the `HF_TOKEN` repo secret present; uploads `eval-report` artifact) |

Requirements:
- Update the matrices in the same commit whenever you add crates, binary targets, or cargo features (e.g. `--features separation` build legs, a `mtab-eval` job).
- Never hardcode tokens in YAML. `GITHUB_TOKEN` (automatic) is enough for releases; `HF_TOKEN` is an optional repo secret for gated-weight jobs. Document any new secret in `docs/weights.md`.
- Release artifacts must NEVER contain model weights (CC BY-NC 4.0). The `mtab download` helper fetches weights to the user's machine at runtime only.
- Keep `Cargo.lock` committed; release builds use `--locked`.
- Before pushing any change that touches CI-relevant surface (features, targets, MSRV), run locally what you can (`cargo fmt`, `cargo clippy`, `cargo test`) and rely on `ci.yml` for the rest.

---

**Now begin.** The repository is already bootstrapped: a minimal workspace that builds green, the three CI workflows, `.gitignore`, `LICENSE`, `NOTICE`, and this prompt. Start at M1 — and after **every** step, commit and push per §11.
