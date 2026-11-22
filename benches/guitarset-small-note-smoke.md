# Small checkpoint: one real GuitarSet excerpt (2026-11-22)

This is a **single five-second real-guitar smoke measurement**, not a
representative or held-out corpus evaluation and not a comparison with
`tab-labeler-v3`. It measures note events only, not string/fret accuracy,
framewise multi-pitch F1, TabCNN/TDR, bends, dynamics, or DAW playback.

Input: the first five seconds of GuitarSet v1.1.0
`00_BN1-129-Eb_solo_mic.wav`; the included excerpt and its modified reference
annotations are CC BY 4.0, with full attribution, changes and hashes in the
[fixture README](../crates/mtab-cli/tests/fixtures/guitarset/README.md).
This excerpt was selected during development and is not a held-out test.

Model: official gated MuScriptor small safetensors checkpoint, SHA-256
`bbd482c786b895cf7d8f44185073d951adae2ebb8a66f82ca84cd1f84569549c`;
weights are CC BY-NC 4.0, not included. The run used greedy decoding,
`--instruments guitar`, seed 0, a 512-token cap and `--strict-eos`. It
completed, produced 14 notes and did not hit the token cap. The reference
contains 20 annotated notes.

| Metric | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pitch + onset | 14 | 0 | 6 | 100.00% | 70.00% | 82.35% |
| Pitch + onset + offset | 9 | 5 | 11 | 64.29% | 45.00% | 52.94% |

The matcher uses rounded MIDI reference pitch, one-to-one maximum-cardinality
matching and a 50 ms onset tolerance. Offset matching also requires a
max(50 ms, 20% of reference duration) tolerance. Offsets are clipped at five
seconds. An independent `mir_eval` 0.8.2 check returned the same TP/FP/FN for
both rows. These values are not extrapolatable to the full corpus.

## Reproduce

After accepting the model's gated license and providing the official checkpoint:

```sh
MTAB_SMALL_WEIGHTS=/path/to/model.safetensors MATMUL_NUM_THREADS=2 \
  cargo test --locked --release -p mtab-cli --test small_checkpoint_smoke \
  real_small_guitarset_excerpt_note_evaluation -- --ignored --nocapture
```

Or run the pipeline directly:

```sh
FIXTURE=crates/mtab-cli/tests/fixtures/guitarset
mtab transcribe "$FIXTURE/guitarset_solo_5s.wav" --size small \
  --weights /path/to/model.safetensors --instruments guitar \
  --format jsonl --strict-eos --max-gen-len 512 --seed 0 -o notes.jsonl
mtab eval --corpus guitarset --annotations "$FIXTURE/guitarset_solo_5s.jams" \
  --predictions notes.jsonl --clip-duration 5
# Optional independent development check, not required by mtab:
python -m pip install mir_eval==0.8.2
python tools/check_notes_mir_eval.py "$FIXTURE/guitarset_solo_5s.jams" \
  notes.jsonl --clip-duration 5
```

On GitHub Actions the optional Nightly `run_real_small` job requires an `HF_TOKEN`
repository secret for a Hugging Face account that has accepted the model's gated
conditions. It downloads and verifies the checkpoint inside the ephemeral runner
and uploads test logs only, never weights.
