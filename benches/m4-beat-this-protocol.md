# Beat This! Rust small-model diagnostic protocol

This protocol was committed before aggregate results were examined. We inspected
one second-take file as a smoke test, so this is **not blinded validation**.

Evaluate the independent MIT-licensed [Beat This! Rust 1.1.0 implementation](https://github.com/danigb/beat-this-rs/tree/1ae768e78f1ad83b0ed3886241dc29ffde853c40)
at commit `1ae768e78f1ad83b0ed3886241dc29ffde853c40`, release build,
with its committed small ONNX beat model and mel model. These are external
research assets: no model or third-party code is being bundled into mtab.
The upstream project attributes its model to Beat This! `small1` (verify
original-checkpoint parity before adopting it). Model SHA-256:

- beat: `a5f8d39d989f31859454ba27afe61c5317ca95e4d9373e6853e5361b8937172f`
- mel: `fdd59e65c515331308e4c8841edf99972deca646bdf6197744c2a5b7755e3de9`

Run its default CPU inference and minimal postprocessing, without tuning,
then compare output quarter-beat and downbeat events separately to the fixed
50 development and 50 second-take validation selections used in
[m4-guitarset-beat-path.md](m4-guitarset-beat-path.md). Both inputs and
estimates are trimmed to events at 5.0 s or later, reference events at or
beyond audio duration are excluded; event matches use mir_eval 0.8.2 at
±70 ms, with TP/FP/FN, micro F1 and mean per-recording F1. Downbeats are
JAMS `beat_position` events with position 1; no meter accuracy claim follows.
Report every recording and input hash. Compare beat F1 with the earlier
fixed tightness-30 baseline, but do not tune or choose any parameters here.

These are click-derived annotations; same performers are in both selections,
GuitarSet informed earlier project work, and upstream model training data may
include this corpus. These scores are not a held-out generalization estimate.
This analysis alone does not integrate automatic tracking. The Rust port
requires Rust 1.89, incompatible with the project's Rust 1.75 MSRV; deployment,
model provenance/parity, licensing attribution, runtime memory, and behavior
on pickups, tempo changes, silence, offbeats and non-guitar mixes must be
resolved before considering adoption.

```sh
python tools/evaluate_beat_this_guitarset.py \
  --annotation-zip /path/annotation.zip --audio-dir /path/audio \
  --executable /path/beat-this --beat-model /path/beat_this_small.onnx \
  --mel-model /path/mel_spectrogram.onnx --take-index 0 > development.md
# Repeat with --take-index 1 > validation.md
```
