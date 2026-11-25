# Examples

Three runnable programs that exercise the library the way a caller would.
Each needs a local MuScriptor checkpoint, which is **not** bundled (CC BY-NC
4.0, gated — see [`docs/weights.md`](../docs/weights.md)).

```sh
cargo run --release --example basic -- song.wav ~/weights/muscriptor-small.safetensors
cargo run --release --example custom_guitar -- song.wav ~/weights/muscriptor-small.safetensors
cargo run --release --example streaming -- ~/weights/muscriptor-small.safetensors a.wav b.wav
```

| Example | Shows |
| --- | --- |
| [`basic.rs`](basic.rs) | the whole pipeline with default settings, two exports and ASCII tab |
| [`custom_guitar.rs`](custom_guitar.rs) | a fixed drop-D tuning behind a capo, and the `midi = open + capo + fret` invariant |
| [`streaming.rs`](streaming.rs) | one model held across files, with `Progress` rendered per stage |

All three were run against `muscriptor-small` on a 10-second GuitarSet excerpt.
`basic` produced 75 notes and wrote both files; `custom_guitar` solved the same
audio as drop-D at capo 2 and passed its invariant assertion on every note;
`streaming` loaded the checkpoint in 0.49 s and transcribed two files in 38.3 s
and 35.4 s. They are compiled on every CI run as `[[example]]` targets, so
they cannot rot silently.
