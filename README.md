# mtab — muscriptor-tab

[![CI](https://github.com/romankolov/mtab/actions/workflows/ci.yml/badge.svg)](https://github.com/romankolov/mtab/actions/workflows/ci.yml)
[![Release](https://github.com/romankolov/mtab/actions/workflows/release.yml/badge.svg)](https://github.com/romankolov/mtab/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-green.svg)](LICENSE)

**Audio → guitar tablature in pure Rust**, powered by [MuScriptor](https://github.com/muscriptor/muscriptor)
(Kyutai + Mirelo). Point it at any audio file plus a MuScriptor weights file
(`small` / `medium` / `large`) and get **MIDI + Guitar Pro 5** tablature out:
tuning, capo, string/fret assignment, rhythm, bends, dead notes.

> **Status: scaffold.** The repo is bootstrapped with a minimal green workspace,
> CI/CD workflows, and the master build prompt that drives development
> ([PROMPT.md](PROMPT.md)). Milestones M1–M8 are landing incrementally.

## User contract

The only things you provide:

1. **Model size** — `small` (103M, CPU), `medium` (307M, default), `large` (1.4B, max accuracy)
2. **Weights** — a local `model.safetensors` (or `hf://` / URL reference)

Everything else is built in: decoding, features, inference, tuning/capo
detection, fretting, quantization, techniques, export.

```console
$ mtab transcribe song.wav --size medium \
    --weights ./weights/muscriptor-medium.safetensors \
    -o song.mid,song.gp5
```

## Weights & licensing

- **Code:** [MIT](LICENSE) — see [NOTICE](NOTICE) for third-party components.
- **MuScriptor weights:** **CC BY-NC 4.0 (non-commercial), gated on
  [Hugging Face](https://huggingface.co/MuScriptor)**. They are **not** included
  in this repo or in release artifacts. Accept the license on the model page,
  authenticate (`hf auth login` / `HF_TOKEN`), and pass your own weights file.

## Development

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
