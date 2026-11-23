# Sample gallery

A worked example of the whole `mtab` pipeline on a real recording, driving
every part of the CLI and publishing the result as a browsable page.

The task here is the hard case, not the easy one: the input is a **full mix**,
not an isolated guitar. Stereo sources have to be folded to mono, the guitar has
to be picked out of everything else, and only then does fretting make sense.
Expect the note-level result to be approximate; `--separate` (guitar stem
isolation) exists precisely for this and is the single biggest accuracy win on
mixed material.

## What it produces

`output/index.html` is a gallery of one card per capability. Each card shows the
exact command line, its runtime, its status, and links to the files it wrote.
Nothing is hand-written: `generate_gallery.py` runs the binary and formats the
result, so the page cannot drift from what the tool actually does.

| Demo | What it shows |
| --- | --- |
| Everything at once | MIDI + GP5 + MusicXML + ASCII + IR from one command |
| Greedy, strict EOS | The deterministic default; fails loudly instead of guessing |
| Beam search | Wider search at a multiple of the decode cost |
| Seeded sampling | Reproducible non-greedy output |
| Fixed tuning and capo | Skipping detection for a known drop-D guitar, capo 2 |
| Unquantized timing | Raw model timing for rubato passages |
| Detected tempo | Beat tracking instead of a constant BPM |
| Machine-readable notes | JSON Lines for downstream tooling |
| Determinism | The same command twice must be byte-identical (§6.14.6) |
| Actionable failure | What the CLI says when it cannot help (§6.4) |

## Running it

```sh
# 1. weights: gated, CC BY-NC 4.0. Accept the license and export a token.
export HF_TOKEN=hf_...
python3 samples/showcase/fetch_weights.py --size large --out weights/

# 2. build and run
cargo build --release -p mtab-cli
python3 samples/showcase/generate_gallery.py \
  --binary target/release/mtab \
  --audio samples/showcase/audio/sample.wav \
  --weights weights/muscriptor-large.safetensors \
  --size large --out samples/showcase/output
```

## In GitHub Actions

`.github/workflows/sample.yml` runs the same thing on a hosted runner and
uploads the gallery as an artifact. Add a repository secret **`HF_TOKEN`** from
an account that has accepted the license at
<https://huggingface.co/MuScriptor/muscriptor-large>, then trigger the workflow
by hand (Actions → Sample gallery → Run workflow).

The workflow downloads the weights into `$RUNNER_TEMP` and uploads only the
transcripts. A final step fails the job if any `*.safetensors` file reached the
upload directory, so a weights leak is a red build rather than a silent one.

## Cost, measured rather than guessed

The decode is CPU-only fp32 and autoregressive, so cost grows with the number of
generated tokens, which grows with audio length. Measured on a 2-vCPU container:

- `small` (411 MB): ~43 µs/token for the transformer block at 32-dim scale,
  extrapolating to roughly 0.4 s/token at `large` dimensions — about 200 tokens
  per 5-second chunk at typical density.
- `large` (5.47 GB): needs ~5.5 GB resident for the fp32 weights plus the KV
  cache. Beam 4 multiplies the KV cache by four.
- Measured in CI at `small`: 236 s per pass over the whole 87-second clip, and
  beam 4 costs 1041 s — the gallery decodes once per demo, so the full clip is
  hours of runner time at `large`.

For that reason the workflow decodes a **12-second excerpt** of the same clip.
It is still real audio, real weights and a real decode; it just finishes. The
timeout is 350 minutes and the job runs on `workflow_dispatch` rather than on
every push. Dispatch with `size: small` for the fastest sample.

## Audio

`audio/sample.wav` is the supplied 87-second clip, converted to mono 16 kHz WAV
(2.8 MB). It is committed so the workflow is hermetic: the original upload is
WebM/Opus behind a file-sharing link that expires, and the decoder does not read
WebM directly — it says so and suggests a conversion.

The recording is the user's own upload.
