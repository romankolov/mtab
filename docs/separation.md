# Optional guitar-stem separation

`mtab-audio` can optionally isolate guitar with the six-source HTDemucs v4 ONNX graph before transcription. Separation is opt-in; the default build does not enable ONNX Runtime, bundle a model, or fetch any model file.

## Validated graph

- **File:** `htdemucs_6s.onnx`
- **SHA-256:** `82cd0479a55b6b4018b948c0bb620975f8cdc75f682f40ad1357d5eb834b438e`
- **Source:** [`smank/htdemucs-onnx`](https://huggingface.co/smank/htdemucs-onnx)
- **License:** MIT (the converted graph derives from Meta Demucs, also MIT)
- **Input:** planar stereo, `[1, 2, 343980]`, 44.1 kHz
- **Output:** `[1, 6, 2, 343980]`; stem order is `drums, bass, other, vocals, guitar, piano`. Guitar is index 4.

This exported graph requires exactly 343,980 samples (7.8 seconds) per inference. The library resamples input to 44.1 kHz, duplicates mono to stereo, processes non-overlapping fixed windows, zero-pads the final window, trims padding, averages the guitar stem's two channels, then resamples it to the transcription model's 16 kHz mono rate. Windows are concatenated without overlap, so audio around a window boundary may have a discontinuity and lose part of an attack or decay. The graph ran at approximately 0.9× real time on the CPU used for validation (about 7.0 seconds per 7.8-second segment).

The model was tested on a real GuitarSet guitar excerpt: 101.6% of the input power landed in the guitar stem (the small excess is output scaling). A guitar-only Karplus–Strong plucked-string test also passes. These are sanity checks, not a guarantee that separation improves every mix or transcription.

## Obtain and verify the model

Download `htdemucs_6s.onnx` yourself from the model source above, review its license and provenance, and verify the checksum before use:

```sh
sha256sum htdemucs_6s.onnx
# 82cd0479a55b6b4018b948c0bb620975f8cdc75f682f40ad1357d5eb834b438e
```

The program never downloads the model. Do not use the four-stem `htdemucs.onnx` for guitar isolation: it has no guitar output. The six-stem graph is about 246 MB.

## CLI

Build the optional feature and pass the local ONNX path explicitly:

```sh
cargo run -p mtab-cli --features separation -- \
  transcribe input.wav --weights /path/to/muscriptor.safetensors \
  --separate /path/to/htdemucs_6s.onnx \
  --outputs output.mid,output.gp5
```

The runtime library is loaded dynamically. Install a compatible ONNX Runtime shared library (API 22 or compatible) and make it discoverable through the platform's normal loader path (`LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH` on macOS, or `PATH` on Windows). `ort` is pinned to `2.0.0-rc.14`; the crate uses its `load-dynamic` mode. Builds without the CLI `separation` feature reject `--separate` with an explanation.

## Rust API

```toml
[dependencies]
mtab-audio = { path = "../crates/mtab-audio", features = ["separation"] }
```

```rust,no_run
use mtab_audio::{load_model_input, separate::Separator};

let audio = load_model_input("input.wav")?;
let mut separator = Separator::new("/path/to/htdemucs_6s.onnx")?;
let guitar = separator.guitar_stem(&audio)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`Separator::new` accepts a local model path only. The six-source stem order and fixed segment size are model-export details, not a promise that arbitrary Demucs ONNX conversions will work with this wrapper.
