# Weights: obtaining and converting MuScriptor checkpoints legally

## The short version

- **Code in this repository is MIT.** The MuScriptor inference code it ports is
  also MIT, and the ported files keep their attribution.
- **The model weights are not MIT.** They are **CC BY-NC 4.0
  (non-commercial)** and gated on Hugging Face. They are not distributed with
  `mtab`, not bundled in releases, and not downloaded automatically.
- You must accept the license on the model page yourself and provide the file
  locally.

## Obtain the weights

1. Sign in at Hugging Face and open the MuScriptor model repository.
2. Accept its license (CC BY-NC 4.0). Commercial use requires a separate
   license from Kyutai / Mirelo; nothing here grants one.
3. Download the `model.safetensors` for the size you want (`small`, `medium` or
   `large`). Size must match `--size`, because the shapes come from the size.
4. Authenticate only if the download requires it (`hf auth login`, or set
   `HF_TOKEN` in your own environment).

Never commit the file. `.gitignore` already excludes it.

## Pass it to mtab

```console
# explicit local path (the recommended form)
mtab transcribe song.wav --size small --weights ./muscriptor-small.safetensors

# or a previously downloaded Hugging Face cache entry: the bare size name,
# resolved through refs/main of the Hub cache (no network access)
mtab transcribe song.wav --size small --weights small

# or an explicit Hub repository reference already present in the cache
mtab transcribe song.wav --size small --weights hf://MuScriptor/muscriptor-small
```

### Or let mtab fetch it, explicitly

`mtab download` is the only command that touches the network, and it only runs
when you call it:

```console
export HF_TOKEN=hf_...           # a token whose account accepted the license
mtab download --size small --out ./muscriptor-small.safetensors
```

It reads the token from the variable you name with `--token-env` (default
`HF_TOKEN`), never writes it to disk or prints it, and:

- asks the Hub for the file's published size and LFS SHA-256 first, so a 401/403
  is reported as what it is — the license gate, not a network failure;
- writes to `<out>.part`, resuming a partial download with a `Range` request;
- verifies the SHA-256 before the file is given its real name, and deletes it on
  mismatch — a corrupt or truncated download can never end up looking like
  weights;
- opens the finished file with the same safetensors parser `inspect-weights`
  uses and checks that its parameter count matches `--size`.

Nothing else in `mtab` fetches anything. `transcribe` still requires a local
path or an existing cache entry.

`mtab inspect-weights model.safetensors` prints size, key names and a parameter
count without running anything — useful to confirm the file matches `--size`
before a long run.

`http(s)://` URLs exist in the `ModelSource` enum but remote fetching is not
implemented: the CLI refuses them and tells you to download the file yourself.
`hf://owner/name` is resolved **offline** through the Hub cache (`refs/main` →
snapshot), so it only works for a repository you already downloaded. That is
deliberate — a silent multi-gigabyte download of NC-licensed weights is not
something the tool should do behind your back.

## Converting other checkpoints

`mtab` reads plain safetensors. If you have a PyTorch checkpoint (`.bin`,
`.pt`, `.ckpt`) from a source that permits conversion, export it to
safetensors with your own tooling (`safetensors.torch.save_file`, or
`transformers`' conversion utilities), keeping the parameter names. The loader
accepts the upstream MuScriptor names, including the single-codebook embedding
layout that upstream ships; no renaming table is hidden in the binary beyond
that remap.

Watch for two things after conversion:

- **Shape mismatch** is reported as an error naming the tensor and the expected
  shape. `--size` and the file must agree.
- **Precision.** The reference CPU path computes in `f64` after loading. A
  checkpoint stored in `bf16`/`fp16` is widened on load, so it runs but carries
  its own rounding error; parity against an upstream reference must be run with
  the same checkpoint you convert.

## Auxiliary models

Optional analyses were designed with ONNX auxiliaries in mind — a guitar-stem
separator (Demucs/htdemucs), an F0 tracker (CREPE), and UVR-MDX-Net — but no
auxiliary model is bundled, downloaded or required. `mtab` currently implements
its analyses in pure Rust and documents each one's limits. If you wire up your
own auxiliary model, you are responsible for its license: `NOTICE` item 5 lists
the upstream licenses as MIT (Demucs), and "see upstream" for CREPE and
UVR-MDX-Net, which is intentionally not a claim that they are permissive.

## CI

`nightly.yml` can run a real-corpus evaluation when the repository has an
authorized `HF_TOKEN` secret. Without it the job runs only the checked-in
five-second development excerpt. The secret is used to fetch gated weights for
that optional job and nothing else; see `.github/workflows/nightly.yml`.

## Licenses at a glance

| Component | License | Distributed here? |
| --- | --- | --- |
| mtab code | MIT | yes |
| MuScriptor code (ported) | MIT | yes, with attribution |
| MuScriptor weights | CC BY-NC 4.0, gated | **no** |
| `guitarpro` crate (vendored) | MIT | yes |
| PyGuitarPro | LGPL-3.0 | test-only, not linked |
| GuitarSet excerpt | CC BY 4.0 | yes, with attribution |
| Auxiliary ONNX models | per model | **no** |
