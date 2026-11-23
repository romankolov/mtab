# AutoTab v3 oracle-note diagnostic (not end-to-end)

The frozen protocol is [m5-autotab-v3-protocol.md](m5-autotab-v3-protocol.md),
committed before evaluation. `autotab` is the **unmodified**
`lepusagri/autotab-rust` application at commit
`a1eebbebbd74db669a6a5f74079eed907717cfb3`. Its v3-egset model is trained on
GuitarSet and EGSet12, so **this is an in-sample diagnostic, not held-out**.
Ground-truth note pitches and times were supplied through MIDI, so neither
MuScriptor nor audio transcription was evaluated. This result does not satisfy
the end-to-end acceptance criterion, and is not a TabCNN/TDR metric.

| Measure | Result |
| --- | ---: |
| GuitarSet 1.1.0 files successfully evaluated | 360 / 360 |
| Ground-truth notes | 62,476 |
| Notes assigned by AutoTab | 62,476 |
| Notes on extra guitars | 5 (3 files) |
| Correct string, all guitars | 53,530 / 62,476 = **85.68%** |
| Correct string, primary guitar only | 53,528 / 62,476 = **85.68%** |
| Unassigned notes | 0 |
| Preparation, CLI or note-identity failures | 0 |

Per-file results and SHA-256 hashes are in
[autotab-v3-results/guitarset-oracle.json](autotab-v3-results/guitarset-oracle.json).
All source-note pitch, tick and channel identities were checked against the
MIDI written from annotations, and assigned and unassigned note indices
partitioned the source notes. A second complete run with invalid HTTP proxy
settings produced a byte-identical report. The application source checks the
expected model hash and takes its local-cache path; this repeat is not proof of
network isolation against every possible mechanism.

For context, our own independently evaluated **reference-note** solver
reported 68.50% on the same corpus with span-four partial retry
([guitarset-oracle.md](guitarset-oracle.md)). This comparison has confounds:
AutoTab was trained on GuitarSet, uses 0–19 frets and 20 µs MIDI timing,
groups close onsets, permits extra guitars and has its own fallback; mtab uses
original note times and 0–22 frets. Both are in-sample, non-end-to-end
analyses, and neither provides the requested full-pipeline benchmark.

## Reproduction and environment

- Annotation archive: GuitarSet 1.1.0, CC BY 4.0; MD5
  `b39b78e63d3446f2e54ddb7a54df9b10`.
- Model: `cstr/tab-labeler-onnx`, `tab-labeler-v3-egset.onnx`, revision
  `434976baacaf194d99f09c3c784682766bedcf97`, CC BY 4.0; SHA-256
  `93114786c6f4781e7a0b69abd7e93a0d4966873f8e54908f97a1de8e8cf788af`.
- Linux x86-64, Rust 1.92.0, unoptimized `cargo build --locked --bin autotab`
  with `CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0
  CARGO_PROFILE_DEV_CODEGEN_UNITS=64`; executable SHA-256
  `8a03a3210b7fa8780f30191787c85b2b9942c9315c6ca136133c1c32789fba9a`.
- ONNX Runtime 1.24.4 shared library SHA-256
  `2d518e11c7767f9fdc4413bdbeb45acf30db504ff76a82fc19d9238a3f9f6c24`;
  Python 3.13, Mido 1.3.3. AutoTab expects its runtime at
  `$AUTOTAB_HOME/runtime/onnxruntime.dll`; on this Linux run that name was a
  symlink to the ELF shared library. The app selects CPU, one thread.
- Run `python tools/evaluate_autotab_v3_oracle.py --annotations annotation.zip
  --executable /path/to/autotab --autotab-home /path/to/home --output report.json`.
  Supply the separately acquired and hash-verified v3 model under
  `models/tab-labeler-v3-egset.onnx`. No weights or external application code
  are included in this repository.

An earlier ONNX Runtime 1.22.1 test could not select a CPU provider; 1.24.4
worked. The original optimized release build exceeded this environment's
memory constraints, so the unoptimized binary above was used. Performance
comparisons were not attempted.
