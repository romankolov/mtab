# Synthetic model parity fixture

`tiny.fixture` is a safetensors-format file containing **randomly initialized,
seeded test tensors**, not a trained MuScriptor checkpoint. Its dimensions are
32 hidden units and two transformer layers; it cannot transcribe audio. It is
included solely so the Rust golden tests run without Python, network access, or
licensed model weights. `wav.f32` is synthetic audio. The remaining files are
reference outputs generated with upstream Python modules at these tiny
settings. No production weights are included.

See `tools/gen_model_fixtures.py` for regeneration requirements and commands.
