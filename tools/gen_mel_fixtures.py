#!/usr/bin/env python3
"""Generate mel parity goldens using the UPSTREAM MIT mel implementation
(tools/upstream/mel_spectrogram_mit.py, verbatim copy of
muscriptor/modules/mel_spectrogram.py).

Mirrors `MelSpectrogramConditioner._mel_embedding` with the released model
config: sample_rate=16000, n_fft=2048, hop=160 (frame_rate=100), n_mels=512,
power=1.0, center=True, pad_mode="reflect", log(mel + 1e-6), normalize_audio
OFF (upstream does not normalize; ingest = mono mean + resample only).

Fixtures are generated at 16 kHz natively so they exercise the mel frontend
in isolation (resampler differences are tested separately).

Usage:
    python3 tools/gen_mel_fixtures.py [--out crates/mtab-audio/tests/fixtures]

Requires: torch (CPU is fine), numpy.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "upstream"))
from mel_spectrogram_mit import _MelSpectrogram  # noqa: E402  (MIT upstream)

SR = 16000
N_FFT = 2048
HOP = 160
N_MELS = 512
POWER = 1.0
EPS = 1e-6


def write_f32(path: str, arr: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def signals() -> dict[str, np.ndarray]:
    t = np.arange(2 * SR) / SR  # 2 s
    # 1) clean tone: 440 Hz sine at 0.5 amplitude
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    # 2) chord stack (E major-ish, guitar-like decay envelopes)
    chord = np.zeros_like(t)
    for f, a in [(82.41, 0.4), (123.47, 0.3), (164.81, 0.25), (207.65, 0.2), (246.94, 0.15), (329.63, 0.1)]:
        chord += a * np.sin(2 * np.pi * f * t) * np.exp(-1.5 * t)
    # 3) messy mix: tone + noise floor + low rumble
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(t.shape[0]) * 0.02
    rumble = 0.1 * np.sin(2 * np.pi * 45.0 * t)
    mix = 0.4 * np.sin(2 * np.pi * 440.0 * t) + noise + rumble
    return {"tone": tone, "chords": chord, "mix": mix}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="crates/mtab-audio/tests/fixtures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Double-precision reference: buffers (window, filterbank) and the STFT
    # all run in float64 so the goldens are precision-clean.
    torch.set_default_dtype(torch.float64)

    transform = _MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=POWER,
        center=True,
        pad_mode="reflect",
    )

    for name, sig in signals().items():
        # Feed the EXACT f32 samples the Rust side reads (stored below),
        # upcast to float64: the reference computation runs in double
        # precision so goldens are free of fp32 FFT noise (~1e-3 abs at
        # large magnitudes); cross-implementation tolerance then only
        # reflects true algorithmic equivalence.
        sig32 = sig.astype(np.float32)
        wav = torch.from_numpy(sig32.copy()).unsqueeze(0).double()  # [1, T]
        with torch.no_grad():
            mel = transform(wav)  # [1, n_mels, T]
        mel = mel[0].transpose(0, 1).contiguous().numpy()  # [T, n_mels]
        log_mel = np.log(mel + EPS)

        write_f32(os.path.join(args.out, f"{name}_input.f32"), sig32)
        write_f32(os.path.join(args.out, f"{name}_mel.f32"), mel)
        write_f32(os.path.join(args.out, f"{name}_logmel.f32"), log_mel)

        manifest = {
            "case": name,
            "sample_rate": SR,
            "n_fft": N_FFT,
            "hop_length": HOP,
            "n_mels": N_MELS,
            "power": POWER,
            "log_scale": True,
            "eps": EPS,
            "input_samples": int(sig.shape[0]),
            "frames": int(mel.shape[0]),
            "generated_by": "tools/gen_mel_fixtures.py (upstream MIT reference)",
        }
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"{name}: {sig.shape[0]} samples -> {mel.shape[0]} frames x {mel.shape[1]} mels")

    print(f"fixtures written to {args.out}")


if __name__ == "__main__":
    main()
