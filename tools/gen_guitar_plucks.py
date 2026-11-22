#!/usr/bin/env python3
"""Regenerate the synthetic 16 kHz plucked-string model-validation fixture.

Requires numpy. Produces mono f32, derived from quantized signed 16-bit audio so
that the same waveform can also be passed to the CLI in a PCM WAV container.
"""
from pathlib import Path
import numpy as np


def main():
    sr = 16000
    output = np.zeros(5 * sr, dtype=np.float64)
    rng = np.random.default_rng(770)
    notes = [(40, 0.05), (47, 0.55), (52, 1.05), (55, 1.55),
             (59, 2.05), (64, 2.55), (52, 3.05), (47, 3.55)]
    for midi, start in notes:
        hz = 440 * 2 ** ((midi - 69) / 12)
        length = round(sr / hz)
        ring = rng.uniform(-1, 1, length)
        ring -= ring.mean()
        begin = round(start * sr)
        count = min(round(1.5 * sr), len(output) - begin)
        for i in range(count):
            index = i % length
            value = ring[index]
            ring[index] = 0.997 * (value + ring[(index + 1) % length]) * 0.5
            output[begin + i] += value
    output = output / max(abs(output)) * 0.9
    pcm = (np.clip(output, -1, 1) * 32767).astype("<i2")
    waveform = (pcm.astype(np.float32) / 32768).astype("<f4")
    path = Path(__file__).resolve().parent.parent / "crates/mtab-core/tests/fixtures/model/guitar_plucks.f32"
    waveform.tofile(path)
    print(f"Wrote {len(waveform)} samples to {path}")


if __name__ == "__main__":
    main()
