#!/usr/bin/env python3
"""Independent NumPy rFFT comparison for the documented onset formula.

Generates deterministic PCM16 inputs, exercises the Rust audio decoder and
onset example, and compares every frame with a separately computed reference.
This tests implementation equivalence, not onset or beat detection quality.
"""
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np


def reference(samples):
    if len(samples) < 1024:
        return np.empty((0, 2))
    x = np.asarray(samples, dtype=np.float64) / 32768.0
    frames = np.lib.stride_tricks.sliding_window_view(x, 1024)[::160]
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(1024) / 1024)
    feature = np.log1p(1000.0 * np.abs(np.fft.rfft(frames * window, axis=1)) / 1024)
    strength = np.zeros(len(frames))
    strength[1:] = np.maximum(0.0, np.diff(feature, axis=0)).mean(axis=1)
    times = (np.arange(len(frames)) * 160 + 512) / 16000
    return np.column_stack((times, strength))


def main():
    executable = Path(sys.argv[1]).resolve()
    rng = np.random.default_rng(67392)
    n = 16000
    t = np.arange(n) / 16000
    tone = np.rint(12000 * np.sin(2 * np.pi * 437 * t)).astype(np.int16)
    pulse = np.zeros(n, dtype=np.int16)
    pulse[[1024, 5000, 8000, 15000]] = [32767, -25000, 12345, 27000]
    gate = np.where(((np.arange(n) // 1333) % 2) == 0, tone, 0).astype(np.int16)
    noise = rng.integers(-32768, 32768, n, dtype=np.int16)
    low = rng.integers(-3, 4, n, dtype=np.int16)
    fixtures = [np.zeros(100, dtype=np.int16), np.zeros(n, dtype=np.int16), tone, pulse, gate, noise, low]
    # The checked-in GuitarSet excerpt is 44.1 kHz. Interpolate it to PCM16 at
    # 16 kHz solely to supply an independent real-audio signal for formula
    # parity. This is not a quality evaluation of the audio resampler.
    root = Path(__file__).resolve().parent.parent
    original = root / 'crates/mtab-cli/tests/fixtures/guitarset/guitarset_solo_5s.wav'
    with wave.open(str(original), 'rb') as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 44100)
        raw = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2')
    positions = np.arange(len(raw) * 16000 // 44100) * (44100 / 16000)
    guitar = np.rint(np.interp(positions, np.arange(len(raw)), raw)).astype(np.int16)
    fixtures.append(guitar)
    max_error = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for i, samples in enumerate(fixtures):
            path = Path(tmp) / f"{i}.wav"
            with wave.open(str(path), 'wb') as f:
                f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                f.writeframes(samples.astype('<i2').tobytes())
            process = subprocess.run([str(executable), str(path)], check=True, text=True,
                                     capture_output=True)
            rows = [tuple(map(float, line.split(','))) for line in process.stdout.splitlines()]
            got = np.asarray(rows, dtype=np.float64).reshape((-1, 2))
            expected = reference(samples)
            assert got.shape == expected.shape, (i, got.shape, expected.shape)
            if len(got):
                assert np.allclose(got[:, 0], expected[:, 0], atol=1e-15, rtol=0)
                error = float(np.max(np.abs(got[:, 1] - expected[:, 1])))
                max_error = max(max_error, error)
                assert error < 1e-12, (i, error)
    print(f'NumPy rFFT parity: 7 synthetic fixtures and one resampled GuitarSet excerpt, maximum absolute strength error {max_error:.3g}')


if __name__ == '__main__':
    main()
