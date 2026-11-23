#!/usr/bin/env python3
"""Independent NumPy oracle for the documented autocorrelation evidence.

This verifies each discrete lag and peak marker, not tempo or beat accuracy.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np


def oracle(x):
    if len(x) < 300 or not np.any(x):
        return np.empty((0, 4))
    x = x / np.max(x)
    x = x - x.mean()
    if not np.any(x):
        return np.empty((0, 4))
    scores = []
    for lag in range(25, 151):
        a, b = x[:-lag], x[lag:]
        den = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
        scores.append(np.clip(np.sum(a * b) / den, -1, 1) if den else 0.0)
    peaks = [score > 0 and (i == 0 or score > scores[i-1]) and
             (i == len(scores)-1 or score >= scores[i+1])
             for i, score in enumerate(scores)]
    return np.column_stack((np.arange(25, 151), 6000 / np.arange(25, 151), scores, peaks))


def run(executable, x):
    text = ''.join(f'{v:.17g}\n' for v in x)
    result = subprocess.run([str(executable)], input=text, capture_output=True, text=True, check=True)
    rows = [tuple(map(float, line.split(','))) for line in result.stdout.splitlines()]
    return np.asarray(rows).reshape(-1, 4)


def main():
    executable = Path(sys.argv[1]).resolve()
    onset_executable = Path(sys.argv[2]).resolve()
    root = Path(__file__).resolve().parent.parent
    guitar = root / 'crates/mtab-cli/tests/fixtures/guitarset/guitarset_solo_5s.wav'
    output = subprocess.run([str(onset_executable), str(guitar)], capture_output=True, text=True, check=True).stdout
    real = np.array([float(line.split(',')[1]) for line in output.splitlines()])
    rng = np.random.default_rng(63942)
    signal = np.zeros(1000)
    signal[10::50] = 1.0
    cases = [np.array([]), np.zeros(300), np.ones(1000), np.ones(299), signal, real]
    cases.extend(rng.random(int(rng.integers(300, 1200))) for _ in range(60))
    maximum = 0.0
    for index, x in enumerate(cases):
        actual = run(executable, x)
        expected = oracle(x)
        assert actual.shape == expected.shape, (index, actual.shape, expected.shape)
        if len(actual):
            assert np.array_equal(actual[:, 0], expected[:, 0])
            assert np.allclose(actual[:, 1], expected[:, 1], rtol=1e-14, atol=1e-14)
            error = np.max(np.abs(actual[:, 2] - expected[:, 2]))
            maximum = max(maximum, error)
            assert error < 1e-12, (index, error)
            assert np.array_equal(actual[:, 3], expected[:, 3]), index
    results = run(executable, real)
    candidates = sorted(results[results[:, 3] == 1], key=lambda row: (-row[2], row[0]))
    print(f'NumPy periodicity parity: {len(cases)} cases; maximum score error {maximum:.3g}')
    print('GuitarSet 5s top candidates (BPM, correlation):', [(round(p[1], 2), round(p[2], 4)) for p in candidates[:3]])


if __name__ == '__main__':
    main()
