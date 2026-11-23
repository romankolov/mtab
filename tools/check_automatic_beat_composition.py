#!/usr/bin/env python3
"""Check the new automatic wrapper against the previously evaluated composition."""
import sys
from pathlib import Path
import numpy as np
from evaluate_guitarset_beat_paths import executable, run, strongest_candidate

examples = Path(sys.argv[1])
for audio in map(Path, sys.argv[2:]):
    bpm = strongest_candidate(examples, audio)
    expected = np.array([float(x) for x in run([executable(examples, 'beat_path_dump'), audio, bpm, 30.0]).splitlines()]) if bpm is not None else np.array([])
    actual = np.array([float(x) for x in run([executable(examples, 'automatic_beat_dump'), audio]).splitlines()])
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
    print(f'Automatic wrapper agrees with evaluated composition: {audio.name}, {len(actual)} beats')
