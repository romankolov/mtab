#!/usr/bin/env python3
"""Synthetic tempo-conditioned beat-path checks and real excerpt diagnostic.

Calls the Rust development example and scores with mir_eval's 70 ms beat
F-measure. Neither the synthetic cases nor the development excerpt establish
held-out beat-tracking performance. No beat/tempo is estimated by the scorer.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import mir_eval
import numpy as np


def write_wav(path, signal):
    with wave.open(str(path), 'wb') as f:
        f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        f.writeframes(signal.astype('<i2').tobytes())


def detect(executable, path, bpm, tightness=3.0):
    result = subprocess.run([str(executable), str(path), str(bpm), str(tightness)],
                            check=True, capture_output=True, text=True)
    return np.array([float(line) for line in result.stdout.splitlines()])


def score(reference, estimated):
    if not len(reference) or not len(estimated):
        return 0.0
    return mir_eval.beat.f_measure(reference, estimated, f_measure_threshold=0.07)


def click_signal(bpm, *, offbeat=False):
    duration, sr = 8, 16000
    signal = np.zeros(duration * sr, dtype=np.float64)
    interval = 60 / bpm
    reference = np.arange(0.25, duration - 0.2, interval)
    rng = np.random.default_rng(3857)
    for beat in reference:
        for time, amp in [(beat, 0.22)] + ([(beat + interval / 2, 0.9)] if offbeat else []):
            index = round(time * sr)
            if index + 100 <= len(signal):
                signal[index:index + 100] += amp * rng.uniform(-1, 1, 100) * np.linspace(1, 0, 100)
    return (np.clip(signal, -1, 1) * 32767).astype(np.int16), reference


def main():
    executable = Path(sys.argv[1]).resolve()
    root = Path(__file__).resolve().parent.parent
    fixture = root / 'crates/mtab-cli/tests/fixtures/guitarset'
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'test.wav'
        for bpm in [60, 90, 120, 180]:
            samples, reference = click_signal(bpm)
            write_wav(path, samples)
            estimated = detect(executable, path, bpm)
            f1 = score(reference, estimated)
            print(f'synthetic {bpm} BPM: reference={len(reference)} estimated={len(estimated)} beat F1={f1:.4f}')
            assert f1 >= 0.9, (bpm, reference, estimated, f1)
        # A louder offbeat shows that phase can be wrong even with oracle BPM.
        samples, reference = click_signal(120, offbeat=True)
        write_wav(path, samples)
        estimated = detect(executable, path, 120)
        offbeat_f1 = score(reference, estimated)
        print(f'120 BPM louder-offbeat control: reference={len(reference)} estimated={len(estimated)} beat F1={offbeat_f1:.4f}')
        assert offbeat_f1 < 0.5, (reference, estimated, offbeat_f1)
        write_wav(path, np.zeros(8 * 16000, dtype=np.int16))
        assert len(detect(executable, path, 120)) == 0
        single = np.zeros(8 * 16000, dtype=np.int16)
        single[16000] = 32000
        write_wav(path, single)
        assert len(detect(executable, path, 120)) <= 1
    beat_bytes = (fixture / 'guitarset_solo_5s_beats.json').read_bytes()
    assert hashlib.sha256(beat_bytes).hexdigest() == '5deac4e160c4a39a0725dc6754acbab4f191b4ae7f55585b31b112bd35a795a2'
    data = json.loads(beat_bytes)
    reference = np.array([b['time'] for b in data['beats']])
    audio = fixture / 'guitarset_solo_5s.wav'
    for bpm in [129, 6000 / 47, 6000 / 93]:
        estimated = detect(executable, audio, bpm)
        f1 = score(reference, estimated)
        matches = mir_eval.util.match_events(reference, estimated, window=0.07)
        print(f'GuitarSet 5s target={bpm:.2f} BPM: TP={len(matches)} FP={len(estimated)-len(matches)} FN={len(reference)-len(matches)} beat F1={f1:.4f}')


if __name__ == '__main__':
    main()
