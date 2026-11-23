#!/usr/bin/env python3
"""External Beat This! Rust diagnostic on the fixed GuitarSet selections.

See benches/m4-beat-this-protocol.md. This is an offline development tool,
not an application runtime dependency, and does not download data or models.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np
from evaluate_guitarset_beat_paths import ANNOTATION_MD5, digest, metadata, metrics, select, summary

BEAT_SHA = 'a5f8d39d989f31859454ba27afe61c5317ca95e4d9373e6853e5361b8937172f'
MEL_SHA = 'fdd59e65c515331308e4c8841edf99972deca646bdf6197744c2a5b7755e3de9'


def parse_estimates(data, duration):
    """Validate and extract unmodified beat/downbeat event times."""
    result = json.loads(data)
    beats, downbeats = [], []
    for event in result['beats']:
        t = float(event['time'])
        if not math.isfinite(t) or t < 0 or t > duration or (beats and t <= beats[-1]):
            raise ValueError('beat times must be finite, sorted, unique and within audio')
        if not isinstance(event['downbeat'], bool):
            raise ValueError('downbeat must be boolean')
        beats.append(t)
        if event['downbeat']:
            downbeats.append(t)
    return tuple(np.array([t for t in seq if t >= 5], dtype=np.float64)
                 for seq in (beats, downbeats))


def downbeat_reference(data, duration):
    annotations = json.loads(data)['annotations']
    beat_data = next(a['data'] for a in annotations if a['namespace'] == 'beat_position')
    times = np.array([b['time'] for b in beat_data
                      if b['value']['position'] == 1 and 5 <= b['time'] < duration], dtype=np.float64)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('invalid or empty downbeat reference')
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('annotation-zip', 'audio-dir', 'executable', 'beat-model', 'mel-model'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--take-index', choices=(0, 1), type=int, required=True)
    args = parser.parse_args()
    if digest(args.annotation_zip, 'md5') != ANNOTATION_MD5:
        raise SystemExit('wrong annotation archive MD5')
    if digest(args.beat_model, 'sha256') != BEAT_SHA or digest(args.mel_model, 'sha256') != MEL_SHA:
        raise SystemExit('model SHA-256 mismatch')
    with zipfile.ZipFile(args.annotation_zip) as archive:
        selected = select(archive.namelist(), args.take_index)
        missing = [n for n in selected if not (args.audio_dir / (n[:-5] + '_mic.wav')).is_file()]
        if missing:
            raise SystemExit(f'missing {len(missing)} WAVs; first: {missing[0]}')
        rows = []
        for i, name in enumerate(selected, 1):
            audio = args.audio_dir / (name[:-5] + '_mic.wav')
            with wave.open(str(audio), 'rb') as wav:
                if wav.getnchannels() != 1 or wav.getframerate() != 44100 or wav.getsampwidth() != 2:
                    raise ValueError(f'{audio}: expected mono 44.1 kHz PCM16')
                duration = wav.getnframes() / wav.getframerate()
            source = archive.read(name)
            _, beat_ref = metadata(source, duration)
            downbeat_ref = downbeat_reference(source, duration)
            output = subprocess.run([str(args.executable), str(audio), '--model', str(args.beat_model),
                                     '--mel-model', str(args.mel_model)], text=True, capture_output=True,
                                    check=True, timeout=180).stdout
            beats, downbeats = parse_estimates(output, duration)
            rows.append({'name': name[:-5], 'beats': metrics(beat_ref, beats),
                         'downbeats': metrics(downbeat_ref, downbeats),
                         'audio_sha': digest(audio, 'sha256'), 'jams_sha': hashlib.sha256(source).hexdigest()})
            print(f'{i}/50 {name}', file=sys.stderr, flush=True)
    print('# Beat This! Rust small-model GuitarSet diagnostic\n')
    print(f'Take index {args.take_index}; same 50-recording selection and 5 s trim as the M4 beat-path diagnostic; '
          'mir_eval 0.8.2 one-to-one ±70 ms; no threshold tuning, filtering or tempo supplied.')
    print(f'\nBeat model SHA-256 `{BEAT_SHA}`; mel model SHA-256 `{MEL_SHA}`; '
          f'executable SHA-256 `{digest(args.executable, "sha256")}`; annotation archive MD5 `{ANNOTATION_MD5}`.\n')
    print('| Mode | TP | FP | FN | Micro F1 | Mean per-recording F1 |\n|---|---:|---:|---:|---:|---:|')
    for label, key in [('Beats', 'beats'), ('Downbeats', 'downbeats')]:
        tp, fp, fn, micro, macro = summary(rows, key)
        print(f'| {label} | {tp} | {fp} | {fn} | {micro:.4f} | {macro:.4f} |')
    print('\n| Recording | Beat TP/FP/FN | Beat F1 | Downbeat TP/FP/FN | Downbeat F1 | WAV SHA-256 | JAMS SHA-256 |')
    print('|---|---|---:|---|---:|---|---|')
    for r in rows:
        b, d = r['beats'], r['downbeats']
        print(f"| {r['name']} | {b[0]}/{b[1]}/{b[2]} | {b[3]:.4f} | {d[0]}/{d[1]}/{d[2]} | {d[3]:.4f} | `{r['audio_sha']}` | `{r['jams_sha']}` |")


if __name__ == '__main__':
    main()
