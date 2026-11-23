#!/usr/bin/env python3
"""Offline, fixed-selection GuitarSet beat-path diagnostic (not a benchmark).

Requires externally obtained GuitarSet 1.1.0 annotation.zip and microphone WAVs,
Rust release examples, NumPy and mir_eval 0.8.2. It never downloads or mutates
input data and only writes a Markdown report to stdout. Tracks are selected by
rule without inspecting scores; see benches/m4-guitarset-beat-path.md.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import wave
import zipfile
from pathlib import Path

import mir_eval
import numpy as np

ANNOTATION_MD5 = 'b39b78e63d3446f2e54ddb7a54df9b10'
STYLES = ('BN', 'Funk', 'Jazz', 'Rock', 'SS')


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def select(names):
    names = sorted(n for n in names if n.endswith('.jams'))
    chosen = []
    for player in range(1, 6):
        for style in STYLES:
            for mode in ('comp', 'solo'):
                options = [n for n in names if n.startswith(f'{player:02d}_{style}') and n.endswith(f'_{mode}.jams')]
                if not options:
                    raise ValueError(f'missing player={player}, style={style}, mode={mode}')
                chosen.append(options[0])
    if len(set(chosen)) != 50:
        raise ValueError('selection contains repeated files')
    return chosen


def metadata(data, duration):
    annotation = json.loads(data)
    beats = [a for a in annotation['annotations'] if a['namespace'] == 'beat_position']
    tempos = [a for a in annotation['annotations'] if a['namespace'] == 'tempo']
    if len(beats) != 1 or len(tempos) != 1 or len(tempos[0]['data']) != 1:
        raise ValueError('expected one beat and one constant tempo annotation')
    raw = beats[0]['data']
    if any(b['value']['beat_units'] != 4 for b in raw):
        raise ValueError('reference beats must be quarter-note units')
    reference = np.array([b['time'] for b in raw if 5 <= b['time'] < duration], dtype=np.float64)
    if not len(reference) or not np.isfinite(reference).all() or np.any(np.diff(reference) <= 0):
        raise ValueError('invalid or empty reference beats after five-second trim')
    bpm = float(tempos[0]['data'][0]['value'])
    if not np.isfinite(bpm) or not 40 <= bpm <= 240:
        raise ValueError('reference tempo outside supported 40–240 BPM')
    return bpm, reference


def executable(directory, name):
    return directory / (name + ('.exe' if os.name == 'nt' else ''))


def run(args, *, input_text=None):
    return subprocess.run([str(a) for a in args], input=input_text, text=True,
                          capture_output=True, check=True, timeout=180).stdout


def strongest_candidate(examples, audio):
    csv = run([executable(examples, 'onset_dump'), audio])
    values = ''.join(line.split(',')[1] + '\n' for line in csv.splitlines())
    output = run([executable(examples, 'periodicity_dump')], input_text=values)
    peaks = []
    for line in output.splitlines():
        lag, bpm, correlation, is_peak = line.split(',')
        if is_peak == '1':
            peaks.append((int(lag), float(bpm), float(correlation)))
    if not peaks:
        return None
    return min(peaks, key=lambda x: (-x[2], x[0]))[1]


def predict(examples, audio, bpm):
    if bpm is None:
        return np.array([], dtype=np.float64)
    lines = run([executable(examples, 'beat_path_dump'), audio, bpm, '3.0'])
    return np.array([float(line) for line in lines.splitlines() if float(line) >= 5.0])


def metrics(reference, estimates):
    matches = len(mir_eval.util.match_events(reference, estimates, window=0.07))
    fp, fn = len(estimates) - matches, len(reference) - matches
    f1 = 2 * matches / (2 * matches + fp + fn) if 2 * matches + fp + fn else 0.0
    if len(estimates):
        assert abs(f1 - mir_eval.beat.f_measure(reference, estimates, f_measure_threshold=0.07)) < 1e-12
    return matches, fp, fn, f1


def summary(rows, which):
    values = [row[which] for row in rows]
    tp, fp, fn = (sum(v[i] for v in values) for i in range(3))
    micro = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    macro = sum(v[3] for v in values) / len(values)
    return tp, fp, fn, micro, macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-zip', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--examples-dir', type=Path, required=True)
    args = parser.parse_args()
    if digest(args.annotation_zip, 'md5') != ANNOTATION_MD5:
        raise SystemExit('annotation.zip does not match the published GuitarSet 1.1.0 MD5')
    with zipfile.ZipFile(args.annotation_zip) as archive:
        selected = select(archive.namelist())
        missing = [n for n in selected if not (args.audio_dir / (n[:-5] + '_mic.wav')).is_file()]
        if missing:
            raise SystemExit(f'missing {len(missing)} microphone WAVs, first: {missing[0]}')
        rows = []
        for i, name in enumerate(selected, 1):
            audio = args.audio_dir / (name[:-5] + '_mic.wav')
            with wave.open(str(audio), 'rb') as wav:
                if wav.getnchannels() != 1 or wav.getframerate() != 44100 or wav.getsampwidth() != 2:
                    raise ValueError(f'{audio}: expected mono 44.1 kHz PCM16')
                duration = wav.getnframes() / wav.getframerate()
            source = archive.read(name)
            reference_bpm, reference = metadata(source, duration)
            candidate = strongest_candidate(args.examples_dir, audio)
            oracle = metrics(reference, predict(args.examples_dir, audio, reference_bpm))
            automatic = metrics(reference, predict(args.examples_dir, audio, candidate))
            rows.append({'name': name[:-5], 'bpm': reference_bpm, 'candidate': candidate,
                         'oracle': oracle, 'auto': automatic, 'audio_sha': digest(audio, 'sha256'),
                         'jams_sha': hashlib.sha256(source).hexdigest()})
            print(f'{i}/50 {name}', file=sys.stderr, flush=True)
    print('# GuitarSet beat-path diagnostic: frozen 50-recording selection\n')
    print('Fixed parameters: quarter-note references, trim both sequences below 5.0 s; '
          'mir_eval 0.8.2, ±70 ms maximum-cardinality matching; tightness 3.0. '
          'Oracle uses annotated constant BPM; auto uses strongest positive autocorrelation local maximum '
          '(no prior, calibrated confidence or abstention except no peak).')
    print('\nSelection: players 01–05, styles BN/Funk/Jazz/Rock/SS, comp and solo; '
          'lexicographically first take for each group. Player 00 was excluded because its excerpt '
          'informed rhythm development. This is a small fixed selection, not the full corpus or a '
          'project-wide held-out set (GuitarSet annotations informed other work).')
    print('\nAnnotation archive MD5: `' + ANNOTATION_MD5 + '`. The per-recording SHA-256 values below '
          'identify analyzed bytes, not independent proof of authenticity.\n')
    print('| Mode | TP | FP | FN | Micro F1 | Mean per-recording F1 |')
    print('|---|---:|---:|---:|---:|---:|')
    for label, key in [('Oracle tempo', 'oracle'), ('Top autocorrelation candidate', 'auto')]:
        tp, fp, fn, micro, macro = summary(rows, key)
        print(f'| {label} | {tp} | {fp} | {fn} | {micro:.4f} | {macro:.4f} |')
    exact = sum(r['candidate'] is not None and abs(r['candidate'] / r['bpm'] - 1) <= 0.04 for r in rows)
    octave = sum(r['candidate'] is not None and any(abs(r['candidate'] / r['bpm'] - ratio) <= 0.04 * ratio for ratio in (0.5, 1, 2)) for r in rows)
    print(f'\nTop tempo within 4% of reference: {exact}/50; within 4% of half, equal or double reference: {octave}/50. '
          'These are simple diagnostic ratios, not calibrated confidence or a standard corpus metric.\n')
    print('| Recording | Reference BPM | Candidate BPM | Oracle TP/FP/FN | Oracle F1 | Auto TP/FP/FN | Auto F1 |')
    print('|---|---:|---:|---|---:|---|---:|')
    for r in rows:
        cand = 'none' if r['candidate'] is None else f"{r['candidate']:.2f}"
        o, a = r['oracle'], r['auto']
        print(f"| {r['name']} | {r['bpm']:.2f} | {cand} | {o[0]}/{o[1]}/{o[2]} | {o[3]:.4f} | {a[0]}/{a[1]}/{a[2]} | {a[3]:.4f} |")
    print('\n## Input hashes\n\n| Recording | WAV SHA-256 | JAMS SHA-256 |\n|---|---|---|')
    for r in rows:
        print(f"| {r['name']} | `{r['audio_sha']}` | `{r['jams_sha']}` |")


if __name__ == '__main__':
    main()
