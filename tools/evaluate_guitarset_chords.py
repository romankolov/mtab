#!/usr/bin/env python3
"""Oracle-note GuitarSet sheet-chord diagnostic; see precommitted protocol.

Input annotation.zip and a locally built mtab-guitar chord_dump example are
required. No audio, model weights or downloads are used.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile

import mir_eval.chord
import numpy as np

ZIP_MD5 = 'b39b78e63d3446f2e54ddb7a54df9b10'
SUFFIX = {'': 'maj', 'm': 'min', 'dim': 'dim', '7': '7', 'maj7': 'maj7', 'm7': 'min7'}


def digest(data, algorithm='sha256'):
    return hashlib.new(algorithm, data).hexdigest()


def inputs(data):
    obj = json.loads(data)
    annotations = obj['annotations']
    notes = []
    seen = set()
    for ann in annotations:
        if ann['namespace'] != 'note_midi':
            continue
        string = int(ann['annotation_metadata']['data_source'])
        if string not in range(6) or string in seen:
            raise ValueError('duplicate or invalid string annotation')
        seen.add(string)
        for row in ann['data']:
            start, duration, pitch = row['time'], row['duration'], row['value']
            if not all(math.isfinite(x) for x in (start, duration, pitch)) or start < 0 or duration <= 0:
                raise ValueError('invalid reference note')
            # Rust's f64::round rounds half away from zero; pitches are nonnegative.
            midi = math.floor(pitch + 0.5)
            if not 0 <= midi <= 127:
                raise ValueError('invalid MIDI pitch')
            notes.append(dict(start=start, end=start + duration, midi=midi, velocity=100, instrument=24))
    if seen != set(range(6)) or not notes:
        raise ValueError('missing string annotation or no notes')
    chords = [ann for ann in annotations if ann['namespace'] == 'chord']
    if len(chords) != 2:
        raise ValueError('expected exactly two chord annotations')
    progression = []
    end = 0.0
    for row in chords[0]['data']:
        start, duration, value = row['time'], row['duration'], row['value']
        if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0 or start < end:
            raise ValueError('invalid, overlapping or out-of-order chord segments')
        mir_eval.chord.validate_chord_label(value)
        end = start + duration
        progression.append((start, end, value))
    if not progression:
        raise ValueError('empty chord progression')
    return notes, progression


def harte(label):
    match = re.fullmatch(r'([A-G](?:#|b)?)(maj7|m7|dim|m|7)?', label)
    if not match:
        raise ValueError(f'unexpected suggestion {label}')
    return match.group(1) + ':' + SUFFIX[match.group(2) or '']


def sequence_labels(segments, times):
    starts = [s[0] for s in segments]
    return [segments[i][2] if (i := bisect.bisect_right(starts, t) - 1) >= 0 and t < segments[i][1] else 'N'
            for t in times]


def score(notes, progression, suggestions):
    last_note = max(n['end'] for n in notes)
    end = min(last_note, progression[-1][1])
    if not math.isfinite(end) or end <= 0 or end > 28800:
        raise ValueError('invalid scoring duration')
    frames = np.arange(0.005, end, 0.01, dtype=np.float64)
    segments = []
    previous_end = 0.0
    for s in suggestions:
        start, stop, name = s['start'], s['end'], s['label']
        if not math.isfinite(start) or not math.isfinite(stop) or start < previous_end or stop <= start:
            raise ValueError('invalid or overlapping suggestion intervals')
        segments.append((start, stop, harte(name)))
        previous_end = stop
    refs = sequence_labels(progression, frames)
    preds = sequence_labels(segments, frames)
    values = mir_eval.chord.triads(refs, preds)
    comparable = values >= 0
    correct = values == 1
    predicted = np.array([p != 'N' for p in preds], dtype=bool)
    return dict(frames=len(frames), incomparable=int(np.sum(~comparable)),
                comparable=int(np.sum(comparable)), suggested=int(np.sum(comparable & predicted)),
                correct=int(np.sum(comparable & correct)),
                correct_suggested=int(np.sum(comparable & predicted & correct)),
                correct_no_suggestion=int(np.sum(comparable & ~predicted & correct)),
                suggestions=len(suggestions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    archive_bytes = args.annotations.read_bytes()
    if digest(archive_bytes, 'md5') != ZIP_MD5:
        raise SystemExit('annotation ZIP MD5 mismatch')
    if mir_eval.__version__ != '0.8.2':
        raise SystemExit('expected mir_eval 0.8.2')
    executable = args.executable.resolve()
    rows = []
    with zipfile.ZipFile(args.annotations) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith('.jams'))
        if len(names) != 360 or len(set(names)) != 360:
            raise SystemExit('expected 360 distinct JAMS files')
        for index, name in enumerate(names, 1):
            data = archive.read(name)
            notes, progression = inputs(data)
            output = subprocess.run([str(executable)], input=json.dumps(notes), text=True,
                                    capture_output=True, check=True, timeout=60)
            row = dict(name=name, jams_sha256=digest(data), notes=len(notes))
            row.update(score(notes, progression, json.loads(output.stdout)))
            rows.append(row)
            print(f'{index}/360 {name}: {row["suggestions"]} suggestions', flush=True)
    report = dict(protocol='benches/m7-guitarset-chord-protocol.md', archive_md5=ZIP_MD5,
                  mir_eval_version=mir_eval.__version__, executable_sha256=digest(executable.read_bytes()), rows=rows)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
