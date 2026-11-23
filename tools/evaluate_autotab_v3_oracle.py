#!/usr/bin/env python3
"""Run the precommitted external AutoTab oracle-note diagnostic, offline.

Requires Mido, GuitarSet's annotation.zip, and the independently built
external executable plus its pretrained ONNX model and runtime. See
benches/m5-autotab-v3-protocol.md. No model or application code is bundled.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

import mido

ZIP_MD5 = 'b39b78e63d3446f2e54ddb7a54df9b10'
MODEL_SHA = '93114786c6f4781e7a0b69abd7e93a0d4966873f8e54908f97a1de8e8cf788af'
PPQ = 25000


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def reference(data):
    obj = json.loads(data)
    arrays = [a for a in obj['annotations'] if a['namespace'] == 'note_midi']
    if len(arrays) != 6:
        raise ValueError('expected six string annotations')
    strings = set()
    result = []
    for a in arrays:
        channel = int(a['annotation_metadata']['data_source'])
        if channel not in range(6) or channel in strings:
            raise ValueError('invalid or duplicate string')
        strings.add(channel)
        for n in a['data']:
            start, duration, pitch = n['time'], n['duration'], n['value']
            if not all(math.isfinite(x) for x in (start, duration, pitch)) or start < 0 or duration <= 0:
                raise ValueError('invalid note')
            start_tick = round(start * 2 * PPQ)
            end_tick = max(start_tick + 1, round((start + duration) * 2 * PPQ))
            pitch = round(pitch)
            if not 0 <= pitch <= 127:
                raise ValueError('invalid pitch')
            result.append((channel, pitch, start_tick, end_tick))
    return result


def write_midi(notes, path):
    # Verify FIFO note-off pairing used by the external program will preserve
    # the endpoints of every individual same-channel, same-pitch note.
    events = []
    for channel, pitch, start, end in notes:
        events.extend([(start, 1, channel, pitch, end), (end, 0, channel, pitch, start)])
    events.sort()
    queues = collections.defaultdict(collections.deque)
    for tick, kind, channel, pitch, other in events:
        queue = queues[channel, pitch]
        if kind:
            queue.append(other)
        elif not queue or queue.popleft() != tick:
            raise ValueError('FIFO MIDI note pairing changes same-string/pitch endpoint')
    if any(queues.values()):
        raise ValueError('unterminated MIDI notes')
    midi = mido.MidiFile(type=0, ticks_per_beat=PPQ)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    previous = 0
    for tick, kind, channel, pitch, _ in events:
        track.append(mido.Message('note_on' if kind else 'note_off', channel=channel,
                                  note=pitch, velocity=100 if kind else 0, time=tick-previous))
        previous = tick
    midi.save(path)


def run_one(executable, home, name, data, work):
    notes = reference(data)
    row = {'name': name, 'jams_sha256': hashlib.sha256(data).hexdigest(), 'notes': len(notes)}
    try:
        input_path, output_path = work / 'input.mid', work / 'output.txt'
        write_midi(notes, input_path)
        for suffix in ('txt', 'mid', 'gp5', 'fingering.json'):
            (work / f'output.{suffix}').unlink(missing_ok=True)
        env = dict(os.environ, AUTOTAB_HOME=str(home))
        result = subprocess.run([str(executable), '--input', str(input_path), '--output', str(output_path),
                                 '--tab-engine', 'v3', '--device', 'cpu', '--threads', '1',
                                 '--midi-track', 'all'], env=env, text=True, capture_output=True, timeout=180)
        if result.returncode:
            raise ValueError(f'external CLI exit {result.returncode}: {result.stderr[-1000:]}')
        report = json.loads((work / 'output.fingering.json').read_text())
        assert report['engine'] == 'V3' or report['engine'] == 'v3'
        sources = report['source_notes']
        source_keys = [(n['channel'], n['pitch'], n['start_tick'], n['end_tick']) for n in sources]
        if collections.Counter(source_keys) != collections.Counter(notes):
            raise ValueError('external source notes differ from generated MIDI references')
        assignment = report['arrangement']['notes']
        indices = [n['index'] for n in assignment]
        if len(set(indices)) != len(indices) or any(i not in range(len(notes)) for i in indices):
            raise ValueError('duplicate or invalid assignment index')
        unassigned = report['arrangement']['unassigned']
        if len(set(unassigned)) != len(unassigned) or set(unassigned) & set(indices) or set(unassigned) | set(indices) != set(range(len(notes))):
            raise ValueError('assignment plus unassigned does not partition input')
        correct = sum(n['string'] == 6 - sources[n['index']]['channel'] for n in assignment)
        primary = [n for n in assignment if n['guitar'] == 1]
        row.update(assigned=len(indices), unassigned=len(unassigned), extra_guitar=len(indices)-len(primary),
                   correct_all=correct,
                   correct_primary=sum(n['string'] == 6 - sources[n['index']]['channel'] for n in primary))
    except (ValueError, KeyError, AssertionError, subprocess.TimeoutExpired) as e:
        row['error'] = str(e) or type(e).__name__
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--autotab-home', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if digest(args.annotations, 'md5') != ZIP_MD5:
        raise SystemExit('annotation ZIP MD5 mismatch')
    model = args.autotab_home / 'models/tab-labeler-v3-egset.onnx'
    if digest(model) != MODEL_SHA:
        raise SystemExit('model SHA-256 mismatch')
    executable = args.executable.resolve()
    home = args.autotab_home.resolve()
    rows = []
    with tempfile.TemporaryDirectory(prefix='mtab-autotab-oracle-') as temp, zipfile.ZipFile(args.annotations) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith('.jams'))
        if len(names) != 360 or len(set(names)) != 360:
            raise SystemExit('expected 360 distinct JAMS files')
        for i, name in enumerate(names, 1):
            row = run_one(executable, home, name, archive.read(name), Path(temp))
            rows.append(row)
            print(f"{i}/360 {name}: {row.get('error', 'ok')}", flush=True)
    result = {'protocol': 'benches/m5-autotab-v3-protocol.md', 'annotation_md5': ZIP_MD5,
              'model_sha256': MODEL_SHA, 'executable_sha256': digest(executable),
              'rows': rows}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('completed:', len(rows), 'failures:', sum('error' in r for r in rows))


if __name__ == '__main__':
    main()
