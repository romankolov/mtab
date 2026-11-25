#!/usr/bin/env python3
"""Score AutoTab's `tab-labeler-v3` fingering on mtab's own predicted notes.

PROMPT.md section 9 asks for the full pipeline's tablature accuracy against
autotab-rust's `tab-labeler-v3`. This tool builds that comparison so that the
fingering engines are the only difference:

1. Read notes that mtab already transcribed from audio (`nN.jsonl`).
2. Write them to a MIDI file and let the **unmodified** AutoTab CLI assign
   string/fret to exactly those notes.
3. Convert AutoTab's assignment into a TabDocument IR.
4. Score that IR against the GuitarSet annotation with `mtab eval
   --tab-manifest` -- the same implementation used for mtab's own numbers.

Step 4 is the point: both fingerers are scored by the same metric code, on the
same notes, against the same reference. What differs is only the string choice.
This does not measure AutoTab's transcription frontend, and AutoTab's v3 model
was trained on GuitarSet, so its figures here are contaminated by construction.

Nothing external is bundled: the AutoTab executable, its ONNX model and the
ONNX Runtime shared library are supplied as arguments, and the model hash is
checked against the pinned value.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import mido

PPQ = 25000
MODEL_SHA = '93114786c6f4781e7a0b69abd7e93a0d4966873f8e54908f97a1de8e8cf788af'
OPEN_STRINGS = [40, 45, 50, 55, 59, 64]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    notes = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        note = json.loads(line)
        values = (float(note['start']), float(note['end']), int(note['midi']))
        if not all(math.isfinite(v) for v in values[:2]) or values[0] < 0 or values[1] <= values[0]:
            raise ValueError(f'{path}: invalid note {note}')
        notes.append(values)
    notes.sort()
    return notes


def write_midi(notes, path):
    """PPQ 25,000 at 500,000 us/beat, one track, note-off before note-on.

    That is 20 us resolution, the same grid the precommitted oracle protocol
    used, so rounding cannot favour either fingerer.
    """
    events = []
    for index, (start, end, pitch) in enumerate(notes):
        first = round(start * 2 * PPQ)
        last = max(first + 1, round(end * 2 * PPQ))
        events.append((first, 1, index, pitch, last))
        events.append((last, 0, index, pitch, first))
    events.sort()
    midi = mido.MidiFile(type=0, ticks_per_beat=PPQ)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    previous = 0
    for tick, kind, _, pitch, _ in events:
        track.append(
            mido.Message(
                'note_on' if kind else 'note_off',
                note=pitch,
                velocity=100 if kind else 0,
                time=tick - previous,
            )
        )
        previous = tick
    midi.save(path)


def run_autotab(executable, home, midi, out_dir, runtime_dir):
    for suffix in ('txt', 'mid', 'gp5', 'fingering.json'):
        (out_dir / f'auto.{suffix}').unlink(missing_ok=True)
    env = dict(os.environ, AUTOTAB_HOME=str(home))
    if runtime_dir is not None:
        env['LD_LIBRARY_PATH'] = f'{runtime_dir}{os.pathsep}{env.get("LD_LIBRARY_PATH", "")}'
    command = [
        str(executable),
        '--input', str(midi),
        '--output', str(out_dir / 'auto.txt'),
        '--midi-output', str(out_dir / 'auto.mid'),
        '--tab-engine', 'v3',
        '--device', 'cpu',
        '--threads', '1',
        '--midi-track', 'all',
    ]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=600)
    if result.returncode:
        raise ValueError(f'AutoTab exited {result.returncode}: {result.stderr[-800:]}')
    return json.loads((out_dir / 'auto.fingering.json').read_text())


def autotab_rows(report, notes):
    """AutoTab numbers strings 1 (high E) .. 6 (low E); GuitarSet 0 (low E) ..
    5 (high E). Its assignment carries second-resolution times, so each row is
    matched back to the input note it must have come from by pitch and nearest
    onset. Extra-guitar notes are dropped: a second guitar is not a valid
    single-guitar transcription, so scoring it would be generous."""
    by_pitch = {}
    for start, end, pitch in notes:
        by_pitch.setdefault(pitch, []).append((start, end))
    rows, extra, unmatched = [], 0, 0
    for note in report['arrangement']['notes']:
        if note['guitar'] != 1:
            extra += 1
            continue
        pitch = int(note['pitch'])
        candidates = by_pitch.get(pitch)
        if not candidates:
            unmatched += 1
            continue
        start, end = min(candidates, key=lambda c: abs(c[0] - float(note['start_s'])))
        rows.append((start, end, pitch, 6 - int(note['string'])))
    return rows, extra, unmatched, len(report['arrangement']['unassigned'])


def document(title, rows):
    """TabDocument IR. `rows` are (start, end, pitch, string) with GuitarSet
    string numbering."""
    tracks = []
    for string in range(6):
        notes = [
            {
                'start': start,
                'end': end,
                'midi': pitch,
                'string': string,
                'fret': pitch - OPEN_STRINGS[string],
                'velocity': 100,
                'bend': None,
                'technique': [],
                'tie_from_prev': False,
            }
            for start, end, pitch, s in rows
            if s == string
        ]
        if not notes:
            continue
        notes.sort(key=lambda n: (n['start'], n['midi']))
        tracks.append(
            {
                'tuning': {'strings': OPEN_STRINGS, 'name': 'standard'},
                'capo': 0,
                'instrument': None,
                'notes': notes,
            }
        )
    return {
        'meta': {
            'title': title,
            'tempo_map': {'changes': [{'time': 0.0, 'bpm': 120.0}]},
            'time_signatures': [(0, 4, 4)],
            'frets': 22,
            'source_hash': None,
        },
        'tracks': tracks,
        # Empty: neither fingerer produces chord symbols here, and the
        # evaluator ignores them. The field is required by the IR schema.
        'chord_symbols': [],
    }


def parse_metric(report_text, name):
    for line in report_text.splitlines():
        if line.startswith(f'| {name} |'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            return cells[1:]
    raise ValueError(f'missing {name} row')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcripts', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--entries', required=True, help='comma-separated 1-based entry numbers')
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--autotab-home', required=True, type=Path)
    parser.add_argument('--runtime-dir', type=Path)
    parser.add_argument('--mtab', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()

    model = args.autotab_home / 'models/tab-labeler-v3-egset.onnx'
    if not model.is_file():
        raise SystemExit(f'missing AutoTab v3 model at {model}')
    if digest(model) != MODEL_SHA:
        raise SystemExit('AutoTab v3 model SHA-256 does not match the pinned value')

    args.work.mkdir(parents=True, exist_ok=True)
    ir_dir = args.work / 'ir'
    ir_dir.mkdir(exist_ok=True)
    manifest = []
    rows = []
    for number in (int(x) for x in args.entries.split(',')):
        transcript = args.transcripts / f'n{number}.jsonl'
        if not transcript.is_file():
            raise SystemExit(f'entry {number}: missing {transcript}')
        notes = read_jsonl(transcript)
        annotation = args.annotations / f'e{number}.jams'
        if not annotation.is_file():
            raise SystemExit(f'entry {number}: missing {annotation}')
        out_dir = args.work / f'e{number}'
        out_dir.mkdir(exist_ok=True)
        write_midi(notes, out_dir / 'notes.mid')
        report = run_autotab(args.executable, args.autotab_home, out_dir / 'notes.mid', out_dir,
                             args.runtime_dir)
        auto, extra, unmatched, unassigned = autotab_rows(report, notes)
        auto_path = ir_dir / f'e{number}.autotab.json'
        auto_path.write_text(json.dumps(document(f'AutoTab v3, entry {number}', auto), indent=1) + '\n')
        manifest.append({'annotation': str(annotation.resolve()), 'ir': str(auto_path.resolve()),
                         'duration': 10.0})
        rows.append({'entry': number, 'notes': len(notes), 'assigned': len(auto),
                     'extra_guitar': extra, 'unassigned': unassigned, 'unmatched': unmatched,
                     'model_revision': report.get('model_revision')})
        print(f'entry {number}: {len(notes)} notes -> {len(auto)} assigned', flush=True)

    manifest_path = args.work / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=1) + '\n')
    scored = subprocess.run(
        [str(args.mtab), 'eval', '--corpus', 'guitarset', '--tab-manifest', str(manifest_path)],
        text=True, capture_output=True, check=False,
    )
    if scored.returncode:
        print(scored.stdout, file=sys.stderr)
        print(scored.stderr, file=sys.stderr)
        raise SystemExit(f'mtab eval --tab-manifest exited {scored.returncode}')

    summary = {
        'auto_tab_executable_sha256': digest(args.executable),
        'auto_tab_model_sha256': MODEL_SHA,
        'entries': rows,
        'autotab_multi_pitch': parse_metric(scored.stdout, 'Multi-pitch frames'),
        'autotab_tab_position': parse_metric(scored.stdout, 'Tab position frames'),
        'autotab_pitch_onset': parse_metric(scored.stdout, 'Pitch + onset notes'),
        'autotab_tab_onset': parse_metric(scored.stdout, 'Tab + onset notes'),
    }
    text = json.dumps(summary, indent=1) + '\n'
    (args.work / 'head-to-head.json').write_text(text)
    (args.work / 'autotab-report.md').write_text(scored.stdout)
    if args.report:
        args.report.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
