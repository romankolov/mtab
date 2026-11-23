#!/usr/bin/env python3
"""Compare a CLI GP5 to its emitted IR, independently reconstructing ties.

Restricted to the CLI's constant integer BPM, 4/4, straight 1/32-grid unmarked notes;
PyGuitarPro is a test-only dependency. This does not evaluate model accuracy.
"""
import json
import math
import sys
import guitarpro

with open(sys.argv[2], encoding='utf-8') as file:
    doc = json.load(file)
song = guitarpro.parse(sys.argv[1])
tempo = doc['meta']['tempo_map']['changes']
assert len(tempo) == 1 and tempo[0]['time'] == 0
bpm = tempo[0]['bpm']
assert 4 <= bpm <= 255 and bpm == int(bpm)
assert song.tempo == bpm and song.versionTuple == (5, 0, 0)
assert len(song.tracks) == len(doc['tracks'])

def grid(time):
    return math.floor(time * bpm / 60 * 8 + 0.5) * 120

def velocity(v):
    v = max(15, min(v, 127))
    return 15 + min(7, (v - 15 + 8) // 16) * 16

for track, source in zip(song.tracks, doc['tracks']):
    pitches = source['tuning']['strings']
    assert [s.value for s in track.strings] == list(reversed(pitches))
    assert track.offset == source['capo']
    expected = []
    for note in source['notes']:
        assert note['bend'] is None and not note['technique'] and not note['tie_from_prev']
        start = grid(note['start'])
        end = max(start + 120, grid(note['end']))
        expected.append((start, end, len(pitches) - note['string'], note['fret'], velocity(note['velocity'])))
    active, actual = {}, []
    for measure in track.measures:
        for beat in measure.voices[0].beats:
            start = beat.start - 960
            for note in beat.notes:
                if note.type.name == 'tie':
                    assert note.string in active, ('orphan tie', note.string, start)
                    record = active[note.string]
                    assert record[1] == start and record[3] == note.value, (record, start, note.value)
                    record[1] = start + beat.duration.time
                else:
                    assert note.type.name == 'normal'
                    if note.string in active:
                        actual.append(tuple(active.pop(note.string)))
                    active[note.string] = [start, start + beat.duration.time, note.string, note.value, note.velocity]
    actual.extend(map(tuple, active.values()))
    assert sorted(actual) == sorted(expected), f'GP5/IR mismatch\nactual={sorted(actual)}\nexpected={sorted(expected)}'
print(f'PyGuitarPro reconstructed {sum(len(t["notes"]) for t in doc["tracks"])} quantized notes from CLI GP5 and IR')
