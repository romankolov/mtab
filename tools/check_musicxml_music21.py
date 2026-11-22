#!/usr/bin/env python3
"""Independent music21 interpretation of the fixed MusicXML test score."""
import sys
from music21 import chord, converter

score = converter.parse(sys.argv[1])
assert len(score.parts) == 4, len(score.parts)  # two staves per guitar
expected = [
    (0.0, 1.0, [(45, 'start')]),
    (1.0, 1.0, [(45, 'continue'), (49, None)]),
    (2.0, 2.0, [(45, 'continue')]),
    (4.0, 1.0, [(45, 'stop'), (52, None)]),
]
for part in score.parts[:2]:
    events = []
    for event in part.recurse().notes:
        notes = event.notes if isinstance(event, chord.Chord) else [event]
        events.append((float(event.getOffsetInHierarchy(part)), float(event.quarterLength),
                       [(n.pitch.midi, n.tie.type if n.tie else None) for n in notes]))
    assert events == expected, events
for part in score.parts[2:]:
    events = list(part.recurse().notes)
    assert len(events) == 1 and events[0].pitch.midi == 36
    assert float(events[0].getOffsetInHierarchy(part)) == 1.0
print('music21 independently read pitches, staff timing, chords and ties')
