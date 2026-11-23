#!/usr/bin/env python3
"""Check GP5 let-ring flags independently, including tied continuations."""
import sys
import guitarpro

song = guitarpro.parse(sys.argv[1])
notes = []
for measure in song.tracks[0].measures:
    for beat in measure.voices[0].beats:
        for note in beat.notes:
            notes.append((beat.start - 960, note.string, note.type.name, note.effect.letRing))
assert notes == [(0, 6, 'normal', True),
                 (960, 5, 'normal', True),
                 (960, 6, 'tie', True),
                 (1920, 6, 'tie', True),
                 (3840, 4, 'normal', False),
                 (3840, 6, 'tie', True)], notes
print('PyGuitarPro verified let-ring attacks and tied segments')
