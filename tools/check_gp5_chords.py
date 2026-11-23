#!/usr/bin/env python3
"""Independently verify GP5 chord-diagram names/roots/qualities (test-only)."""
import sys
import guitarpro

song = guitarpro.parse(sys.argv[1])
actual = []
for measure in song.tracks[0].measures:
    for beat in measure.voices[0].beats:
        if beat.effect.chord is None:
            continue
        chord = beat.effect.chord
        actual.append((beat.start - 960, chord.name,
                       chord.root.value if chord.root is not None else None,
                       chord.type.name if chord.type is not None else None,
                       chord.firstFret))
# 120 BPM, 1/32 grid (120-tick units): Em7 at 0 s, C at 1.5 s, G at 3.0 s.
assert actual == [
    (0, 'Em7', 4, 'minorSeventh', 0),
    (2880, 'C', 0, 'major', 0),
    (5760, 'G', 7, 'major', 0),
], actual
print('PyGuitarPro verified chord names, roots, qualities and anchors')
