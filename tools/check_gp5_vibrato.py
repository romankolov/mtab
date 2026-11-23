#!/usr/bin/env python3
"""Independently verify GP5 note vibrato flags and ordering (test-only)."""
import sys
import guitarpro

song = guitarpro.parse(sys.argv[1])
actual = []
for measure in song.tracks[0].measures:
    for beat in measure.voices[0].beats:
        for note in beat.notes:
            actual.append((beat.start - 960, note.string, note.value, note.type.name,
                           note.effect.vibrato, note.effect.bend is not None,
                           note.effect.palmMute))
expected = [
    (0, 6, 2, 'normal', True, False, False),
    (960, 6, 2, 'normal', False, False, True),
    (1920, 6, 2, 'normal', True, True, False),
]
assert actual == expected, f"actual={actual!r}\nexpected={expected!r}"
print("PyGuitarPro verified vibrato flags, bend coexistence and untouched neighbours")
