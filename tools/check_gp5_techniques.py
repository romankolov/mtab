#!/usr/bin/env python3
"""Independently verify dead-note and palm-mute GP5 encoding (test-only)."""
import sys
import guitarpro

song = guitarpro.parse(sys.argv[1])
assert len(song.tracks) == 2
actual = []
for measure in song.tracks[0].measures:
    for beat in measure.voices[0].beats:
        for note in beat.notes:
            actual.append((beat.start - 960, note.string, note.value, note.type.name,
                           note.effect.palmMute, note.effect.staccato))
expected = [
    (0, 6, 3, 'dead', True, False),
    (960, 5, 2, 'normal', True, False),
    (960, 6, 3, 'tie', True, False),
    (1920, 6, 3, 'tie', True, False),
    (3840, 4, 0, 'dead', False, False),
    (3840, 6, 3, 'tie', True, False),
]
assert actual == expected, f"actual={actual!r}\nexpected={expected!r}"
print("PyGuitarPro verified dead-note attacks and palm-mute marks, including ties")
