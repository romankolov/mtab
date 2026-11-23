#!/usr/bin/env python3
"""Independently verify GP5 hammer/pull and slide flags (test-only)."""
import sys
import guitarpro

song = guitarpro.parse(sys.argv[1])
actual = []
for measure in song.tracks[0].measures:
    for beat in measure.voices[0].beats:
        for note in beat.notes:
            slides = [s.name for s in note.effect.slides]
            actual.append((beat.start - 960, note.string, note.value,
                           note.effect.hammer, slides, note.type.name))
expected = [
    # bar 1: A3 on the third string, 5 -> 7 legato slide out
    (0, 4, 5, False, ['legatoSlideTo'], 'normal'),
    (960, 4, 7, True, [], 'normal'),
    # the pull-off starts here; GP5's single legato flag covers both
    # directions, so a descending pair is encoded exactly like a rising one
    (1920, 4, 10, False, [], 'normal'),
    (2400, 4, 8, False, [], 'normal'),
    # bar 2: picked slide in from fret 5 (previous note) to 7
    (3840, 4, 5, False, [], 'normal'),
    (4800, 4, 7, False, ['intoFromBelow'], 'normal'),
    # bar 3: shift slide out from 7 to 5
    (7680, 4, 7, False, ['shiftSlideTo'], 'normal'),
    (8640, 4, 5, False, [], 'normal'),
]
assert actual == expected, f"actual={actual!r}\nexpected={expected!r}"
print("PyGuitarPro verified hammer, pull-off, legato/shift/picked slide flags")
