#!/usr/bin/env python3
"""Independently verify GP5 bend types and control points (test-only)."""
import sys
import guitarpro

song = guitarpro.parse(sys.argv[1])
notes = [note for measure in song.tracks[0].measures
         for beat in measure.voices[0].beats for note in beat.notes]
actual = [(note.type.name, note.effect.palmMute, note.effect.bend.type.name,
           note.effect.bend.value,
           [(p.position, p.value, p.vibrato) for p in note.effect.bend.points])
          for note in notes]
expected = [
    ('normal', True, 'bend', 50, [(0, 0, False), (4, 2, False), (12, 2, False)]),
    ('normal', False, 'prebendRelease', 100, [(0, 4, False), (6, 4, False), (12, 0, False)]),
    ('normal', False, 'bendRelease', 50, [(0, 0, False), (8, 2, False), (12, 0, False)]),
]
assert actual == expected, f"actual={actual!r}\nexpected={expected!r}"
print("PyGuitarPro verified bend, prebend release, bend-release and palm-mute control points")
