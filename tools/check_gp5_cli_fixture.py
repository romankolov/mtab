#!/usr/bin/env python3
"""Independently check the model-free CLI mixed-output GP5 fixture (test-only)."""
import sys
import guitarpro

song = guitarpro.parse(sys.argv[1])
assert song.versionTuple == (5, 0, 0)
assert song.tempo == 120
assert len(song.tracks) == 1
track = song.tracks[0]
assert track.offset == 0
assert [s.value for s in track.strings] == [64, 59, 55, 50, 45, 40]
notes = [(beat.start - 960, beat.duration.time, note.string, note.value, note.velocity, note.type.name)
         for measure in track.measures for beat in measure.voices[0].beats for note in beat.notes]
assert notes == [(960, 960, 6, 0, 79, "normal")], notes
print("PyGuitarPro verified the CLI GP5 fixture")
