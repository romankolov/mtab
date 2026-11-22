#!/usr/bin/env python3
"""Verify a GP5 written by a silent synthetic-model CLI test."""
import sys
import guitarpro
song = guitarpro.parse(sys.argv[1])
assert len(song.tracks) == 1
assert len(song.tracks[0].measures) == 1
assert not any(note for track in song.tracks for measure in track.measures
               for voice in measure.voices for beat in voice.beats for note in beat.notes)
print('PyGuitarPro verified an empty score with one guitar track')
