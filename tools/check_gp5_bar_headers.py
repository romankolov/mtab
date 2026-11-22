#!/usr/bin/env python3
"""Check single- and multi-measure GP5 framing with the independent parser."""
import sys
import guitarpro
song = guitarpro.parse(sys.argv[1])
bars = int(sys.argv[2])
assert len(song.tracks) == 1
assert len(song.measureHeaders) == bars
track = song.tracks[0]
assert len(track.measures) == bars
for index, measure in enumerate(track.measures):
    assert measure.timeSignature.numerator == 4
    assert measure.timeSignature.denominator.value == 4
    beats = measure.voices[0].beats
    assert len(beats) == 1
    beat = beats[0]
    assert beat.start - 960 == 3840 * index
    assert beat.duration.time == 3840
    assert len(beat.notes) == 1
    note = beat.notes[0]
    assert (note.string, note.value, note.type.name) == (6, 3, 'normal' if index == 0 else 'tie')
