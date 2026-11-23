#!/usr/bin/env python3
"""Make a negative-control MIDI with channel pitch-wheel messages removed.

Preserves delta timing while removing events; mido is a test-only dependency.
"""
import sys
import mido

midi = mido.MidiFile(sys.argv[1])
for index, track in enumerate(midi.tracks):
    result = mido.MidiTrack()
    accumulated = 0
    for event in track:
        accumulated += event.time
        if event.type != 'pitchwheel':
            result.append(event.copy(time=accumulated))
            accumulated = 0
    midi.tracks[index] = result
midi.save(sys.argv[2])
