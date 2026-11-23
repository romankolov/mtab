#!/usr/bin/env python3
"""Verify the controlled dynamics MIDI and produce a constant-velocity control.

Test-only dependency: mido==1.3.3. Does not measure real-guitar dynamics.
"""
import sys
import mido

source, destination = sys.argv[1:]
midi = mido.MidiFile(source)
if midi.type != 1 or midi.ticks_per_beat != 480:
    raise ValueError('expected type 1 MIDI with 480 PPQ')
notes_on, notes_off, tempos, meters = [], [], [], []
for track in midi.tracks:
    absolute = 0
    for message in track:
        absolute += message.time
        if message.type == 'note_on' and message.velocity:
            notes_on.append((absolute, message.note, message.velocity))
            message.velocity = 80
        elif message.type == 'note_off' or (message.type == 'note_on' and not message.velocity):
            notes_off.append((absolute, message.note))
        elif message.type == 'set_tempo':
            tempos.append((absolute, message.tempo))
        elif message.type == 'time_signature':
            meters.append((absolute, message.numerator, message.denominator))
if sorted(notes_on) != [(192, 45, 40), (768, 45, 80), (1344, 45, 120)]:
    raise AssertionError(f'unexpected note-ons: {notes_on}')
if sorted(notes_off) != [(480, 45), (1056, 45), (1632, 45)]:
    raise AssertionError(f'unexpected note-offs: {notes_off}')
if tempos != [(0, 500000)] or meters != [(0, 4, 4)]:
    raise AssertionError(f'unexpected tempo or meter: {tempos}, {meters}')
midi.save(destination)
print('Mido verified dynamics fixture; wrote constant-velocity control')
