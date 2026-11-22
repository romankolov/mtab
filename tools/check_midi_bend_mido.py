#!/usr/bin/env python3
"""Independent parser check for the Rust MIDI bend fixture; requires mido."""
import sys
import mido

midi = mido.MidiFile(sys.argv[1])
assert midi.type == 1 and midi.ticks_per_beat == 480 and len(midi.tracks) == 2
absolute = 0
wheels = []
notes = []
rpn = []
for msg in midi.tracks[1]:
    absolute += msg.time
    if msg.type == 'pitchwheel':
        wheels.append((absolute, msg.channel, msg.pitch))
    if msg.type in ('note_on', 'note_off'):
        notes.append((absolute, msg.channel, msg.type, msg.note, msg.velocity))
    if msg.type == 'control_change':
        rpn.append((msg.channel, msg.control, msg.value))
assert notes == [(0, 0, 'note_on', 60, 92), (960, 0, 'note_off', 60, 0)], notes
assert rpn == [(0, 101, 0), (0, 100, 0), (0, 6, 2), (0, 38, 0), (0, 101, 127), (0, 100, 127)], rpn
assert wheels[0] == (0, 0, 0) and wheels[-1] == (960, 0, 0)
assert all(channel == 0 for _, channel, _ in wheels)
assert all(a[0] < b[0] for a, b in zip(wheels, wheels[1:]))
peak = wheels.index((480, 0, 4096))
assert 2 < peak < len(wheels) - 3
assert all(a[2] <= b[2] for a, b in zip(wheels[:peak], wheels[1:peak + 1]))
assert all(a[2] >= b[2] for a, b in zip(wheels[peak:], wheels[peak + 1:]))
print(f'validated {len(wheels)} interpolated wheel events, endpoints, peak, RPN and note')
