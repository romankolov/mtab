#!/usr/bin/env python3
"""Independent check for a bend attached from a synthetic isolated note."""
import sys
import mido

midi = mido.MidiFile(sys.argv[1])
seconds, tempo, notes, wheels = 0.0, 500000, [], []
for msg in mido.merge_tracks(midi.tracks):
    seconds += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)
    if msg.type == 'set_tempo':
        tempo = msg.tempo
    elif msg.type in ('note_on', 'note_off'):
        notes.append((msg.type, msg.note, seconds, msg.channel))
    elif msg.type == 'pitchwheel':
        wheels.append((seconds, msg.pitch, msg.channel))
assert [event[:2] for event in notes] == [('note_on', 45), ('note_off', 45)], notes
assert abs(notes[0][2] - 0.2) < 0.002 and abs(notes[1][2] - 1.2) < 0.002, notes
channel = notes[0][3]
active = [(t, value) for t, value, ch in wheels if ch == channel and 0.2 <= t <= 1.2]
assert active and max(value for _, value in active) > 2500, active
assert all(0 <= value <= 8191 for _, value in active), active
assert any(value == 0 and t >= notes[1][2] for t, value, ch in wheels if ch == channel), wheels
print('Mido verified synthetic inferred bend, note timing, and wheel reset')
