#!/usr/bin/env python3
"""Independent Mido check of a synthetic offset-beat variable-tempo MIDI fixture."""
import sys
import mido

midi = mido.MidiFile(sys.argv[1])
assert midi.type == 1 and midi.ticks_per_beat == 480
notes, tempos, meters = [], [], []
for track in midi.tracks:
    tick = 0
    for msg in track:
        tick += msg.time
        if msg.type == 'set_tempo':
            tempos.append((tick, msg.tempo))
        elif msg.type == 'time_signature':
            meters.append((tick, msg.numerator, msg.denominator))
        elif msg.type in ('note_on', 'note_off'):
            notes.append((tick, msg.type, msg.note, msg.velocity))
assert tempos == [(0, 250000), (480, 500000), (960, 600000), (1440, 700000)], tempos
assert meters == [(0, 4, 4)], meters
expected = []
for on, off, pitch in [(240, 480, 40), (720, 960, 45), (1200, 1440, 50), (1680, 1920, 55)]:
    expected.extend([(on, 'note_on', pitch, 90), (off, 'note_off', pitch, 0)])
assert sorted(notes) == sorted(expected), notes
# Mido's independent tempo map integration recovers absolute wall-clock times.
wall_times, seconds, tempo = [], 0.0, 500000
for message in mido.merge_tracks(midi.tracks):
    # Use current tempo for the preceding delta; updates apply afterward.
    seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
    if message.type == 'set_tempo':
        tempo = message.tempo
    elif message.type in ('note_on', 'note_off'):
        wall_times.append((message.type, message.note, seconds))
expected_seconds = [(0.125, 0.25), (0.5, 0.75), (1.05, 1.35), (1.7, 2.05)]
for i, (start, end) in enumerate(expected_seconds):
    assert abs(wall_times[2*i][2] - start) < 1e-9, wall_times
    assert abs(wall_times[2*i+1][2] - end) < 1e-9, wall_times
print('Mido verified offset-beat tempo ticks, metadata, velocities and playback times')
