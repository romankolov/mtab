#!/usr/bin/env python3
"""Independently check playback times in a synthetic auto-beat CLI document."""
import sys
import mido

midi = mido.MidiFile(sys.argv[1])
assert midi.type == 1 and midi.ticks_per_beat == 480
seconds, tempo, tempos, notes = 0.0, 500000, 0, []
for msg in mido.merge_tracks(midi.tracks):
    seconds += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)
    if msg.type == 'set_tempo':
        tempos += 1
        tempo = msg.tempo
    elif msg.type in ('note_on', 'note_off'):
        notes.append((msg.type, msg.note, seconds))
assert tempos >= 2, tempos
expected = [('note_on', 40, 0.2), ('note_off', 40, 0.35),
            ('note_on', 45, 2.1), ('note_off', 45, 2.35)]
assert len(notes) == len(expected), notes
for actual, want in zip(notes, expected):
    assert actual[:2] == want[:2] and abs(actual[2] - want[2]) < 0.0021, (actual, want)
print('Mido verified auto-beat document note playback times')
