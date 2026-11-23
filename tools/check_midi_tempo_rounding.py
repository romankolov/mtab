#!/usr/bin/env python3
"""Check playback seconds independently after many fractional MIDI tempos."""
import sys
import mido

midi = mido.MidiFile(sys.argv[1])
assert midi.ticks_per_beat == 480
seconds, tempo, notes, tempos, peaks = 0.0, 500000, [], 0, []
for message in mido.merge_tracks(midi.tracks):
    seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
    if message.type == 'set_tempo':
        tempo = message.tempo
        assert tempo == 500000
        tempos += 1
    elif message.type == 'pitchwheel' and message.pitch == 4096:
        peaks.append(seconds)
    elif message.type in ('note_on', 'note_off'):
        notes.append((message.type, message.note, seconds))
assert tempos == 10000, tempos
assert len(peaks) == 1, peaks
assert [(kind, note) for kind, note, _ in notes] == [('note_on', 40), ('note_off', 40)]
expected = 9999 * 0.50000049 + 0.25
for observed, target in [(notes[0][2], expected), (peaks[0], expected + 0.0625), (notes[1][2], expected + 0.125)]:
    assert abs(observed - target) <= 0.500000 / (2 * 480) + 1e-8, (observed, target, observed - target)
print(f'Mido: long fractional-tempo onset/peak/offset errors {notes[0][2] - expected:+.6f}s / {peaks[0] - expected - 0.0625:+.6f}s / {notes[1][2] - expected - 0.125:+.6f}s')
