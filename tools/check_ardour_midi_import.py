#!/usr/bin/env python3
"""Compare the generated bend MIDI to Ardour 8.12's saved imported source.

Mido is a test-only dependency. Ardour stores each imported source as type 0,
19200 PPQ. This fixture verifies the observed note/wheel preservation AND RPN
loss; it does not assert that any other DAW/version behaves the same way.
"""
import sys
from fractions import Fraction
import mido


def extract(path):
    midi = mido.MidiFile(path)
    events = {'notes': [], 'wheels': [], 'programs': [], 'rpn_controls': []}
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += message.time
            at = Fraction(absolute, midi.ticks_per_beat)
            if message.type in ('note_on', 'note_off'):
                events['notes'].append((at, message.type, message.channel, message.note, message.velocity))
            elif message.type == 'pitchwheel':
                events['wheels'].append((at, message.channel, message.pitch))
            elif message.type == 'program_change':
                events['programs'].append((at, message.channel, message.program))
            elif message.type == 'control_change' and message.control in (6, 38, 100, 101):
                events['rpn_controls'].append((at, message.channel, message.control, message.value))
    return midi, events


def main():
    source, imported = sys.argv[1:]
    original, before = extract(source)
    saved, after = extract(imported)
    if (original.type, original.ticks_per_beat) != (1, 480):
        raise AssertionError('unexpected source MIDI format')
    if (saved.type, saved.ticks_per_beat) != (0, 19200):
        raise AssertionError('unexpected Ardour saved-source MIDI format')
    expected_notes = [(Fraction(0), 'note_on', 0, 60, 92),
                      (Fraction(2), 'note_off', 0, 60, 0)]
    expected_rpn = [(Fraction(0), 0, control, value)
                    for control, value in [(101, 0), (100, 0), (6, 2),
                                           (38, 0), (101, 127), (100, 127)]]
    if before['notes'] != expected_notes or after['notes'] != expected_notes:
        raise AssertionError('imported note events or velocities differ')
    if before['programs'] != [(Fraction(0), 0, 25)] or after['programs'] != before['programs']:
        raise AssertionError('program change differs')
    if len(before['wheels']) != 51 or after['wheels'] != before['wheels']:
        raise AssertionError('imported pitch-wheel events or musical times differ')
    if before['rpn_controls'] != expected_rpn or after['rpn_controls']:
        raise AssertionError('observed RPN loss differs from the documented fixture')
    print('Ardour saved source: note, velocity, program and 51 pitch-wheel events preserved; pitch-sensitivity RPN absent')


if __name__ == '__main__':
    main()
