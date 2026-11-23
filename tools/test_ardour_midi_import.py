#!/usr/bin/env python3
"""Positive and adversarial tests of the Ardour imported-source comparator."""
import pathlib
import subprocess
import sys
import tempfile
import mido

source, imported = map(pathlib.Path, sys.argv[1:])
checker = pathlib.Path(__file__).with_name('check_ardour_midi_import.py')


def check(path, expected):
    result = subprocess.run([sys.executable, str(checker), str(source), str(path)], capture_output=True, text=True)
    if (result.returncode == 0) != expected:
        raise AssertionError(f'expected success={expected}, got {result.returncode}: {result.stderr}')


check(imported, True)
with tempfile.TemporaryDirectory() as folder:
    output = pathlib.Path(folder) / 'changed.mid'
    midi = mido.MidiFile(imported)
    wheel = next(event for track in midi.tracks for event in track if event.type == 'pitchwheel')
    wheel.pitch += 1
    midi.save(output)
    check(output, False)
    midi = mido.MidiFile(imported)
    note = next(event for track in midi.tracks for event in track if event.type == 'note_on' and event.velocity)
    note.velocity -= 1
    midi.save(output)
    check(output, False)
    midi = mido.MidiFile(imported)
    midi.tracks[0].insert(0, mido.Message('control_change', channel=0, control=101, value=0))
    midi.save(output)
    check(output, False)
print('Ardour MIDI comparator passed positive and three mutation tests')
