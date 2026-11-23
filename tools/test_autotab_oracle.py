#!/usr/bin/env python3
"""Model-free tests for MIDI preparation and reference pairing."""
import json
from pathlib import Path
import tempfile
import unittest

import mido
import evaluate_autotab_v3_oracle as oracle


class OracleTest(unittest.TestCase):
    def test_reference_and_midi_preserve_note_identities(self):
        annotations = []
        for string in range(6):
            annotations.append({'namespace': 'note_midi', 'annotation_metadata': {'data_source': str(string)},
                                'data': [{'time': string + 0.25, 'duration': 0.5, 'value': 40 + string}]})
        notes = oracle.reference(json.dumps({'annotations': annotations}).encode())
        self.assertEqual(notes[0], (0, 40, 12500, 37500))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'notes.mid'
            oracle.write_midi(notes, path)
            midi = mido.MidiFile(path)
            self.assertEqual(midi.ticks_per_beat, 25000)
            tick, ons, offs = 0, [], []
            for message in midi.tracks[0]:
                tick += message.time
                if message.type == 'note_on':
                    ons.append((message.channel, message.note, tick))
                elif message.type == 'note_off':
                    offs.append((message.channel, message.note, tick))
            self.assertEqual(ons, [(c, p, start) for c, p, start, end in notes])
            self.assertEqual(offs, [(c, p, end) for c, p, start, end in notes])

    def test_nested_same_string_note_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'FIFO'):
                oracle.write_midi([(0, 60, 1, 10), (0, 60, 2, 5)], Path(temp) / 'notes.mid')
            # The same pitch on separate channels is unambiguous.
            oracle.write_midi([(0, 60, 1, 10), (1, 60, 2, 5)], Path(temp) / 'notes.mid')

    def test_same_tick_note_off_precedes_note_on(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'notes.mid'
            oracle.write_midi([(0, 60, 1, 10), (0, 60, 10, 20)], path)
            kinds = [m.type for m in mido.MidiFile(path).tracks[0] if m.type in ('note_on', 'note_off')]
            self.assertEqual(kinds, ['note_on', 'note_off', 'note_on', 'note_off'])


if __name__ == '__main__':
    unittest.main()
