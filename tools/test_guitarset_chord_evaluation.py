#!/usr/bin/env python3
"""Model-free scorer checks for the experimental chord diagnostic."""
import json
import unittest
import evaluate_guitarset_chords as e


class ChordEvaluationTest(unittest.TestCase):
    def test_harte_spelling_and_quality(self):
        for name, expected in [('C', 'C:maj'), ('F#m', 'F#:min'), ('Ebdim', 'Eb:dim'),
                               ('Bb7', 'Bb:7'), ('Cmaj7', 'C:maj7'), ('Am7', 'A:min7')]:
            self.assertEqual(e.harte(name), expected)
        with self.assertRaises(ValueError):
            e.harte('C13')

    def test_frames_compare_reduced_triads_coverage_and_no_chord(self):
        notes = [{'start': 0, 'end': 1.0, 'midi': 60, 'velocity': 100, 'instrument': 24}]
        refs = [(0.0, 0.5, 'C:maj'), (0.5, 0.75, 'N'), (0.75, 1.0, 'X')]
        suggestions = [{'start': 0.0, 'end': 0.25, 'label': 'Cmaj7'},
                       {'start': 0.25, 'end': 0.5, 'label': 'Cm'}]
        counts = e.score(notes, refs, suggestions)
        self.assertEqual(counts['frames'], 100)
        self.assertEqual(counts['incomparable'], 25)
        self.assertEqual(counts['comparable'], 75)
        self.assertEqual(counts['suggested'], 50)
        self.assertEqual(counts['correct_suggested'], 25)
        self.assertEqual(counts['correct_no_suggestion'], 25)
        self.assertEqual(counts['correct'], 50)

    def test_invalid_overlapping_reference_segments_fail(self):
        notes = [{'namespace': 'note_midi', 'annotation_metadata': {'data_source': str(s)},
                  'data': [{'time': 0, 'duration': 1, 'value': 40}] if s == 0 else []} for s in range(6)]
        chord = {'namespace': 'chord', 'data': [{'time': 0, 'duration': 1, 'value': 'C:maj'},
                                                {'time': 0.5, 'duration': 1, 'value': 'C:maj'}]}
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            e.inputs(json.dumps({'annotations': notes + [chord, chord]}).encode())

    def test_suggestion_segments_cannot_overlap(self):
        notes = [{'start': 0, 'end': 1, 'midi': 60}]
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            e.score(notes, [(0, 1, 'C:maj')], [{'start': 0, 'end': 0.8, 'label': 'C'},
                                               {'start': 0.5, 'end': 1, 'label': 'Am'}])


if __name__ == '__main__':
    unittest.main()
