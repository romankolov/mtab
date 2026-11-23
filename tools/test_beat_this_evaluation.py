#!/usr/bin/env python3
"""Unit checks for the external Beat This! evaluation harness."""
import json
import unittest

import numpy as np
from evaluate_beat_this_guitarset import downbeat_reference, parse_estimates


class BeatThisEvaluationTests(unittest.TestCase):
    def test_extract_trim_and_downbeats(self):
        events = [{'time': 0, 'downbeat': True}, {'time': 4.99, 'downbeat': False},
                  {'time': 5.0, 'downbeat': False}, {'time': 6.0, 'downbeat': True}]
        beats, downbeats = parse_estimates(json.dumps({'beats': events}), 7)
        np.testing.assert_array_equal(beats, [5, 6])
        np.testing.assert_array_equal(downbeats, [6])
        empty = parse_estimates('{"beats": []}', 7)
        self.assertEqual(tuple(len(a) for a in empty), (0, 0))

    def test_reject_malformed_times_and_flags(self):
        for events in ([{'time': -1, 'downbeat': False}],
                       [{'time': 8, 'downbeat': False}],
                       [{'time': float('nan'), 'downbeat': False}],
                       [{'time': 1, 'downbeat': 1}],
                       [{'time': 2, 'downbeat': False}, {'time': 2, 'downbeat': True}],
                       [{'time': 2, 'downbeat': False}, {'time': 1, 'downbeat': False}]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                parse_estimates(json.dumps({'beats': events}), 7)

    def test_reference_first_position_only(self):
        data = {'annotations': [{'namespace': 'beat_position', 'data': [
            {'time': 4, 'value': {'position': 1}},
            {'time': 5, 'value': {'position': 2}},
            {'time': 6, 'value': {'position': 1}},
            {'time': 8, 'value': {'position': 1}},
        ]}]}
        np.testing.assert_array_equal(downbeat_reference(json.dumps(data), 8), [6])
        with self.assertRaises(ValueError):
            downbeat_reference(json.dumps(data), 6)


if __name__ == '__main__':
    unittest.main()
