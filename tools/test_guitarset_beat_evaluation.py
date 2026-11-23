#!/usr/bin/env python3
"""Unit checks for the offline GuitarSet beat evaluation harness."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import evaluate_guitarset_beat_paths as evaluator


class TestBeatEvaluation(unittest.TestCase):
    def test_selection_is_fixed_balanced_and_excludes_player_zero(self):
        names = []
        for player in range(6):
            for style in evaluator.STYLES:
                for take in ('2-180-G', '1-120-A'):
                    for mode in ('solo', 'comp'):
                        names.append(f'{player:02d}_{style}{take}_{mode}.jams')
        selected = evaluator.select(list(reversed(names)))
        self.assertEqual(len(selected), 50)
        self.assertTrue(all(not n.startswith('00_') and '1-120-A' in n for n in selected))
        self.assertEqual(sum(n.endswith('_comp.jams') for n in selected), 25)
        self.assertEqual(sum(n.endswith('_solo.jams') for n in selected), 25)
        with self.assertRaises(ValueError):
            evaluator.select([])

    def test_metadata_trims_first_five_seconds_and_validates_meter(self):
        beats = [{'time': time, 'value': {'beat_units': 4}} for time in (0, 4.99, 5, 5.5, 6, 9)]
        annotation = {'annotations': [
            {'namespace': 'beat_position', 'data': beats},
            {'namespace': 'tempo', 'data': [{'value': 120}]},
        ]}
        bpm, times = evaluator.metadata(json.dumps(annotation).encode(), 6.0)
        self.assertEqual(bpm, 120)
        np.testing.assert_array_equal(times, [5.0, 5.5])
        beats[0]['value']['beat_units'] = 8
        with self.assertRaises(ValueError):
            evaluator.metadata(json.dumps(annotation).encode(), 6.0)

    def test_beat_matching_is_one_to_one_and_counts_empty(self):
        ref = np.array([5.0, 6.0, 7.0])
        est = np.array([5.01, 5.02, 7.04, 8.0])
        tp, fp, fn, f1 = evaluator.metrics(ref, est)
        self.assertEqual((tp, fp, fn), (2, 2, 1))
        self.assertAlmostEqual(f1, 4 / 7)
        self.assertEqual(evaluator.metrics(ref, np.array([])), (0, 0, 3, 0.0))

    def test_candidate_selects_maximum_and_shortest_lag_on_tie(self):
        onset = '0.032,0.1\n0.042,0.2\n'
        corr = '25,240,0.4,1\n50,120,0.8,1\n51,117.6,0.8,1\n60,100,0.9,0\n'
        with patch.object(evaluator, 'run', side_effect=[onset, corr]):
            self.assertEqual(evaluator.strongest_candidate(Path('.'), Path('x')), 120.0)
        with patch.object(evaluator, 'run', side_effect=[onset, '']):
            self.assertIsNone(evaluator.strongest_candidate(Path('.'), Path('x')))

    def test_prediction_trim_and_abstention(self):
        with patch.object(evaluator, 'run', return_value='4.9\n5.0\n5.1\n'):
            np.testing.assert_array_equal(evaluator.predict(Path('.'), Path('x'), 120), [5.0, 5.1])
        with patch.object(evaluator, 'run') as run:
            self.assertEqual(len(evaluator.predict(Path('.'), Path('x'), None)), 0)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
