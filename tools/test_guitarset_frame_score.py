#!/usr/bin/env python3
"""Independent brute-force frame-set oracle for `mtab eval --tab-ir`.

Dev/CI only: Python is not an mtab runtime dependency. This implements the
Wiggins–Kim Eq. 8 ratio directly over binary matrices. It checks the project's
specific 10 ms note-event convention, not equivalence to the TabCNN protocol.
"""
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

binary = str(Path(sys.argv[1]).resolve())
opens = [40, 45, 50, 55, 59, 64]
rng = random.Random(9173)


def brute(ref, pred):
    result = dict(pitch_tp=0, pitch_ref=0, pitch_pred=0, tab_tp=0,
                  tab_ref=0, tab_pred=0, wrong=0, paired=0)
    for frame in range(100):
        t = (frame + .5) * .01
        r = {(s, p) for start, end, s, p in ref if start <= t < end}
        p = {(s, p) for start, end, s, p in pred if start <= t < end}
        rp = {pitch for _, pitch in r}
        pp = {pitch for _, pitch in p}
        result['pitch_tp'] += len(rp & pp)
        result['pitch_ref'] += len(rp)
        result['pitch_pred'] += len(pp)
        result['tab_tp'] += len(r & p)
        result['tab_ref'] += len(r)
        result['tab_pred'] += len(p)
        for pitch in rp & pp:
            rr = {s for s, x in r if x == pitch}
            pr = {s for s, x in p if x == pitch}
            result['paired'] += min(len(rr), len(pr))
            result['wrong'] += min(len(rr), len(pr)) - len(rr & pr)
    return result


with tempfile.TemporaryDirectory(prefix='mtab-frame-oracle-') as temp:
    root = Path(temp)
    for case in range(20):
        def notes():
            out = []
            for _ in range(24):
                s = rng.randrange(6)
                start = rng.randrange(0, 900) / 1000
                end = start + rng.randrange(20, 300) / 1000
                out.append((start, end, s, opens[s] + rng.randrange(0, 15)))
            return out
        ref, pred = notes(), notes()
        # Include a correct unison on two distinct strings and an exact duplicate.
        ref.extend([(0.0, 0.3, 0, 50), (0.0, 0.3, 1, 50)])
        pred.extend([(0.0, 0.3, 0, 50), (0.0, 0.3, 1, 50), (0.0, 0.3, 1, 50)])
        annotations = []
        for string in range(6):
            data = [{'time': start, 'duration': end-start, 'value': pitch}
                    for start, end, s, pitch in ref if s == string]
            annotations.append({'namespace': 'note_midi',
                                'annotation_metadata': {'data_source': str(string)}, 'data': data})
        jams = root/'reference.jams'
        jams.write_text(json.dumps({'annotations': annotations}), encoding='utf-8')
        ir = root/'prediction.json'
        tab_notes = [{'start': start, 'end': end, 'midi': pitch, 'string': string,
                      'fret': pitch-opens[string], 'velocity': 100, 'bend': None,
                      'technique': [], 'tie_from_prev': False}
                     for start, end, string, pitch in pred]
        ir.write_text(json.dumps({'meta': {'title': None, 'tempo_map': {'changes': [{'time': 0.0, 'bpm': 120.0}]},
                                          'time_signatures': [[0, 4, 4]], 'frets': 22, 'source_hash': None},
                                  'tracks': [{'tuning': {'name': 'standard', 'strings': opens},
                                              'capo': 0, 'notes': tab_notes, 'instrument': None}],
                                  'chord_symbols': []}), encoding='utf-8')
        run = subprocess.run([binary, 'eval', '--annotations', str(jams), '--tab-ir', str(ir),
                              '--clip-duration', '1'], text=True, capture_output=True, check=True)
        actual = brute(ref, pred)
        for name, tp, total_r, total_p in [('Multi-pitch frames', 'pitch_tp', 'pitch_ref', 'pitch_pred'),
                                           ('Tab position frames', 'tab_tp', 'tab_ref', 'tab_pred')]:
            match = re.search(r'\| ' + name + r' \| (\d+) \| (\d+) \| (\d+) \|', run.stdout)
            assert match, run.stdout
            found = tuple(map(int, match.groups()))
            want = (actual[tp], actual[total_p]-actual[tp], actual[total_r]-actual[tp])
            assert found == want, (case, name, found, want)
        tdr = re.search(r'TDR-style ratio .*?: (\d+) / (\d+) =', run.stdout)
        wrong = re.search(r'Wrong-string among pitch-matched string instances: (\d+) / (\d+) =', run.stdout)
        assert tdr and tuple(map(int, tdr.groups())) == (actual['tab_tp'], actual['pitch_tp'])
        assert wrong and tuple(map(int, wrong.groups())) == (actual['wrong'], actual['paired'])
print('20 randomized frame-set, unison, duplicate, TDR-ratio and wrong-string cases passed')
