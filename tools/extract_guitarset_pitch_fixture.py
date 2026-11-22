#!/usr/bin/env python3
"""Extract five seconds of per-string F0 annotations from the licensed JAMS.

The input must be the original verified GuitarSet 1.1.0 JAMS for
00_BN1-129-Eb_solo; it is not downloaded by this tool. Data are CC BY 4.0;
see crates/mtab-cli/tests/fixtures/guitarset/README.md.
"""
import argparse
import hashlib
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('source', type=pathlib.Path)
parser.add_argument('output', type=pathlib.Path)
args = parser.parse_args()
raw = args.source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '2258ef6f5dc58437f2807dfc0c05416ad8498cd345f95d6563b13d19283e6d54'
doc = json.loads(raw)
strings = [[] for _ in range(6)]
for annotation in doc['annotations']:
    if annotation['namespace'] != 'pitch_contour':
        continue
    index = int(annotation['annotation_metadata']['data_source'])
    data = annotation['data']
    assert len(data['time']) == len(data['value'])
    strings[index] = [[time, value['frequency']] for time, value in zip(data['time'], data['value'])
                      if 0 <= time < 5 and value['voiced'] and value['frequency'] > 0]
assert len(strings) == 6 and sum(map(len, strings)) > 0
args.output.write_text(json.dumps({'strings': strings}, separators=(',', ':')) + '\n', encoding='utf-8')
