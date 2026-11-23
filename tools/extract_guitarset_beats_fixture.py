#!/usr/bin/env python3
"""Extract metronome-derived beats for the documented GuitarSet 5s fixture."""
import hashlib
import json
import sys
from pathlib import Path

EXPECTED = '2258ef6f5dc58437f2807dfc0c05416ad8498cd345f95d6563b13d19283e6d54'
source = Path(sys.argv[1]).read_bytes()
if hashlib.sha256(source).hexdigest() != EXPECTED:
    raise SystemExit('unexpected source JAMS SHA-256')
data = json.loads(source)
annotations = [a for a in data['annotations'] if a['namespace'] == 'beat_position']
assert len(annotations) == 1
beats = [
    {'time': b['time'], 'position': b['value']['position'],
     'measure': b['value']['measure'], 'num_beats': b['value']['num_beats'],
     'beat_units': b['value']['beat_units']}
    for b in annotations[0]['data'] if 0 <= b['time'] < 5
]
assert len(beats) == 11 and all(b['num_beats'] == 4 and b['beat_units'] == 4 for b in beats)
out = {'source_jams_sha256': EXPECTED, 'duration_seconds': 5, 'beats': beats}
Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8', newline='\n')
