#!/usr/bin/env python3
"""Copy tracked synthetic seeds into the ignored libFuzzer corpora."""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
for kind in ('safetensors', 'audio_decode'):
    dest = root / 'corpus' / kind
    dest.mkdir(parents=True, exist_ok=True)
    for seed in (root / 'seeds' / kind).iterdir():
        shutil.copyfile(seed, dest / seed.name)
print('Seeded', root / 'corpus')
