#!/usr/bin/env python3
"""Prepare a deterministic local GuitarSet 1.1.0 mono-mic evaluation manifest.

Dev-only corpus preparation; no download or Python is required to run mtab.
By default exactly 360 JAMS/audio pairs are required. Use --allow-subset only
when a deliberately selected subset has already been prepared in its own
annotation directory. The user is responsible for documenting selection,
checking dataset provenance, licenses, model settings and held-out splits.
"""
import argparse
import json
from pathlib import Path
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-subset', action='store_true')
    args = parser.parse_args()
    annotation_dir = args.annotations.resolve(strict=True)
    audio_dir = args.audio.resolve(strict=True)
    if not annotation_dir.is_dir() or not audio_dir.is_dir():
        parser.error('annotations and audio must be directories')
    annotations = sorted(annotation_dir.glob('*.jams'))
    if not annotations or (not args.allow_subset and len(annotations) != 360):
        parser.error(f'expected 360 annotations, found {len(annotations)}; use --allow-subset only for an intentional subset')
    output = args.output.resolve()
    manifest = []
    protected = set()
    seen_annotations, seen_audio = set(), set()
    for annotation in annotations:
        audio = audio_dir / f'{annotation.stem}_mic.wav'
        if not audio.is_file():
            parser.error(f'missing matching microphone WAV: {audio}')
        try:
            with wave.open(str(audio), 'rb') as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 44100 or wav.getcomptype() != 'NONE':
                    parser.error(f'expected 44.1 kHz mono 16-bit PCM WAV: {audio}')
                # Leave at least one 16 kHz sample below the nominal duration;
                # this avoids asking for more than the resampler produces and
                # is robust to floating-point conversion at sample boundaries.
                samples = wav.getnframes() * 16000 // wav.getframerate() - 1
                if samples < 1:
                    parser.error(f'WAV is too short: {audio}')
        except (wave.Error, EOFError) as error:
            parser.error(f'invalid WAV {audio}: {error}')
        annotation = annotation.resolve(strict=True)
        audio = audio.resolve(strict=True)
        if annotation in seen_annotations or audio in seen_audio:
            parser.error('duplicate files or symlink aliases detected')
        seen_annotations.add(annotation)
        seen_audio.add(audio)
        protected.update([annotation, audio])
        manifest.append({'audio': str(audio), 'annotation': str(annotation), 'duration': samples / 16000})
    if output in protected:
        parser.error('output must not overwrite an input audio or annotation')
    args.output.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(manifest)} explicit pairs to {args.output}; '
          f'{"subset explicitly allowed" if args.allow_subset else "expected 360"}.')


if __name__ == '__main__':
    main()
