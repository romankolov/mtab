#!/usr/bin/env python3
"""Offline tests for the GuitarSet manifest preparation helper."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

SCRIPT = Path(__file__).with_name('make_guitarset_audio_manifest.py')


class ManifestTest(unittest.TestCase):
    def test_explicit_subset_pairing_duration_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root/'annotations'
            audio = root/'audio'
            annotations.mkdir()
            audio.mkdir()
            jams = annotations/'clip.jams'
            jams.write_text('{}', encoding='utf-8')
            wav = audio/'clip_mic.wav'
            with wave.open(str(wav), 'wb') as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(44100)
                out.writeframes(b'\x00\x00' * 44100)
            manifest = root/'manifest.json'
            command = [sys.executable, str(SCRIPT), '--annotations', str(annotations),
                       '--audio', str(audio), '--output', str(manifest)]
            def run(extra=()):
                return subprocess.run(command + list(extra), capture_output=True, text=True)
            self.assertNotEqual(run().returncode, 0, 'a subset requires explicit consent')
            result = run(['--allow-subset'])
            self.assertEqual(result.returncode, 0, result.stderr)
            pairs = json.loads(manifest.read_text())
            self.assertEqual(len(pairs), 1)
            self.assertEqual(Path(pairs[0]['audio']), wav.resolve())
            self.assertEqual(Path(pairs[0]['annotation']), jams.resolve())
            self.assertEqual(pairs[0]['duration'], 15999/16000)
            protected = [sys.executable, str(SCRIPT), '--annotations', str(annotations),
                         '--audio', str(audio), '--output', str(jams), '--allow-subset']
            original = jams.read_bytes()
            self.assertNotEqual(subprocess.run(protected, capture_output=True).returncode, 0)
            self.assertEqual(jams.read_bytes(), original)
            (annotations/'missing.jams').write_text('{}', encoding='utf-8')
            previous = manifest.read_bytes()
            result = run(['--allow-subset'])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing matching microphone WAV', result.stderr)
            self.assertEqual(manifest.read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()
