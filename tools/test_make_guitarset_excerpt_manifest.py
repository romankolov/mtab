#!/usr/bin/env python3
"""Offline tests for the GuitarSet excerpt manifest helper.

The regression this guards against: the first version trimmed audio at the first
annotated onset but paired the clip with the *untrimmed* JAMS file. Every clip
then had its own t=0 while its annotation still counted from the recording's
t=0, so an excerpt with a lead-in scored near zero for correctly timed notes
(4.4% instead of 88.9% on one GuitarSet solo excerpt, and 24 of 40 excerpts had
a lead-in). These tests fail on that version.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

SCRIPT = Path(__file__).with_name('make_guitarset_excerpt_manifest.py')
RATE = 8000

# 1.5 s of real lead-in, then a two-note chord, a melody note, a note that runs
# past the excerpt window, and one far outside it. Pitches respect standard
# tuning's 0-22 fret range for the string they are on.
NOTES = [
    (1.5, 0.5, 64, 5),
    (1.5, 0.5, 60, 4),
    (1.75, 0.25, 62, 3),
    (1.9, 5.0, 67, 2),
    (9.0, 1.0, 55, 3),
]


def write_audio(path, seconds):
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(b'\x00\x00' * int(seconds * RATE))


def write_jams(path, notes):
    annotations = []
    for string in range(6):
        data = [
            {'time': time, 'duration': duration, 'value': float(midi)}
            for time, duration, midi, on_string in notes
            if on_string == string
        ]
        annotations.append(
            {
                'namespace': 'note_midi',
                'data': data,
                'annotation_metadata': {'data_source': str(string)},
            }
        )
    # A namespace mtab does not read, to prove it is left alone.
    annotations.append({'namespace': 'beat_position', 'data': {'x': 1}})
    path.write_text(json.dumps({'annotations': annotations}))


def note_table(jams_path):
    data = json.loads(Path(jams_path).read_text())
    rows = []
    for annotation in data['annotations']:
        if annotation['namespace'] != 'note_midi':
            continue
        for value in annotation['data']:
            rows.append((round(value['time'], 6), round(value['duration'], 6)))
    return sorted(rows)


class ExcerptManifestTest(unittest.TestCase):
    def build(self, root, seconds=2.0):
        root = Path(root)
        (root / 'annotations').mkdir()
        (root / 'audio').mkdir()
        (root / 'eval').mkdir()
        write_jams(root / 'annotations' / '03_take.jams', NOTES)
        write_audio(root / 'audio' / '03_take_mic.wav', 12.0)
        manifest = root / 'eval' / 'excerpt.json'
        command = [
            sys.executable, str(SCRIPT),
            '--audio', str(root / 'audio'),
            '--annotations', str(root / 'annotations'),
            '--out', str(manifest),
            '--clips', str(root / 'eval' / 'clips'),
            '--seconds', str(seconds),
            '--sessions', '03',
        ]
        return manifest, subprocess.run(command, capture_output=True, text=True)

    def test_annotation_is_rebased_onto_the_trimmed_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, result = self.build(tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            entries = json.loads(Path(manifest).read_text())
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            annotation = Path(manifest).parent / entry['annotation']
            # The clip starts at the first onset (1.5 s), so that chord is at 0 s
            # in the excerpt. Leaving it at 1.5 s is exactly the bug.
            self.assertEqual(
                note_table(annotation),
                [(0.0, 0.5), (0.0, 0.5), (0.25, 0.25), (0.4, 1.6)],
            )
            self.assertEqual(entry['duration'], 2.0)
            self.assertEqual(entry['annotation_sha256'],
                             hashlib.sha256(annotation.read_bytes()).hexdigest())
            original = Path(tmp) / 'annotations' / '03_take.jams'
            self.assertNotEqual(entry['annotation_sha256'],
                                hashlib.sha256(original.read_bytes()).hexdigest())
            with wave.open(str(Path(manifest).parent / entry['audio'])) as handle:
                self.assertEqual(handle.getnframes(), 2 * RATE)

    def test_long_note_is_clipped_to_the_excerpt_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, result = self.build(tmp, seconds=1.75)
            self.assertEqual(result.returncode, 0, result.stderr)
            entry = json.loads(Path(manifest).read_text())[0]
            annotation = Path(manifest).parent / entry['annotation']
            tables = note_table(annotation)
            # 1.9 s - 1.5 s = 0.4 s onset, clipped at the 1.75 s window end.
            self.assertIn((0.4, 1.35), tables)
            self.assertTrue(all(time < 1.75 for time, _ in tables), tables)
            self.assertTrue(all(time + length <= 1.75 + 1e-9 for time, length in tables),
                            tables)

    def test_notes_outside_the_window_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = self.build(tmp)
            entry = json.loads(Path(manifest).read_text())[0]
            annotation = Path(manifest).parent / entry['annotation']
            # Nothing may survive at its original 9 s placement.
            self.assertTrue(all(time < 2.0 for time, _ in note_table(annotation)))

    def test_untouched_namespaces_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = self.build(tmp)
            entry = json.loads(Path(manifest).read_text())[0]
            data = json.loads((Path(manifest).parent / entry['annotation']).read_text())
            beat = [a for a in data['annotations'] if a['namespace'] == 'beat_position']
            self.assertEqual(beat, [{'namespace': 'beat_position', 'data': {'x': 1}}])

    def test_rebase_volume_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, result = self.build(tmp)
            # Four notes survive the 2 s window; the 9 s one is dropped. The
            # straddling note is clipped, not dropped, so it is not counted.
            self.assertIn('note times rebased onto each excerpt: 4 kept, 1 outside',
                          result.stdout)


if __name__ == '__main__':
    unittest.main()
