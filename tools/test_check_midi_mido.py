#!/usr/bin/env python3
"""Unit tests for the optional independent MIDI checker; not a runtime dependency."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mido


class CheckerTest(unittest.TestCase):
    def test_valid_and_invalid_events(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            notes = root / "notes.jsonl"
            midi = root / "notes.mid"
            notes.write_text(json.dumps({"start": 0.5, "end": 1.0, "midi": 64, "velocity": 89, "instrument": 24}) + "\n", encoding="utf-8")
            smf = mido.MidiFile(type=1, ticks_per_beat=480)
            meta = mido.MidiTrack()
            meta.append(mido.MetaMessage("set_tempo", tempo=500000))
            meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
            track = mido.MidiTrack()
            track.append(mido.Message("note_on", note=64, velocity=89, time=480))
            track.append(mido.Message("note_off", note=64, velocity=0, time=480))
            smf.tracks.extend([meta, track])
            checker = Path(__file__).with_name("check_midi_mido.py")
            def run():
                return subprocess.run([sys.executable, str(checker), str(notes), str(midi)], capture_output=True, text=True)
            smf.save(midi)
            result = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("1 matching notes", result.stdout)
            track[0].velocity = 88
            smf.save(midi)
            result = run()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("MIDI event mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
