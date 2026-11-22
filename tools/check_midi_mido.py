#!/usr/bin/env python3
"""Independent reader check for preliminary CLI MIDI against its JSONL.

Test/development only: ``python -m pip install mido==1.3.3``. Given separate
runs with identical inputs/settings, compare every note onset, offset, pitch,
and velocity; additionally check the fixed tempo/meter and SMF type. This is
not a DAW playback test or a measure of transcription accuracy.
"""
import argparse
import json
import math
from collections import Counter

import mido


def tick(seconds):
    # Current CLI MIDI uses 120 BPM and 480 ticks per quarter. Inputs are
    # nonnegative, so floor(x + .5) matches Rust's f64::round.
    return math.floor(seconds * 960 + 0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help="raw JSONL from mtab transcribe")
    parser.add_argument("midi", help="MIDI from the same input/seed/options")
    args = parser.parse_args()
    with open(args.jsonl, encoding="utf-8") as file:
        notes = [json.loads(line) for line in file if line.strip()]
    smf = mido.MidiFile(args.midi)
    if smf.type != 1 or smf.ticks_per_beat != 480:
        raise ValueError("expected SMF type 1 with 480 ticks per beat")
    expected = Counter()
    for note in notes:
        onset = tick(note["start"])
        end = max(tick(note["end"]), onset + 1)
        expected[(onset, note["midi"], note["velocity"], "on")] += 1
        expected[(end, note["midi"], 0, "off")] += 1
    actual = Counter()
    tempos = []
    meters = []
    for track in smf.tracks:
        absolute = 0
        for msg in track:
            absolute += msg.time
            if msg.type == "set_tempo":
                tempos.append((absolute, msg.tempo))
            elif msg.type == "time_signature":
                meters.append((absolute, msg.numerator, msg.denominator))
            elif msg.type == "note_on" and msg.velocity:
                actual[(absolute, msg.note, msg.velocity, "on")] += 1
            elif msg.type == "note_off" or (msg.type == "note_on" and not msg.velocity):
                actual[(absolute, msg.note, 0, "off")] += 1
    if tempos != [(0, 500000)] or meters != [(0, 4, 4)]:
        raise ValueError(f"unexpected tempo/meter: {tempos}, {meters}")
    if actual != expected:
        raise ValueError(f"MIDI event mismatch; missing {expected - actual}, unexpected {actual - expected}")
    print(f"Mido read SMF1: {len(smf.tracks)} tracks, {len(notes)} matching notes, 120 BPM, 4/4")


if __name__ == "__main__":
    main()
