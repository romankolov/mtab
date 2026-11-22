#!/usr/bin/env python3
"""Independently cross-check GuitarSet note matching with mir_eval 0.8.2.

Development only: install mir_eval==0.8.2 and its dependencies separately.
Assumes the same 0-based audio time, 50 ms onset and max(50 ms, 20%) offset
thresholds as `mtab eval --predictions`; compares rounded MIDI pitches. This
is not a runtime dependency or a corpus-level/tablature accuracy benchmark.
"""
import argparse
import json
import math

import mir_eval
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", help="one GuitarSet JAMS file")
    parser.add_argument("predictions", help="matching mtab JSONL")
    parser.add_argument("--clip-duration", type=float)
    args = parser.parse_args()
    if args.clip_duration is not None and (not math.isfinite(args.clip_duration) or args.clip_duration <= 0):
        parser.error("--clip-duration must be finite and positive")
    with open(args.annotation, encoding="utf-8") as stream:
        jams = json.load(stream)
    notes = []
    for annotation in jams["annotations"]:
        if annotation["namespace"] != "note_midi":
            continue
        for event in annotation["data"]:
            notes.append((event["time"], event["time"] + event["duration"], math.floor(event["value"] + 0.5)))
    with open(args.predictions, encoding="utf-8") as stream:
        predicted = [(n["start"], n["end"], n["midi"]) for n in map(json.loads, filter(str.strip, stream))]
    if args.clip_duration is not None:
        def clip(items):
            return [(a, min(b, args.clip_duration), p) for a, b, p in items if a < args.clip_duration]
        notes, predicted = clip(notes), clip(predicted)
    notes.sort()
    predicted.sort()
    def arrays(items):
        intervals = np.array([(a, b) for a, b, _ in items], dtype=float).reshape(-1, 2)
        frequencies = np.array([440.0 * 2 ** ((pitch - 69) / 12.0) for _, _, pitch in items], dtype=float)
        return intervals, frequencies
    ref_intervals, ref_hz = arrays(notes)
    est_intervals, est_hz = arrays(predicted)
    for label, offset in [("pitch + onset", None), ("pitch + onset + offset", 0.2)]:
        matches = mir_eval.transcription.match_notes(ref_intervals, ref_hz, est_intervals, est_hz,
            onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=offset, offset_min_tolerance=0.05)
        count = len(matches)
        print(f"{label}: TP={count}, FP={len(predicted)-count}, FN={len(notes)-count}")


if __name__ == "__main__":
    main()
