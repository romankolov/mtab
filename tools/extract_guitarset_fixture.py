#!/usr/bin/env python3
"""Recreate the CC BY 4.0 five-second GuitarSet audio + note fixture.

Inputs are the *original* version 1.1.0 audio and JAMS files, extracted from
GuitarSet's Zenodo archives. Python is needed only to regenerate the fixture,
not to run mtab. See the fixture README for attribution and source URLs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import wave

AUDIO_SHA256 = "2a6057d0485d8ee2f4a04175ee085d610c0f7a13194e083b13ce803bec091d43"
JAMS_SHA256 = "2258ef6f5dc58437f2807dfc0c05416ad8498cd345f95d6563b13d19283e6d54"


def checked(path, expected):
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"unexpected SHA-256 for {path}: {actual}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    checked(args.audio, AUDIO_SHA256)
    annotation = json.loads(checked(args.annotation, JAMS_SHA256))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.audio), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (44100, 1, 2):
            raise ValueError("expected 44.1 kHz mono 16-bit PCM")
        samples = source.readframes(5 * source.getframerate())
        with wave.open(str(args.output_dir / "guitarset_solo_5s.wav"), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(44100)
            out.writeframes(samples)
    annotation["file_metadata"]["duration"] = 5.0
    annotation["file_metadata"]["title"] += "_first5s"
    notes = [a for a in annotation["annotations"] if a.get("namespace") == "note_midi"]
    if len(notes) != 6:
        raise ValueError("expected six per-string note_midi annotations")
    for string in notes:
        string["time"] = 0.0
        string["duration"] = 5.0
        string["data"] = [event for event in string["data"] if 0.0 <= event["time"] < 5.0]
        for event in string["data"]:
            event["duration"] = min(event["duration"], 5.0 - event["time"])
    annotation["annotations"] = notes
    out = args.output_dir / "guitarset_solo_5s.jams"
    out.write_text(json.dumps(annotation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote 5 seconds and {sum(len(a['data']) for a in notes)} reference notes to {args.output_dir}")


if __name__ == "__main__":
    main()
