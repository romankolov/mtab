#!/usr/bin/env python3
"""Build a GuitarSet end-to-end evaluation manifest from raw mono-mic audio.

This is the corpus-preparation step for §6.14.3: it pairs each excerpt with its
JAMS annotation, writes a trimmed WAV of a fixed length, and emits the JSON
manifest `mtab eval --audio-manifest` consumes.

Two properties matter more than convenience here:

**A fixed excerpt window keeps the evaluation comparable across models and
costs.** GuitarSet recordings are one to three minutes; transcribing all 360 in
full is hours of CPU inference and would make the harness impractical to re-run.
Each excerpt starts at the first annotated onset so the window contains music
rather than leading silence, and the annotations are clipped to the same window.

**The split must be explicit and by recording session.** GuitarSet's six
sessions are six different players. Tuning anything on one set of players and
reporting on another is the only way the number means something, so the caller
picks sessions explicitly and cannot accidentally evaluate on what it tuned on.

Dev-only: `mtab` itself never runs Python, never downloads, and never needs this.
"""

import argparse
import hashlib
import json
import wave
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def first_onset(jams_path):
    """Earliest annotated note onset, or None when the excerpt has no notes."""
    import json as _json

    data = _json.loads(jams_path.read_text())
    earliest = None
    for annotation in data.get("annotations", []):
        if annotation.get("namespace") != "note_midi":
            continue
        for value in annotation.get("data", []):
            time = value.get("time")
            if isinstance(time, (int, float)) and time >= 0:
                earliest = time if earliest is None else min(earliest, time)
    return earliest


def read_wav(path):
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        frames = source.readframes(params.nframes)
    if params.sampwidth != 2:
        raise SystemExit(f"{path}: expected 16-bit PCM, found {params.sampwidth * 8}-bit")
    return params, frames


def write_wav(path, params, frames):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        out.writeframes(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path, help="dir of *_mic.wav")
    parser.add_argument("--annotations", required=True, type=Path, help="dir of *.jams")
    parser.add_argument("--out", required=True, type=Path, help="manifest destination")
    parser.add_argument("--clips", required=True, type=Path, help="dir for trimmed WAVs")
    parser.add_argument("--seconds", type=float, default=10.0, help="excerpt length")
    parser.add_argument(
        "--sessions",
        default="",
        help="comma-separated GuitarSet session prefixes to include, e.g. 03,04,05",
    )
    args = parser.parse_args()

    if not 1.0 <= args.seconds <= 120.0:
        parser.error("--seconds must be between 1 and 120")

    wanted = {s.strip() for s in args.sessions.split(",") if s.strip()}
    clips = args.clips.resolve()
    clips.mkdir(parents=True, exist_ok=True)

    import os

    def rel(path):
        """Path relative to the manifest, since that is how mtab resolves them."""
        return os.path.relpath(path.resolve(), args.out.parent.resolve())

    manifest = []
    skipped = {"no_audio": 0, "no_notes": 0, "session": 0, "short": 0}
    for jams in sorted(args.annotations.glob("*.jams")):
        stem = jams.stem
        session = stem.split("_")[0]
        if wanted and session not in wanted:
            skipped["session"] += 1
            continue
        audio = args.audio / f"{stem}_mic.wav"
        if not audio.is_file():
            skipped["no_audio"] += 1
            continue
        onset = first_onset(jams)
        if onset is None:
            skipped["no_notes"] += 1
            continue

        params, frames = read_wav(audio)
        frame_bytes = params.nchannels * params.sampwidth
        start = int(onset * params.framerate)
        count = int(args.seconds * params.framerate)
        if (params.nframes - start) * frame_bytes < count * frame_bytes:
            skipped["short"] += 1
            continue
        trimmed = frames[start * frame_bytes : (start + count) * frame_bytes]

        clip = clips / f"{stem}_{int(args.seconds)}s.wav"
        write_wav(clip, params, trimmed)
        manifest.append(
            {
                "audio": rel(clip),
                "annotation": rel(jams),
                "duration": args.seconds,
                "audio_sha256": sha256(clip),
                "annotation_sha256": sha256(jams),
            }
        )

    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(1 for _ in args.annotations.glob("*.jams"))
    print(f"{len(manifest)} excerpts of {args.seconds:g}s written to {args.out}")
    print(f"  considered {total} annotations; skipped {skipped}")
    if not manifest:
        raise SystemExit("manifest is empty")


if __name__ == "__main__":
    main()
