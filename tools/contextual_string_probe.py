#!/usr/bin/env python3
"""Exploratory held-out probe: can local pitch context predict picked guitar string?

This is a diagnostic for a possible learned string-choice prior, not a tab
transcriber and not the project's acceptance benchmark. It reads GuitarSet
JAMS note_midi annotations, trains on sessions 00-02 and tests once on 03-05.
The model sees pitch, duration, chord summaries, and neighboring onset-group
pitch summaries; it never sees the annotated string, file name, or audio as an
input feature. See benches/contextual-string-probe.md for scope and results.

Requires NumPy and scikit-learn; neither is a runtime dependency of mtab.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

OPEN_STRINGS = np.array([40, 45, 50, 55, 59, 64], dtype=np.int16)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_events(path: Path) -> list[tuple[float, float, int, int]]:
    obj = json.loads(path.read_text())
    events = []
    seen: set[int] = set()
    for ann in obj.get("annotations", []):
        if ann.get("namespace") != "note_midi":
            continue
        string = int(ann["annotation_metadata"]["data_source"])
        if string not in range(6) or string in seen:
            raise ValueError(f"{path}: invalid or duplicate string {string}")
        seen.add(string)
        for item in ann["data"]:
            start = float(item["time"])
            duration = float(item["duration"])
            pitch = int(round(float(item["value"])))
            if not (math.isfinite(start) and math.isfinite(duration)) or start < 0 or duration <= 0:
                raise ValueError(f"{path}: malformed note {item}")
            if not 0 <= pitch <= 127:
                raise ValueError(f"{path}: pitch out of MIDI range: {pitch}")
            events.append((start, start + duration, pitch, string))
    if len(seen) != 6:
        raise ValueError(f"{path}: expected six note_midi arrays, found {len(seen)}")
    return sorted(events, key=lambda event: (event[0], event[3], event[1], event[2]))


def feature_rows(events: list[tuple[float, float, int, int]]) -> tuple[list[list[float]], list[int]]:
    """Features for each note, using 30 ms onset groups to summarize context."""
    groups: list[list[object]] = []  # [first onset, events]
    for event in events:
        if not groups or event[0] - float(groups[-1][0]) > 0.03:
            groups.append([event[0], []])
        groups[-1][1].append(event)  # type: ignore[union-attr]

    # time, mean pitch, min pitch, max pitch, distinct pitches, mean duration
    summaries: list[tuple[float, float, float, float, int, float]] = []
    for time, items in groups:
        pitches = np.array(sorted({int(e[2]) for e in items}), dtype=np.float64)
        durations = [float(e[1]) - float(e[0]) for e in items]
        summaries.append((
            float(time), float(pitches.mean()), float(pitches.min()),
            float(pitches.max()), int(len(pitches)), float(np.mean(durations)),
        ))

    rows: list[list[float]] = []
    labels: list[int] = []
    for gi, (time, items) in enumerate(groups):
        pitch_set = sorted({int(e[2]) for e in items})
        group_pitch = float(summaries[gi][1])
        group_low = float(summaries[gi][2])
        group_high = float(summaries[gi][3])
        for start, end, pitch, string in items:
            pitch_class = pitch % 12
            row = [
                float(pitch), float(pitch_class),
                math.sin(2 * math.pi * pitch_class / 12),
                math.cos(2 * math.pi * pitch_class / 12),
                math.log1p(end - start), float(len(pitch_set)),
                group_low, group_high, group_pitch,
                float(pitch) - group_pitch, float(pitch) - group_low,
                group_high - float(pitch),
            ]
            for offset in (1, 2, 3):
                for direction in (-1, 1):
                    j = gi + direction * offset
                    if 0 <= j < len(summaries):
                        other = summaries[j]
                        row.extend([
                            other[1], other[1] - pitch, other[2], other[3],
                            other[4], other[5], other[0] - float(time),
                        ])
                    else:
                        row.extend([0.0] * 7)
            rows.append(row)
            labels.append(string)
    return rows, labels


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "correct": int(np.count_nonzero(y_true == y_pred)),
        "notes": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_rows_true_columns_predicted_strings_0_to_5": confusion_matrix(
            y_true, y_pred, labels=list(range(6))
        ).tolist(),
    }


def read_predicted_jsonl(path: Path) -> list[tuple[float, float, int, int]]:
    events = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        start, end, pitch = float(item["start"]), float(item["end"]), int(item["midi"])
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end <= start or not 0 <= pitch <= 127:
            raise ValueError(f"{path}: malformed predicted note {item}")
        # String is a dummy placeholder; the feature model never reads it.
        events.append((start, end, pitch, 0))
    return sorted(events, key=lambda event: (event[0], event[1], event[2]))


def probability_costs(probability: np.ndarray, pitch: int) -> list[float]:
    frets = pitch - OPEN_STRINGS
    legal = (frets >= 0) & (frets <= 22)
    raw_cost = -np.log(np.maximum(probability, 1e-9))
    raw_cost[~legal] = 1000.0
    raw_cost[legal] -= float(raw_cost[legal].min())
    return [float(x) for x in raw_cost]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="directory containing the 360 GuitarSet JAMS files")
    parser.add_argument("--output", type=Path, required=True,
                        help="write machine-readable JSON results here")
    parser.add_argument("--write-test-priors-dir", type=Path,
                        help="optionally write per-note held-out oracle-note costs")
    parser.add_argument("--predicted-transcripts-dir", type=Path,
                        help="optional JSONL directory to score with the same trained model")
    parser.add_argument("--write-predicted-priors-dir", type=Path,
                        help="write per-note costs for predicted JSONL notes")
    parser.add_argument("--train-sessions", default="00,01,02",
                        help="comma-separated training sessions (default: 00,01,02)")
    parser.add_argument("--test-sessions", default="03,04,05",
                        help="comma-separated test sessions (default: 03,04,05)")
    args = parser.parse_args()
    train_sessions = set(args.train_sessions.split(","))
    test_sessions = set(args.test_sessions.split(","))
    if not train_sessions or not test_sessions or train_sessions & test_sessions:
        raise SystemExit("train and test session sets must be nonempty and disjoint")
    if (train_sessions | test_sessions) - {"00", "01", "02", "03", "04", "05"}:
        raise SystemExit("session names must be in 00 through 05")

    paths = sorted(args.annotations.glob("*.jams"))
    if len(paths) != 360:
        raise SystemExit(f"expected 360 .jams files, found {len(paths)}")
    if any(path.stem[:2] not in {"00", "01", "02", "03", "04", "05"} for path in paths):
        raise SystemExit("unexpected session prefix; expected sessions 00 through 05")

    train_x: list[list[float]] = []
    train_y: list[int] = []
    test_x: list[list[float]] = []
    test_y: list[int] = []
    test_session: list[str] = []
    note_counts: dict[str, int] = {}
    hashes = []
    for path in paths:
        rows, labels = feature_rows(read_events(path))
        session = path.stem[:2]
        note_counts[session] = note_counts.get(session, 0) + len(labels)
        hashes.append((path.name, digest(path)))
        if session in train_sessions:
            train_x.extend(rows)
            train_y.extend(labels)
        elif session in test_sessions:
            test_x.extend(rows)
            test_y.extend(labels)
            test_session.extend([session] * len(labels))

    x_train = np.asarray(train_x, dtype=np.float32)
    x_test = np.asarray(test_x, dtype=np.float32)
    y_train = np.asarray(train_y, dtype=np.int8)
    y_test = np.asarray(test_y, dtype=np.int8)
    classifier = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=0,
    )
    classifier.fit(x_train, y_train)
    probabilities = classifier.predict_proba(x_test)
    classes = classifier.classes_.astype(int)

    if args.write_test_priors_dir is not None:
        args.write_test_priors_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if path.stem[:2] not in test_sessions:
                continue
            events = read_events(path)
            rows, _ = feature_rows(events)
            per_note_probability = classifier.predict_proba(np.asarray(rows, dtype=np.float32))
            records = []
            for (start, end, pitch, truth), probability in zip(events, per_note_probability):
                records.append({
                    "start": start,
                    "end": end,
                    "midi": pitch,
                    "reference_string": truth,
                    "costs": probability_costs(probability, pitch),
                })
            (args.write_test_priors_dir / f"{path.stem}.json").write_text(
                json.dumps({"annotation": path.name, "notes": records}, separators=(",", ":")) + "\n"
            )

    if args.predicted_transcripts_dir is not None:
        transcript_paths = sorted(
            args.predicted_transcripts_dir.glob("n*.jsonl"),
            key=lambda path: int(path.stem[1:]),
        )
        if not transcript_paths:
            raise SystemExit("no nN.jsonl predicted transcripts found")
        if args.write_predicted_priors_dir is None:
            raise SystemExit("--predicted-transcripts-dir requires --write-predicted-priors-dir")
        args.write_predicted_priors_dir.mkdir(parents=True, exist_ok=True)
        for path in transcript_paths:
            events = read_predicted_jsonl(path)
            rows, _ = feature_rows(events)
            per_note_probability = classifier.predict_proba(np.asarray(rows, dtype=np.float32))
            output_rows = [
                {"start": start, "end": end, "midi": pitch,
                 "costs": probability_costs(probability, pitch)}
                for (start, end, pitch, _), probability in zip(events, per_note_probability)
            ]
            out_path = args.write_predicted_priors_dir / f"{path.stem}.json"
            out_path.write_text(json.dumps({"transcript": path.name, "notes": output_rows}, separators=(",", ":")) + "\n")

    unconstrained = classes[np.argmax(probabilities, axis=1)].astype(np.int8)
    legal_masked = []
    lowest_fret = []
    for row, probability in zip(x_test, probabilities):
        pitch = int(row[0])
        frets = pitch - OPEN_STRINGS
        legal = np.flatnonzero((frets >= 0) & (frets <= 22))
        if not len(legal):
            legal_masked.append(int(unconstrained[len(legal_masked)]))
            lowest_fret.append(-1)
            continue
        legal_class_indices = [i for i, cls in enumerate(classes) if int(cls) in legal]
        best = max(legal_class_indices, key=lambda i: float(probability[i]))
        legal_masked.append(int(classes[best]))
        lowest_fret.append(int(legal[np.argmin(frets[legal])]))

    result: dict[str, object] = {
        "protocol": "benches/m6-contextual-string-probe.md",
        "train_sessions": sorted(train_sessions),
        "test_sessions": sorted(test_sessions),
        "input_feature_note": "pitch, duration, onset-group pitch summaries, and up to three neighboring onset groups; no string, filename, or audio features",
        "onset_group_tolerance_seconds": 0.03,
        "classifier": "sklearn HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)",
        "files": len(paths),
        "notes_by_session": note_counts,
        "training_notes": len(train_y),
        "test_notes": len(test_y),
        "annotations_sha256_manifest": hashlib.sha256(
            "\n".join(f"{name} {sha}" for name, sha in hashes).encode()
        ).hexdigest(),
        "unconstrained_classifier": score(y_test, unconstrained),
        "classifier_with_0_to_22_fret_legality_mask": score(y_test, np.asarray(legal_masked)),
        "lowest_fret_baseline": score(y_test, np.asarray(lowest_fret)),
        "per_session": {},
    }
    per_session: dict[str, object] = {}
    session_array = np.asarray(test_session)
    for session in sorted(test_sessions):
        mask = session_array == session
        per_session[session] = {
            "notes": int(mask.sum()),
            "classifier_accuracy": float(accuracy_score(y_test[mask], np.asarray(legal_masked)[mask])),
            "lowest_fret_accuracy": float(accuracy_score(y_test[mask], np.asarray(lowest_fret)[mask])),
        }
    result["per_session"] = per_session
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "train_notes": len(train_y),
        "test_notes": len(test_y),
        "classifier_accuracy": result["classifier_with_0_to_22_fret_legality_mask"]["accuracy"],
        "lowest_fret_accuracy": result["lowest_fret_baseline"]["accuracy"],
        "per_session": per_session,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
