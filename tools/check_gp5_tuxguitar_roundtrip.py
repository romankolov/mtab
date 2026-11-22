#!/usr/bin/env python3
"""Compare mtab GP5 with TuxGuitar's GP5 re-export. Test-only PyGuitarPro.

Bend preset type/header value are deliberately not compared: TuxGuitar 1.6.6
normalizes these, while retaining the actual pitch/time control points.
"""
import sys
import guitarpro


def snapshot(path):
    song = guitarpro.parse(path)
    tracks = []
    for track in song.tracks:
        measures = []
        for measure in track.measures:
            voices = []
            for voice_index, voice in enumerate(measure.voices):
                if voice_index > 0:
                    assert not any(beat.notes for beat in voice.beats), "unexpected notes in extra voice"
                    continue  # TuxGuitar materializes an empty second voice.
                beats = []
                for beat in voice.beats:
                    notes = []
                    for note in beat.notes:
                        bend = note.effect.bend
                        points = None if bend is None else [
                            (p.position, p.value, p.vibrato) for p in bend.points
                        ]
                        notes.append((note.string, note.value, note.type.name,
                                      note.velocity, note.effect.palmMute, points))
                    beats.append((beat.start, beat.duration.value,
                                  beat.duration.isDotted, notes))
                voices.append(beats)
            measures.append(voices)
        tracks.append((track.name, track.offset,
                       [(s.number, s.value) for s in track.strings], measures))
    return song.title, song.tempo, tracks


before, after = map(snapshot, sys.argv[1:3])
assert before == after, f"round-trip mismatch\nbefore={before!r}\nafter={after!r}"
print("TuxGuitar GP5 re-export preserves tested notes, timing, ties, dynamics, tuning, capo, palm mutes and bend points")
