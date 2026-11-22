#!/usr/bin/env python3
"""Development-only independent validation of the mtab GP5 round-trip fixture."""
import sys
import guitarpro

song = guitarpro.parse(sys.argv[1])
assert song.versionTuple == (5, 0, 0), song.versionTuple
assert song.title == "GP5 roundtrip", song.title
assert song.tempo == 120, song.tempo
assert len(song.tracks) == 2
assert len(song.measureHeaders) == 2
tracks = song.tracks
assert [s.value for s in tracks[0].strings] == [64, 59, 55, 50, 45, 40]
assert tracks[0].offset == 2
assert [s.value for s in tracks[1].strings] == [64, 59, 55, 50, 45, 40, 35]
assert tracks[1].offset == 0
assert [t.channel.instrument for t in tracks] == [25, 30]
for t in tracks:
    assert len(t.measures) == 2
    assert all(m.timeSignature.numerator == 4 and m.timeSignature.denominator.value == 4 for m in t.measures)

def notes(track):
    result = []
    for measure in track.measures:
        for beat in measure.voices[0].beats:
            for note in beat.notes:
                result.append((beat.start - 960, beat.duration.time, note.string, note.value, note.velocity, note.type.name))
    return result

actual = [notes(t) for t in tracks]
expected = [
    [(0, 960, 6, 3, 79, 'normal'), (960, 960, 5, 2, 111, 'normal'), (960, 960, 6, 3, 79, 'tie'),
     (1920, 1920, 6, 3, 79, 'tie'), (3840, 960, 4, 0, 127, 'normal'), (3840, 960, 6, 3, 79, 'tie')],
    [(960, 960, 7, 1, 47, 'normal'), (1920, 1920, 7, 1, 47, 'tie'), (3840, 960, 7, 1, 47, 'tie')],
]
assert actual == expected, f"notes do not match:\nactual={actual}\nexpected={expected}"
print("PyGuitarPro verified title, tempo, meters, tunings, capo, programs, attacks, ties and dynamics")
