# M6: limited TuxGuitar GP5 interoperability check

Date: 2026-11-22. Environment: Debian TuxGuitar 1.6.6+dfsg1-3, Java 21,
virtual X display; PyGuitarPro 0.10.1 for independent comparison. This checks
two synthetic library-IR fixtures, **not** audio transcription, a benchmark,
all GP5 features, or Guitar Pro itself.

1. Opened `bends.gp5` in the actual TuxGuitar GUI. It displayed 1/2, 1, and
   1/2 whole-tone bend labels for the source's 100, 200, and 100 cents; the
   first note also displayed P.M. A previous writer used half the correct
   GP5 point values and produced 1/4, 1/2, 1/4; this app check caught the bug.
   See [bend display](images/m6-tuxguitar-bends.png).
2. Opened `techniques.gp5` in the GUI. Its first track showed dead-note x marks,
   palm mutes and tied segments; the document contained both six- and
   seven-string tracks. See [technique display](images/m6-tuxguitar-techniques.png).
3. Used File → Export → Export Guitar Pro 5 in TuxGuitar for both documents.
   Parsed original and re-exported files using PyGuitarPro and compared title,
   tempo, track names, capo, string tunings, measure and first-voice beat
   timing/durations, string/fret, note types, velocities, palm-mute flags,
   and bend control points. Both comparisons passed. TuxGuitar materialized a
   silent second voice; the checker permits this only if it contains no notes.
   TuxGuitar also normalized bend preset types and bend header values; it
   preserved the tested bend points, which the checker compares directly.

**Not tested:** playback (the sandbox had no available MIDI audio line),
Guitar Pro, other TuxGuitar versions/platforms, arbitrary scores, audible bend
interpolation, and files generated from real model inference. GP5 remains an
experimental restricted encoder; application round-trip is not evidence that
transcription quality meets the project's GuitarSet target.

## Reproduce fixtures and compare an app re-export

Install `PyGuitarPro==0.10.1`, then from the repository root:

```sh
MTAB_GP5_FIXTURE_DIR=/tmp/mtab-gp5-fixtures \
MTAB_PYGUITARPRO_PYTHON=python \
cargo test --locked -p mtab-export --test gp5_roundtrip -- --ignored --nocapture
```

For each `bends.gp5` and `techniques.gp5`, open it in TuxGuitar and use File →
Export → Export Guitar Pro 5 to a **different** file. Then compare, for example:

```sh
python tools/check_gp5_tuxguitar_roundtrip.py \
    /tmp/mtab-gp5-fixtures/bends.gp5 /tmp/bends-tuxguitar.gp5
python tools/check_gp5_tuxguitar_roundtrip.py \
    /tmp/mtab-gp5-fixtures/techniques.gp5 /tmp/techniques-tuxguitar.gp5
```

The two TuxGuitar re-exports are checked in under
`crates/mtab-export/tests/fixtures/tuxguitar-1.6.6/`. GitHub Actions compares
freshly generated fixtures against these historical app outputs. The GUI
interaction and fresh app re-export described here are manual and are **not**
part of CI.
