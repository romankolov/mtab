# M6: limited MuseScore MusicXML test and known data loss

Date: 2026-11-22. Application: Debian MuseScore 3.2.3+dfsg2-19; the actual
application was run headlessly with `-s -m` (no sound or MIDI). Input: the
synthetic two-track `musicxml` integration-test fixture, with six- and
seven-string tunings, capo 2 on track 1, multiple notes, ties, a dead note,
and a palm mute. The input was validated against the MusicXML 3.1 schema.

MuseScore imported the file and exported a new MusicXML file. The re-export
also passed the schema, and music21 independently read the expected pitch,
chord, duration, and tie events on both staves. The generated re-export is
checked in as `crates/mtab-export/tests/fixtures/musescore-3.2.3/fixture.musicxml`;
CI rechecks those pitch/timing semantics and schema. It does **not** run a new
MuseScore export in CI.

**Important failure:** the source put MIDI 45 at string 6, fret 3, capo 2.
MuseScore's re-export puts that pitch at string 5, fret 0; other tablature
assignments also changed, and the output omitted the `capo` and `palm mute`
elements. The dead note's x head was retained in this fixture. The fact that
pitches survive does not mean the tablature or performance does. Users must
review or avoid this exporter for MuseScore if these details matter.

This is a single synthetic fixture, not a universal claim about MuseScore or
other versions. No audio playback, real-model outputs, Guitar Pro or other
MusicXML applications were tested. The source exporter includes the string,
fret and capo values in valid MusicXML, but application preservation remains
an unresolved interoperability problem.

## Reproduce

Generate the input fixture using Python with lxml installed:

```sh
MTAB_XML_PYTHON=python MTAB_XML_FIXTURE_DIR=/tmp/mtab-musicxml \
  cargo test --locked -p mtab-export --test musicxml -- --ignored --nocapture
QT_QPA_PLATFORM=offscreen musescore3 -s -m \
  -o /tmp/musescore-output.musicxml /tmp/mtab-musicxml/fixture.musicxml
python tools/check_musicxml_schema.py /tmp/musescore-output.musicxml
python tools/check_musicxml_music21.py /tmp/musescore-output.musicxml
```

The final check requires `music21==9.7.1`. Inspect the XML `staff-details`,
`technical/string`, `technical/fret`, and `other-technical` fields before and
after to reproduce the documented losses. Python packages are test-only.

## Supplied let-ring markings

The notation planner accepts explicit `Technique::LetRing`. GP5 writes a
let-ring flag on each generated segment; PyGuitarPro 0.10.1 independently
verified flags on synthetic attacks and ties. MusicXML emits a `tied` element
with type `let-ring` on the final segment of each marked note in both staves;
the W3C 3.1 XSD and an independent lxml parser verify those placements. ASCII
uses `l` in its legend. These checks run in CI. This is not inference from
audio, a playback check, or a notation-app round trip. MIDI does not lengthen
the given note-off based on this symbol; application playback varies.
