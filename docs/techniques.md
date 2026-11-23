# Techniques

mtab distinguishes three very different things, and the reader should never
confuse them:

1. **Techniques supplied in the IR** (`TabDocument`) — preserved exactly by
   every exporter, validated against the notation's rules.
2. **Technique suggestions from audio** — opt-in, uncalibrated, marked as
   suggestions in the output.
3. **Techniques nothing infers** — the model head emits notes only.

## The IR

`mtab_guitar::TabNote` carries:

| Field | Meaning |
| --- | --- |
| `string`, `fret` | 0-based string (low string first) and fret relative to the capo |
| `midi` | sounding pitch, which must equal `open + capo + fret` |
| `velocity` | 1–127, real dynamics rather than a constant |
| `bend` | one `PitchBend { start, peak, depth_cents, kind }` contour |
| `technique` | a `Vec<Technique>` of the marks below |
| `tie_from_prev` | the note continues the previous note on the same string |

`Technique` is `Slide { to_fret, kind }`, `HammerOn`, `PullOff`,
`Vibrato { rate_hz, depth_cents }`, `PalmMute`, `DeadNote`, `Harmonic`,
`LetRing`.

## What each export can carry

| Technique | GP5 | ASCII | MusicXML | MIDI |
| --- | --- | --- | --- | --- |
| Dead note | `x` note type | `x` | percussion-style note head | velocity only |
| Palm mute | note flag | `p` | text/technical annotation | not represented |
| Let ring | note flag | `l` | tied/let-ring notation | note length |
| Vibrato | boolean flag (no rate/depth) | `v` | `wavy-line` start/continue/stop | not represented |
| Hammer-on / pull-off | note flag | `h` | `technical/hammer-on` | no separate event |
| Slide | direction and type flags | `s<fret>`, `S<fret>`, `S^<fret>` | `notations/slide` start/stop | pitch bend in MIDI exporter |
| Bend | one of three supported contours | `b`/`r`/`br` with semitones and peak | rejected (no timed bends yet) | pitch wheel |
| Harmonic | rejected by the notation plan | rejected | rejected | pitch only |

GP5 has no field for vibrato rate or depth, only a per-note boolean. MusicXML
writes every note on both the standard and the tab staff, so any count of an
element is twice the note count — tests assert per-staff positions instead.

Because GP5 stores slides as a flag with no target and hammer-ons as a flag on
the second note, the planner validates them against the **adjacent attack on
the same string**: a hammer/pull-off needs a following attack at a different
fret, a slide's `to_fret` must equal the next note's fret, and a dead-note
target is rejected. A technique that cannot be verified is an error, never a
dropped mark.

## Chord symbols

`ChordSymbol { start, end, label }` entries are chart annotations, not sounding
events. The notation plan accepts only the fixed vocabulary — roots C, C#, D,
Eb, E, F, F#, G, Ab, A, Bb, B with suffixes `""`, `m`, `dim`, `7`, `maj7`, `m7`
— and anchors each symbol to the containing beat of the first track. A symbol
after the last beat anchors to the last beat; two symbols on one beat, or
symbols out of order, are rejected. GP5 gets a chord diagram with name, root
and quality but no invented fingering; MusicXML gets `<harmony>`; ASCII gets a
`chd` row.

## Suggestions from audio

`--techniques` runs `mtab_audio::articulation::suggest_articulations`:

- Two 32 ms windows per note: the attack, and the tail 16 ms after it.
- **Dead note**: at least 18 dB attack-to-tail loss *and* an attack centroid
  under 900 Hz.
- **Palm mute**: at least 6 dB loss with a centroid under 1200 Hz.
- Anything else is left open. Notes that do not fit two full windows are
  reported as too short; notes below the silence floor as silent.

Both measures are gain-invariant, which is why they are usable at all, but the
thresholds are fixed by hand, the classifier is untrained and not
per-instrument, window placement follows transcription timing rather than a
detected onset, and on a mix the drums, bass and other instruments share the
same windows. Treat every mark as a suggestion to review.

`--chords` runs `mtab_guitar::suggest_chords`, which labels simultaneous
pitch-class sets from the detected notes and abstains whenever the set is not
exactly one vocabulary chord. On the GuitarSet oracle diagnostic it covered
15.8% of frames with 87.0% precision when it fired; that measures the
vocabulary's abstention more than it measures chord recognition.

`--experimental-bends` runs `mtab_audio::bend_analysis::suggest_bends`, which
tracks F0 inside each isolated note and accepts a coherent bend/release shape of
at least 40 cents within GP5's supported contours. It is MIDI-only, skips
overlapping notes, and its output is not confidence.

## What is not inferred

The MuScriptor head emits note on/off, instrument and (with `--dynamics`)
estimated velocity. It does not emit techniques. Nothing in mtab infers a
harmonic, a vibrato rate or a slide from audio automatically; those must come
in through the IR.

## Where the code lives

- `crates/mtab-guitar/src/tab.rs` — the IR types.
- `crates/mtab-export/src/gp5_plan.rs` — validation and anchoring.
- `crates/mtab-export/src/{gp5,ascii,musicxml,midi}.rs` — per-format encoding.
- `crates/mtab-audio/src/{bend_analysis,dynamics,articulation}.rs` — suggestions.
- `tools/check_*.py` — independent PyGuitarPro, MusicXML XSD and MIDI readers.
