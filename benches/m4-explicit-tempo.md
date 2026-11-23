# Explicit constant tempo across exports

The CLI accepts an optional integer `--bpm` in 4–255; default is 120 BPM.
This is **not audio beat detection**, downbeat estimation, meter detection,
quantization confidence, or rubato tracking. MIDI uses a single tempo event
and 4/4; note timestamps remain based on original seconds, subject to MIDI
tick rounding. GP5, MusicXML and ASCII use the chosen tempo and the existing
fixed 4/4 straight 1/32-note notation planner; boundary shifts are reported.
Specifying the wrong BPM therefore changes notation, not performance speed.

A synthetic fixture built through the CLI document path at 96 BPM was checked
using Mido 1.3.3, PyGuitarPro 0.10.1, the standard-library MusicXML parser and
the emitted IR. Notes at 0–0.3125 s and 0.625–1.25 s were encoded in MIDI at
0–240 and 480–960 ticks, respectively, at 480 PPQ. MIDI recorded 625,000 µs
per quarter, GP5 reported 96 BPM with reconstructed ties/fingerings, MusicXML
recorded 96 BPM, and the ASCII header matched. This checks one synthetic
example, not upstream beat parity or notation in every application. The
independent checks run on GitHub CI; no Python is needed by the application.

## Supplied variable quarter beats

`TempoMap::from_quarter_beat_times` accepts explicitly supplied, strictly
increasing quarter-beat times starting at zero and makes one piecewise-constant
tempo per interval; the final interval's tempo continues afterward. It does
not detect beat phase, first downbeats or meter. MIDI uses an indexed timeline
for timestamp-to-beat mapping and rejects tempo changes that collapse to one
MIDI tick. GP5/MusicXML/ASCII currently reject changing tempos. The MIDI tick
grid and integer-microsecond tempo events necessarily introduce rounding; the
exactness of one carefully chosen fixture does not imply sample-accurate MIDI
for arbitrary times or long maps.

The synthetic `midi_variable_tempo_fixture` uses beats at 0, 0.5, 1.1, 1.8,
and 2.3 s. Mido independently verified tempo events at ticks 0, 480, 960 and
1440 (500,000; 600,000; 700,000; 500,000 µs/quarter), note ticks and velocities,
and the reconstructed seconds. CI repeats this without a model checkpoint.
The CLI also accepts this map via `--beat-times beats.json` for MIDI-only
output. A CLI document-path test feeds the same synthetic notes and supplied
beats into MIDI and IR and runs the independent Mido check; a separate tiny
synthetic-model test exercises audio through transcription to MIDI and IR.
Neither test is a real-checkpoint or audio beat-tracking validation. The input
must begin at zero; 4/4 is assumed and the last tempo is extrapolated.
