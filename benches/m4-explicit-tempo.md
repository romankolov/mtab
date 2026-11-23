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
