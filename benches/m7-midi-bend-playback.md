# Synthetic MIDI bend/release playback sanity check

The `midi_bend_fixture` example emits one 1-second MIDI 60 note at 120 BPM,
with a supplied bend/release 0 → 100 → 0 cents and an explicit ±2-semitone
pitch-bend RPN. The independent mido check verifies event timing and the wheel
ramp; this additional check renders that file with FluidSynth and estimates
pitch independently from the rendered waveform. The estimator uses 80 ms
Hann windows, a zero-padded FFT, and a weighted sum of four harmonics around
the expected fundamental. It checks five timestamps within 20 cents of the
ideal linear contour; it is a coarse playback sanity check, not ground-truth
pitch measurement of arbitrary instruments or a DAW round-trip.

Local run on Linux: FluidSynth **2.4.4**, 44.1 kHz stereo, reverb/chorus off.

| Soundfont | SHA-256 | Estimated cents at 0.1, 0.3, 0.5, 0.7, 0.9 s |
| --- | --- | --- |
| TimGM6mb | `c5378b62028c920cb11e4803327983fee2f2cdff5dc89c708e39da417e51c854` | 18.7, 58.9, 98.8, 64.3, 24.1 |
| FluidR3_GM | `74594e8f4250680adf590507a306655a299935343583256f3b722c48a1bc1cb0` | 16.4, 55.1, 94.5, 60.0, 19.8 |
| Ideal | — | 20, 60, 100, 60, 20 |

Removing all pitch-wheel messages while preserving MIDI deltas caused the
TimGM6mb render to fail the pitch-contour check at 0.3 s (measured 2.6 cents
against 60 expected), as intended. CI installs FluidSynth and TimGM6mb
separately and checks both the bend and flat negative control. Neither the
rendered audio nor soundfonts are included in the repository or releases.
FluidSynth is LGPL-2.1-or-later; Debian identifies TimGM6mb as GPL-2 and
FluidR3_GM as MIT. See `NOTICE` and each upstream package's licensing.

Reproduce with Python + `numpy`, FluidSynth and a separately installed
soundfont:

```sh
cargo run --locked -p mtab-export --example midi_bend_fixture -- /tmp/bend.mid
fluidsynth -ni -o synth.reverb.active=0 -o synth.chorus.active=0 \
  -F /tmp/bend.wav -r 44100 /usr/share/sounds/sf2/TimGM6mb.sf2 /tmp/bend.mid
python tools/check_midi_bend_audio.py /tmp/bend.wav
```

No real model, recorded guitar bend, subjective listening test, DAW import, or
playback compatibility with other synthesizers is implied by this fixture.
