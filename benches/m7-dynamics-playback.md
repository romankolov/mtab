# Synthetic MIDI dynamics playback sanity check

This is **not** a real-guitar transcription or a MIDI-velocity calibration
benchmark. The `midi_dynamics_fixture` example writes three isolated MIDI 45
notes at 0.2, 0.8 and 1.4 s, each 0.3 s long, with velocities **40, 80, 120**.
Mido independently checks MIDI note-on/off ticks, velocities, tempo and meter;
it also writes a control with all three note-on velocities set to 80. FluidSynth
renders the MIDI to audio with reverb and chorus disabled. The Rust audio loader
mixes and resamples it to 16 kHz, and the experimental `estimate_velocities`
measures 50 ms attack RMS relative to a two-second local RMS window.

A local Linux run with FluidSynth 2.4.4 at 44.1 kHz produced:

| Soundfont and fixture | Input velocities | Estimated velocities |
| --- | --- | --- |
| TimGM6mb, rising | 40, 80, 120 | 76, 101, 127 |
| FluidR3_GM, rising | 40, 80, 120 | 67, 90, 119 |
| TimGM6mb, flat control | 80, 80, 80 | 114, 113, 121 |

Thus the estimator ranked this one rising sequence correctly but is **not
calibrated**: both soundfont choice and local-window context alter its values,
and even equal input velocities received different estimates. CI tests the
TimGM6mb rising sequence and requires the flat control not to yield the same
strictly increasing ordering. This is not a statistical false-positive test;
other constant-level recordings can still be misclassified as dynamic.

Reproduce with separately installed FluidSynth and TimGM6mb, plus mido 1.3.3
(test-only):

```sh
cargo run --locked -p mtab-export --example midi_dynamics_fixture -- /tmp/dynamics.mid
python tools/check_midi_dynamics_mido.py /tmp/dynamics.mid /tmp/dynamics-flat.mid
for mode in dynamics dynamics-flat; do
  fluidsynth -ni -o synth.reverb.active=0 -o synth.chorus.active=0 \
    -F "/tmp/$mode.wav" -r 44100 /usr/share/sounds/sf2/TimGM6mb.sf2 "/tmp/$mode.mid"
done
cargo run --locked -p mtab-audio --example check_dynamics_render -- /tmp/dynamics.wav rising
cargo run --locked -p mtab-audio --example check_dynamics_render -- /tmp/dynamics-flat.wav flat
```

Soundfont hashes, license notes and render limitations are documented in
[m7-midi-bend-playback.md](m7-midi-bend-playback.md) and `NOTICE`. No audio,
SoundFonts or model weights are bundled. This test has no model inference,
polyphony, real guitar recording, instrument separation or listener evaluation.
