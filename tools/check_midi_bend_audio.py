#!/usr/bin/env python3
"""Check an offline FluidSynth render of the known MIDI bend fixture.

Dev/CI-only: requires numpy; no Python, synth or soundfont at mtab runtime.
Render `mtab-export`'s `midi_bend_fixture` using FluidSynth + TimGM6mb, with
chorus/reverb disabled, before running this. It tests one renderer/soundfont,
not general DAW compatibility or detection of real guitar bends.
"""
import argparse
import wave
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('audio')
args = parser.parse_args()
with wave.open(args.audio, 'rb') as wav:
    if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (44100, 2, 2):
        raise ValueError('expected 44.1 kHz stereo 16-bit PCM render')
    rate = wav.getframerate()
    samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').reshape(-1, 2).mean(axis=1) / 32768.0
if len(samples) < rate:
    raise ValueError('render is shorter than the one-second fixture')
base_hz = 440.0 * 2 ** (-9.0 / 12.0)  # MIDI 60
nfft = 262144
freqs = np.fft.rfftfreq(nfft, 1.0 / rate)
grid = np.linspace(250.0, 290.0, 4001)
measured = []
for time in [0.1, 0.3, 0.5, 0.7, 0.9]:
    window = samples[int((time - 0.04) * rate):int((time + 0.04) * rate)]
    rms = np.sqrt(np.mean(window * window))
    if rms < 0.0001:
        raise ValueError(f'render is silent at {time:.1f} s')
    spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window)), n=nfft))
    score = sum(np.interp(grid * harmonic, freqs, spectrum) / harmonic for harmonic in range(1, 5))
    hz = grid[np.argmax(score)]
    cents = float(1200 * np.log2(hz / base_hz))
    expected = 200 * min(time, 1.0-time)  # 0→100→0 cents in one second
    print(f'{time:.1f}s: {cents:.1f} cents, expected {expected:.1f}, error {cents-expected:+.1f}')
    if abs(cents - expected) > 20.0:
        raise AssertionError(f'pitch contour differs by more than 20 cents at {time:.1f} s')
    measured.append(cents)
if not (measured[0] < measured[1] < measured[2] > measured[3] > measured[4]):
    raise AssertionError('render does not rise and fall in pitch')
print('FluidSynth synthetic bend/release render passed')
