# Experimental spectral-flux onset envelope

`mtab_audio::onset::spectral_flux` is a preliminary library building block,
**not a beat tracker or onset-event detector**, and is not connected to CLI
output. It consumes finite mono 16 kHz audio. Full 1024-sample periodic-Hann
windows start every 160 samples (10 ms); timestamps denote window centers,
starting at 32 ms. For each one-sided FFT bin it computes
`ln(1 + 1000 * magnitude / 1024)`, then averages positive bin differences
from the previous window. The first frame is defined as zero. The result is
nonnegative but unnormalized and not a probability. The function rejects more
than 30,000 analysis frames (~5 min) rather than truncate; the caller still
owns the entire decoded audio buffer. Short clips produce an empty result.

This uses the standard positive-difference spectral-flux idea described in
[librosa's onset-strength documentation](https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_strength.html),
but **does not claim librosa parity**: its features are log-magnitude linear
FFT bins, not librosa's default log-power mel spectrogram. There is no local
frequency-max vibrato filter, gain normalization, adaptive threshold, peak
picking, tempo inference, phase estimate, meter or downbeat tracking.
Mixed instruments, vibrato, noise, repeated notes, amplitude and boundary
conditions can change the result; first-window attacks are not recoverable by
this definition. Findings must not be interpreted as beat accuracy.

## Independent numerical check

Run from the repository root with Python, NumPy and a Rust toolchain:

```sh
cargo build --locked -p mtab-audio --example onset_dump
python tools/check_spectral_flux_numpy.py target/debug/examples/onset_dump
```

A separately implemented NumPy rFFT compares every output frame on seven
synthetic PCM16 signals (short, silence, tone, impulses, gated tone, noise,
low-amplitude noise) and the checked-in five-second GuitarSet excerpt. The
latter is linearly interpolated from 44.1 to 16 kHz solely to obtain common
PCM input to the two implementations, not to assess resampling quality. On
NumPy 2.3.5, the maximum absolute strength difference was **2.28e-15**; CI
runs the comparison with NumPy 1.26.4 and a tolerance of `1e-12`. This validates
the specified numerical formula, not onset detection quality. The GuitarSet
fixture is documented and attributed in `crates/mtab-cli/tests/fixtures/guitarset/README.md`.

Potential next work: a separately evaluated peak picker and beat tracker,
including negative controls, double/half-tempo ambiguity, time alignment and
held-out real-audio beat annotations. GuitarSet beats are derived from the
recording click track rather than estimating local musician timing; see the
[GuitarSet dataset paper](https://archives.ismir.net/ismir2018/paper/000188.pdf).
