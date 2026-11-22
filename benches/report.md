# Preliminary microbenchmark

This is **not** a transcription benchmark and does not measure or establish the
real-time-factor, memory or accuracy targets in PROMPT.md §6.13. Those require
real release weights and end-to-end runs, which have not yet been performed.

On 2026-11-22 in a Linux x86-64 sandbox reporting an Intel Xeon 2.60 GHz CPU,
with Rust 1.100.0, `--release`, and `MATMUL_NUM_THREADS=1`, one run of the
batched linear microbenchmark measured:

| Operation | Time |
| --- | ---: |
| f64 GEMM, including f32-to-f64 weight conversion | 38.2 ms |
| Previous four-lane scalar dot-product loop | 149.5 ms |

The matrix dimensions were 256×768 times the transpose of 768×768; the measured
maximum absolute difference was 1.71e-13. This is a single, non-statistical
observation on a shared machine. Full transformer golden and exact token-stream
tests remain the correctness gate. Reproduce with:

```sh
MATMUL_NUM_THREADS=1 cargo run -p mtab-core --release --example bench_linear
```

The GEMM library also supports up to four threads via `MATMUL_NUM_THREADS`;
single-token decoding still uses the scalar path. GPU acceleration and
end-to-end release-checkpoint performance remain unmeasured.

## Full small-architecture execution smoke (synthetic parameters)

On the same Linux x86-64 sandbox, a locally generated 103M-parameter **random,
untrained** checkpoint (393 MiB, 14 layers, 768 dimensions) was run with the
release CLI on a 0.5-second synthetic tone. With `MATMUL_NUM_THREADS=2` and a
limit of two generated tokens, one run took 16.1 seconds including checkpoint
loading and reported peak RSS 789 MiB. It hit the token limit and produced no
notes. The temporary checkpoint was not committed. This checks that the full
small-size architecture executes under the environment's memory constraints;
it does **not** measure real-checkpoint transcription quality, representative
RTF, throughput at full sequence lengths, or the project acceptance targets.

## Official small checkpoint validation (2026-11-22)

The gated official `MuScriptor/muscriptor-small` checkpoint was downloaded after
access approval and verified against its published SHA-256
`bbd482c786b895cf7d8f44185073d951adae2ebb8a66f82ca84cd1f84569549c`.
The upstream source at commit `68456b17da8eea651eee6084c104611089810231`
was run with PyTorch 2.15.0 CPU float32. The Rust reference path uses f64.
On the five-second synthetic plucked-string fixture:

| Scenario | Exact token agreement | Maximum scaled logit error |
| --- | ---: | ---: |
| Greedy | 48/48 (includes EOS) | 2.142e-5 |
| Forced prompt | 80/80 (hit the token cap) | 2.520e-5 |

All 48 and 77 generated logit rows, respectively, met the 1e-3 scaled-error
criterion, including matching masked positions. The CLI ran the same waveform
to EOS and produced 10 JSONL note records, byte-identical across two runs. The
preliminary fingering library found valid standard-tuning assignments for those
notes; this is a feasibility check, not a tablature-quality evaluation.
One separate run took 23.5 seconds including checkpoint load, with peak RSS
789 MiB, `MATMUL_NUM_THREADS=2`, on the sandbox above. This is a single
synthetic input, not a GuitarSet evaluation, and does not establish accuracy or
the performance targets in §6.13. Current velocity remains the upstream
constant 100. Model weights and generated reference logits are not committed.

## Independent MIDI reader check (2026-11-22)

With the same official small checkpoint and five-second synthetic plucked-string
waveform, separate CLI runs generated JSONL and preliminary MIDI. An independent
Mido 1.3.3 parser read the SMF type 1 file: 2 tracks, 10 note-ons and 10
note-offs matching the JSONL pitches, velocities and tick positions, with the
expected 120 BPM / 4/4 map. Reproduce with identical input, model and generation
settings, then `python tools/check_midi_mido.py notes.jsonl song.mid` (Mido is a
**test-only** Python dependency). This checks file structure and events only;
it is not playback, a DAW round-trip, musical-accuracy evaluation, or validation
of dynamics and bends (which the transcription path does not yet infer).

## Experimental dynamics integration (2026-11-22)

On the same synthetic fixture and official small checkpoint, `--dynamics`
estimated velocities for all 10 generated notes; a MIDI parser confirmed the
exported note-on velocities and timings match the estimator's output. An
independent Mido parse also matched 10 notes against a second `--dynamics`
JSONL run (estimated velocities 106, 109, 109, 109, 104, 104, 104, 104, 112,
78). The estimator measures 50 ms attack RMS relative to a two-second local RMS window
on the mono mix. This establishes wiring and non-constant output, **not**
accuracy, calibration to MIDI velocity, separation of overlapping sources, or
performance on real music. The flag is opt-in; without it, upstream's constant
velocity is preserved.

## Experimental tuning evidence (2026-11-22)

A bounded monophonic YIN-style pitch histogram and open-string template scorer
are integrated into `detect-tuning` and the MIDI path's automatic settings.
Synthetic isolated sine sequences covering eight-string open notes and a
standard-tuning capo-2 sequence passed pipeline tests; a standard six-string
sequence correctly abstained because extended tunings share its open pitches.
The five-second synthetic plucked-string fixture also abstained: its best and
runner-up hypotheses tied at an open-pitch fraction of 0.511. These checks are
not a real-recording evaluation. Fretted notes and polyphonic mixtures can
produce confident but incorrect guesses; thresholds are uncalibrated. The CLI
warns and falls back to standard / no capo for insufficient evidence (preserving
explicit overrides), and users should verify or set tuning and capo manually.
