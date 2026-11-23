# mtab patches to `symphonia-format-riff` 0.5.5

Source: <https://crates.io/crates/symphonia-format-riff/0.5.5> (MPL-2.0,
`LICENSE-MPL-2.0`). The `src/` tree is the published crate unmodified except
for the changes below, applied because `mtab` decodes untrusted audio files.

1. **Zero sample rate (WAVE and AIFF).** Both readers call
   `TimeBase::new(1, sample_rate)` without validating `fmt `'s sample rate.
   `TimeBase::new` asserts a non-zero denominator, so a malformed file aborted
   the whole process instead of returning an error. Each reader now rejects a
   zero sample rate with `unsupported_error("<wav|aiff>: zero sample rate")`.
   Replay seed: `fuzz/seeds/audio_decode/zero_sample_rate.wav`.

These patches are provided under the MPL-2.0 like the original crate. They are
upstream-report candidates; if a fixed release appears, drop the `[patch]`.
