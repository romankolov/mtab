# mtab patches to `symphonia-codec-aac` 0.5.5

Source: <https://crates.io/crates/symphonia-codec-aac/0.5.5> (MPL-2.0,
`LICENSE-MPL-2.0`). Unmodified except for:

1. **`adts.rs::approximate_frame_count` subtraction overflow.** The function
   computes `total_len - original_pos`, but `MediaSourceStream::byte_len()`
   returns the absolute stream length while `original_pos` is the probe
   position. Probing an ADTS stream that starts after the file header (for
   example inside a WAVE file) underflowed and panicked. It also used
   `original_pos` twice while treating `total_len` as a remainder. The function
   now uses `byte_len().checked_sub(original_pos)`, returns `Ok(None)` for an
   empty remainder or a zero step, and samples over
   `original_pos..original_pos + total_len - step`.
   Replay seed: `fuzz/seeds/audio_decode/m4a_like_crash.bin`.

2. **Scalefactor-band count (`aac/ics/mod.rs`).** `max_sfb` (4 or 6 bits) was
   accepted up to 63, but the sample-rate subband tables are shorter (for
   example 42 entries for 48 kHz long windows), and `decode_spectrum` indexed
   `bands[sfb + 1]` unconditionally. Long and short windows now reject
   `max_sfb >= MAX_SFBS`, and `decode_spectrum` rejects `max_sfb` beyond the
   table for the active sample rate. Replay seed:
   `fuzz/seeds/audio_decode/aac_max_sfb.bin`.
