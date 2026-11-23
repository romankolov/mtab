# mtab patches to `symphonia-utils-xiph` 0.5.5

Source: <https://crates.io/crates/symphonia-utils-xiph/0.5.5> (MPL-2.0,
`LICENSE-MPL-2.0`). Unmodified except for:

1. **FLAC application block (`flac/metadata.rs::read_application_block`).**
   `block_length as usize - 4` panicked in debug builds for a declared length
   below 4, and the remaining length was allocated up front. The subtraction is
   now checked and the payload is read in bounded chunks.
   Replay seed: `fuzz/seeds/audio_decode/flac_comment_length.bin` (same FLAC
   metadata path; the application-block corpus entry is the minimized crash).
