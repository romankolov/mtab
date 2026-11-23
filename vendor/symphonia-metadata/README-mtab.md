# mtab patches to `symphonia-metadata` 0.5.5

Source: <https://crates.io/crates/symphonia-metadata/0.5.5> (MPL-2.0,
`LICENSE-MPL-2.0`). Unmodified except for:

1. **FLAC picture block length fields (`flac.rs`).** The MIME-type,
   description and picture-data lengths are read as `u32` and used directly in
   `vec![0u8; len]`, so a tiny file could request up to ~4 GiB and abort inside
   the allocator. `read_exact_bounded` now reads in 64 KiB chunks that grow only
   as bytes actually arrive; the enclosing scoped block returns an error once
   the read runs past the declared block.
   Replay seed: `fuzz/seeds/audio_decode/flac_picture_length.bin`.
2. **Vorbis comment length (`vorbis.rs::read_comment_no_framing`).** A
   comment length of 0xffffffff previously caused the same up-front allocation
   abort; comments are now read the same way.
   Replay seed: `fuzz/seeds/audio_decode/flac_comment_length.bin`.
