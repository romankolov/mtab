# Vendored guitarpro 0.1.0 (MIT)

This directory contains the library source from the `guitarpro` 0.1.0 crate
by slundi: https://crates.io/crates/guitarpro/0.1.0 and
https://github.com/slundi/guitarpro. Its MIT license is in `LICENSE`.
The package manifest was reduced to the library dependencies, and the
workspace excludes this vendored crate from standalone builds. The newer crate
uses Rust edition 2024, incompatible with mtab's Rust 1.75 minimum.

Changes made for mtab's limited GP5 5.00 encoder:

- Fix the integer-size string length, used by GP5 lyric blocks.
- Encode GP5 text as Windows-1252 and write byte rather than UTF-8 character
  counts. The mtab wrapper rejects unsupported characters before encoding.
- Write multi-track measures in time-major rather than track-major order.
- Correct the GP5 measure-header reserved-byte condition.
- Correct inverted default-effect flags for beats and notes, and correct the
  GP5 palm-mute bit (it must not be encoded as staccato).
- Fix the second note-effect byte for tremolo picking, slides, harmonics,
  trills and vibrato; the old writer incorrectly set the staccato bit for all
  five. PyGuitarPro tests single natural harmonics and combined flags/payloads;
  this does not validate every effect variant or expose these effects in mtab.
- Match PyGuitarPro's integer rounding when packing velocity dynamics.
- Remove channel-writing debug output and expose a MIDI program setter.
- Allow upstream's existing unused private helpers under `-D warnings`.
- Make `chord::PitchClass::from` and `PitchClass::from_note` public so the
  exporter can attach a chord root/quality to a beat.
- Rewrite `write_chord_v4` to the documented GP4/GP5 chord layout. The old
  writer emitted ints where the format (and PyGuitarPro, and this crate's own
  reader) uses bytes for root, type, extension, fifth/ninth/eleventh, and for
  the barre count/frets/starts/ends, so any file with a chord diagram was
  unreadable past the first chord.
- Write the 7 fret ints of a GP4/GP5 chord slot (7-string layout) instead of 6,
  and fix the reader to assign each fret into its string slot instead of
  appending (which duplicated slots and shifted every later field).

This is **not** an assertion that all features of the old upstream library are
correct. mtab validates and restricts inputs to its independently tested GP5
subset, and does not use this fork as an arbitrary untrusted-file parser.
PyGuitarPro is a test-only independent reader and is not included or used by
the runtime.
