# Vendored guitarpro 0.1.0 (MIT)

This directory contains the library source from the `guitarpro` 0.1.0 crate
by slundi: https://crates.io/crates/guitarpro/0.1.0 and
https://github.com/slundi/guitarpro. Its MIT license is in `LICENSE`.
The package manifest was reduced to the library dependencies, and the
workspace excludes this vendored crate from standalone builds. The newer crate
uses Rust edition 2024, incompatible with mtab's Rust 1.75 minimum.

Changes made for mtab's limited GP5 5.00 encoder:

- Fix the integer-size string length, used by GP5 lyric blocks.
- Write multi-track measures in time-major rather than track-major order.
- Correct inverted default-effect flags for beats and notes.
- Match PyGuitarPro's integer rounding when packing velocity dynamics.
- Remove channel-writing debug output and expose a MIDI program setter.
- Allow upstream's existing unused private helpers under `-D warnings`.

This is **not** an assertion that all features of the old upstream library are
correct. mtab validates and restricts inputs to its independently tested GP5
subset, and does not use this fork as an arbitrary untrusted-file parser.
PyGuitarPro is a test-only independent reader and is not included or used by
the runtime.
