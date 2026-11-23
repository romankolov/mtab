# GuitarSet symbolic chord diagnostic protocol

Frozen before evaluation. This is an **oracle-note** diagnostic of
`mtab_guitar::suggest_chords`, not transcription from audio and not a chord
recognition acceptance benchmark. GuitarSet's first `chord` annotation is a
sheet-derived harmonic progression, not necessarily the exact guitar voicing
or pitch classes sounding at every instant. The second chord annotation is
sheet-informed symbolic transcription; it is not used here. Comp and solo
tracks share a progression, so neither comparison proves that an exact chord
is played in the solo. Interpret all figures accordingly.

- Inputs: every `.jams` in official GuitarSet 1.1.0 annotation.zip, sorted by
  name (360 total); archive MD5 `b39b78e63d3446f2e54ddb7a54df9b10`.
- Provide the six `note_midi` annotations as ground-truth note events to the
  library; use fixed `minimum_duration = 0.1` seconds. No audio or MuScriptor
  predictions. No post-hoc tuning on this evaluation.
- Compare to the first `chord` annotation, without bass-inversion evaluation,
  using the chord root and the major/minor triad reduction from the pinned
  `mir_eval` version 0.8.2. Use 10 ms frame centers starting at 5 ms and ending
  at the last reference-note offset (or reference-chord annotation end,
  whichever is earlier). A missing suggestion is `N`; normalize proposed
  labels to Harte notation and convert enharmonic spellings by mir_eval.
- Calculate frame-micro agreement as correct comparable frames divided by all
  comparable reference frames, suggestion coverage as suggested frames divided
  by comparable reference frames, and conditional agreement as correct frames
  divided by suggested comparable frames. Mark unsupported reference qualities
  as incomparable according to mir_eval's triad metric, and report their count.
  Also report the same counts separately for comp and solo recordings. Fail on
  malformed input or overlapping, duplicate, or out-of-order chord intervals;
  do not silently skip any file.
- Report package version, executable hash, per-file hashes, errors and
  reproducibility instructions. These descriptive corpus results do not prove
  generalization, musical quality or downstream export support.
