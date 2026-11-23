# External AutoTab v3 oracle-note diagnostic protocol

Precommitted before running the evaluation. This is **not** the M5/M8
end-to-end acceptance benchmark: it supplies ground-truth pitches and timings,
not transcribed audio, and its model was trained on GuitarSet. It is an
in-sample diagnostic, not a generalization estimate. It does not measure
TabCNN/TDR, audio recognition, or the full MuScriptor-to-tab pipeline.

Run the unmodified external AutoTab CLI from
`lepusagri/autotab-rust` commit `a1eebbebbd74db669a6a5f74079eed907717cfb3`
with `--tab-engine v3 --device cpu --threads 1 --midi-track all` on each
GuitarSet 1.1.0 JAMS annotation (all 360 files, sorted by filename). Use its
specified `cstr/tab-labeler-onnx` v3-egset model, revision
`434976baacaf194d99f09c3c784682766bedcf97`, SHA-256
`93114786c6f4781e7a0b69abd7e93a0d4966873f8e54908f97a1de8e8cf788af`;
weights are CC BY 4.0 and are not committed. Model and application versions,
runtime, binary hash, input hashes, failures and exact selection must be
reported. The source repository does not declare a license for its application
code; run it as a separate executable and do not copy its code into mtab.

Write each reference JAMS `note_midi` event to a MIDI file at PPQ 25,000 and
500,000 µs/beat: time resolution 20 µs, rounding to nearest tick and forcing
at least one tick duration. Map GuitarSet string 0 (low E) through 5 (high E)
to MIDI channels 0–5 purely as a traceable ID. The external fingering search
does not use source channels as features: check the cited upstream source.
Export all events to a single track, sorted by tick, note-off before note-on,
then reference string and pitch. Handle overlapping identical pitches on the
same reference string carefully: the external MIDI reader pairs note-offs FIFO;
if this would change a note's endpoint, exclude the entire file and report it
rather than score the wrong reference. Verify the CLI report's original
`source_notes` mapping back to every generated note (pitch, start and end tick,
channel) before scoring. Compare predicted string to `6 - channel` because
AutoTab counts high E as string 1 and GuitarSet counts low E as string 0. If
any note is unmatched, duplicated, dropped, unassigned or placed on an extra
guitar, report those counts; accuracy uses **all** source notes as denominator,
and unassigned notes are incorrect. Report first-guitar-only and all-guitar
scores separately; an extra guitar is not a valid single-guitar transcription.

Report per-file outcomes and totals with no post-hoc exclusion. Compare only
qualitatively against `benches/guitarset-oracle.md`: the latter used original
floating-point note times, different fret limits, different grouping and
failure policies. Even a favorable number cannot establish acceptance.
Use no MuScriptor weights, no network during evaluation, and no unreviewed
changes to AutoTab or its model. Preserve a reproducible script and logs.
