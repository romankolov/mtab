# Frozen protocol: beat-path tightness exploration and validation

This protocol was committed **before** running the tightness grid and before
examining scores on the validation recordings. It tests only the transition
penalty of the existing `track_with_tempo` objective; it does not propose
changing the onset envelope, tempo candidate selection, model, scoring, code,
or introducing a CLI tracker. It is a small engineering experiment, not a
claim of general beat-tracking quality.

- **Development:** the previously reported 50 recordings: GuitarSet 1.1.0
  players 01–05 × styles BN/Funk/Jazz/Rock/SS × comp/solo, lexicographically
  **first** JAMS filename in each group. The existing tightness 3.0 results
  are already known; this set is now explicitly tuning data.
- **Validation:** the same groups, lexicographically **second** JAMS filename
  in each group, chosen solely by the rule before seeing validation results.
  No recording overlaps development, but performers, styles and often chord
  progression overlap, and GuitarSet annotations informed unrelated mtab
  work. This is not a project-wide or performer-held-out test.
- **Grid:** evaluate tightness **1, 3, 10, 30, 100** on the development set.
  Select the highest micro beat F1 of the automatic-tempo baseline (highest
  positive autocorrelation peak; if no peak, no predictions). If scores tie,
  choose the smallest tightness. Use TP/FP/FN integers, not rounded table values,
  to compare. The oracle-tempo score is reported for diagnosis, not selection.
- **Final check:** evaluate the selected value and original value 3.0 once on
  all 50 validation recordings; report both micro and mean per-recording F1,
  per-track counts and source hashes. Do not choose a different value after
  observing validation outcomes. No confidence interval or broad quality
  claims are justified by five shared performers and correlated takes.
- **Metrics:** trim reference and estimated beats before 5.0 s; one-to-one
  ±70 ms matching via mir_eval 0.8.2, against click-derived quarter-beat
  references. Tempo inference uses the entire recording and does not see the
  reference BPM; the oracle branch is separately given the annotated BPM.
  This evaluates beat alignment after five seconds, not downbeats or rubato.

Reproduce data preparation as documented in `m4-guitarset-beat-path.md`. With
the same `--annotation-zip`, `--audio-dir` and `--examples-dir` arguments,
run the evaluator using `--take-index 0 --tightness N` for each grid value.
After selecting by the frozen rule, run with `--take-index 1` once at the
selected value and once at 3.0. The tool reports both baselines and hashes.
The results and precise executable/environment provenance will be published
separately; neither result will be promoted to production by this protocol.
