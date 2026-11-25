# Held-out contextual string-choice probe

## Question and scope

Can pitch-and-phrase context provide a useful string-choice signal beyond the
lowest-fret-per-note rule, when the guitarist is unseen? This is an exploratory
oracle-note diagnostic, not an audio transcription result, a playable tab, or a
replacement for the existing sequence solver. It does not benchmark Songsterr
or claim acceptance-level accuracy.

The previously suggested oracle/predicted-note split is already covered by
[`guitarset-oracle.md`](guitarset-oracle.md) and
[`m5-tab-labeler-v3-head-to-head.md`](m5-tab-labeler-v3-head-to-head.md): the
former gives the project solver reference notes, and the latter gives AutoTab
and mtab the same predicted notes. This probe tests a different idea: a small,
held-out supervised model of picked string from local note context.

## Fixed experiment

- **Training:** GuitarSet sessions 00–02.
- **Test:** sessions 03–05, held out by session/player.
- **Input:** reference MIDI pitches and durations; simultaneous-onset group
  summaries (30 ms grouping); and up to three preceding and following
  onset-group summaries.
- **Excluded from features:** annotated string, recording filename, audio,
  session identity, and transcriber output.
- **Classifier:** scikit-learn `HistGradientBoostingClassifier`, 250 iterations,
  learning rate 0.08, 31 maximum leaves, L2 1.0, seed 0. No parameter sweep.
- **Legality mask:** at inference, discard predicted strings that cannot play
  that pitch in standard tuning at frets 0–22. This is only a per-note fret
  range check; there is no overlap, hand-span, or phrase-playability constraint.
- **Baseline:** independently choose the legal string with the lowest fret.

Reproduce with NumPy and scikit-learn installed:

```sh
python3 tools/contextual_string_probe.py \
  --annotations /path/to/GuitarSet/jams \
  --output /tmp/contextual-string-probe.json
```

The output from the available local annotation set is saved as
[`contextual-string-probe.json`](contextual-string-probe.json), including the
per-session scores, confusion matrix, and a hash over the sorted annotation
file hashes.

## Result

The training set contained **34,150** note annotations and the held-out set
**28,326**. The per-note classifier assigned the correct annotated string for
**20,825 / 28,326 = 73.52%** of test annotations. The lowest-fret baseline
scored **12,696 / 28,326 = 44.82%**. The legality mask did not change the
classifier's aggregate result on this dataset.

| Held-out session | Notes | Context classifier | Lowest-fret baseline |
| --- | ---: | ---: | ---: |
| 03 | 9,358 | 73.45% | 50.68% |
| 04 | 10,253 | 73.27% | 45.54% |
| 05 | 8,715 | 73.90% | 37.68% |

This is evidence that local pitch sequence/chord context carries string-choice
information beyond a pitch-only lowest-fret rule. It is **not** evidence that
the classifier improves mtab: it predicts each note independently and can
produce same-string collisions or impossible hand spans. Also, its denominator
contains all annotation events, while the published mtab oracle headline is
restricted to solver-feasible excerpts. Do not compare those percentages as if
they were a matched benchmark. The classifier is below the existing mtab
feasible-only oracle number, but the scopes differ.

## Decision and next step

Do **not** replace the current solver or merge this independent classifier.
The result is a positive signal probe but not a playable fingering system. The
next useful test is to feed the learned per-string probabilities into a
constraint-aware sequence search as unary costs, then evaluate that combined
solver against the current solver on the **same held-out excerpts**, first with
oracle notes and then with the existing predicted-note JSONL. Report feasible
and infeasible annotations separately, plus both tab-position accuracy and
wrong-string rate. The training split, feature set, and classifier settings
must remain fixed before running that comparison.

GuitarSet annotations are CC BY 4.0; annotation errors and hexaphonic pickup
bleed can create duplicated note events. This small model and its probe are
research diagnostics only and do not add a runtime dependency to mtab.
