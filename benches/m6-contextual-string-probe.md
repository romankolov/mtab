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

## Constraint-aware integration result

The per-note classifier was used only as an additive string-cost prior inside
the production bounded fingering beam. Playability constraints and the
existing ergonomic costs remain active; pitch/onset/offset predictions are
unchanged. The library now exposes `assign_notes_with_string_costs` and its
partial-assignment variant for callers that have externally calibrated
per-note candidate costs.

### Choose the prior scale on validation, then freeze it

Train the classifier on sessions 00–01, select the cost multiplier on session
02, and only then refit on 00–02 for the held-out 03–05 test. The fixed
validation grid was λ ∈ {0, 0.25, 0.5, 1, 2}; score is exact string accuracy
on the 41 feasible session-02 recordings (5,935 notes).

| λ | Correct / feasible notes | Accuracy |
| ---: | ---: | ---: |
| 0 (current solver) | 4,357 / 5,935 | 73.41% |
| 0.25 | 4,793 / 5,935 | 80.76% |
| 0.5 | 4,774 / 5,935 | 80.44% |
| **1.0 (selected)** | **4,802 / 5,935** | **80.91%** |
| 2.0 | 4,794 / 5,935 | 80.78% |

The selected λ=1.0 is fixed for the final test; no test-set tuning was used.

### Oracle-note test, held-out sessions 03–05

On the same **131 feasible recordings / 20,396 reference notes**, with 49
infeasible recordings reported separately, the current solver scored
**15,852 / 20,396 = 77.72%**. Adding the contextual prior scored
**16,565 / 20,396 = 81.22%**, a **+3.50 percentage-point** gain. Neither run
failed on a feasible recording. This is an oracle-note assignment result, not
an audio-to-tab score.

### Same predicted notes, 57 matched excerpts

On the exact transcript set used by the existing AutoTab comparison, both
variants receive the same 3,113 notes. Pitch rows are therefore unchanged by
construction. The three entries that lack a transcript remain unscored on both
sides. The prior changes only string/fret assignment:

| Metric | Current solver | With prior | Change |
| --- | ---: | ---: | ---: |
| Multi-pitch frame F1 | 70.20% | 70.20% | 0.00 pp |
| Pitch + onset F1 | 77.55% | 77.55% | 0.00 pp |
| **Tab-position frame F1** | 47.05% | **52.99%** | **+5.94 pp** |
| **Tab + onset F1** | 50.65% | **58.40%** | **+7.75 pp** |
| Pitch + onset + offset F1 | 40.46% | 40.46% | 0.00 pp |
| Tab + onset + offset F1 | 26.03% | **30.84%** | **+4.81 pp** |

Wrong-string rate on pitch-matched instances fell from **32.89% to 24.41%**.
This closes a substantial part of the earlier AutoTab gap, but does not close
it: on that same 57-entry set AutoTab v3 scored 58.92% tab-position F1 and
64.62% tab + onset F1. Its comparison remains training-contaminated, as
explained in the head-to-head report.

### Reproduction commands

Use the same full annotation directory as the prior probe, the saved mtab
transcripts for the 57-entry matched comparison, and local scratch directories
for per-note costs and IRs:

```sh
# Select lambda on session 02, training only on 00-01.
python3 tools/contextual_string_probe.py \
  --annotations <guitarset-jams> --output /tmp/context-tune.json \
  --train-sessions 00,01 --test-sessions 02 \
  --write-test-priors-dir /tmp/context-costs-02
cargo run -p mtab-cli --example contextual_fingering_eval -- \
  <guitarset-jams> /tmp/context-costs-02 02 0,0.25,0.5,1,2

# Refit on 00-02; evaluate oracle-note costs once on held-out sessions 03-05.
python3 tools/contextual_string_probe.py \
  --annotations <guitarset-jams> --output /tmp/context-test.json \
  --write-test-priors-dir /tmp/context-costs-03-05
cargo run -p mtab-cli --example contextual_fingering_eval -- \
  <guitarset-jams> /tmp/context-costs-03-05 03,04,05 0,1

# Apply the same trained classifier to the frozen predicted-note JSONL set.
python3 tools/contextual_string_probe.py \
  --annotations <guitarset-jams> --output /tmp/context-predicted.json \
  --predicted-transcripts-dir <nN.jsonl-directory> \
  --write-predicted-priors-dir /tmp/context-predicted-costs
cargo run -p mtab-cli --example contextual_transcript_eval -- \
  <nN.jsonl-directory> /tmp/context-predicted-costs <60-entry-manifest> \
  /tmp/context-ir 1
mtab eval --tab-manifest /tmp/context-ir/baseline-manifest.json
mtab eval --tab-manifest /tmp/context-ir/prior-manifest.json
```

The result is promising enough to keep developing, but it is not yet a
production-ready learned fingering system. The contextual probabilities are
trained on GuitarSet strings; the current held-out split is only one player
split; and the search still uses a fixed hand-window policy. Next, validate on
a second independent dataset / guitarist, inspect the six largest per-entry
regressions, and confirm that the gain survives the full 60-entry partial
fingering path before changing default behavior.

GuitarSet annotations are CC BY 4.0; annotation errors and hexaphonic pickup
bleed can create duplicated note events. The Python model is an offline
research tool and adds no runtime dependency to mtab. Full aggregate counts
are in [`contextual-fingering-results.json`](contextual-fingering-results.json).
