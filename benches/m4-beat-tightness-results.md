# Beat-path tightness: frozen development/validation experiment

The [protocol](m4-beat-tightness-protocol.md) was committed as `76b975d`
before this exploration or the validation measurements. The previously
published first-take 50-recording set became **development data**. We
selected tightness using its automatic-tempo micro F1, then measured the
preselected second-take set once at the winner and at the original 3.0.
The validation takes differ from development but share five performers,
styles and often progressions. GuitarSet annotations were also used in other
mtab work. This is **not** a project-wide, performer-held-out or full-corpus
benchmark, and the five correlated performers do not justify broad
statistical claims. No parameter was changed after the validation results.

## Development: select by automatic-tempo micro F1

| Tightness | Oracle-tempo micro F1 | Automatic-tempo micro F1 | Automatic TP / FP / FN |
|---:|---:|---:|---:|
| 1 | 48.25% | 45.72% | 907 / 1181 / 973 |
| 3 | 52.82% | 45.48% | 880 / 1110 / 1000 |
| 10 | 63.96% | 47.46% | 912 / 1051 / 968 |
| **30** | **65.53%** | **48.87%** | **940 / 1027 / 940** |
| 100 | 63.14% | 43.62% | 839 / 1128 / 1041 |

The predeclared selection rule picked **30**, using the unrounded integer
counts. Oracle tempo is a diagnostic, not the selection criterion. The
previous 3.0 report was reproduced byte-for-byte.

## Separate 50-take validation

| Supplied tempo | Tightness | TP | FP | FN | Micro F1 | Mean per-take F1 |
|---|---:|---:|---:|---:|---:|---:|
| Reference BPM | 3 | 1168 | 743 | 702 | 61.78% | 61.46% |
| Reference BPM | 30 | 1354 | 538 | 516 | 71.98% | 71.15% |
| Top autocorrelation peak | 3 | 1097 | 1327 | 773 | 51.09% | 49.65% |
| Top autocorrelation peak | 30 | 1138 | 1261 | 732 | 53.31% | 51.27% |

The selected value improved automatic-tempo micro F1 by **2.22 percentage
points** on this fixed validation set; reference-tempo micro F1 improved by
10.20 points. These are descriptive differences, not significance claims.
An additional post-hoc per-recording comparison found automatic-tempo F1
improved on 29 takes, was unchanged on 4, and worsened on 17. The top tempo
candidate was within 4% of reference on **18/50** takes, or within 4% of
half, equal or double reference on **30/50**; changing tightness does not
change that candidate. This set covers just five reference tempi: 68, 90,
97, 147 and 200 BPM, all 4/4.

Even after the observed improvement, automatic-tempo errors remain substantial.
As an additional post-hoc synthetic control, the existing louder-offbeat
fixture still scores **0% F1** with either 3 or 30: a tighter period cannot
resolve this phase ambiguity. The check reports this score rather than requiring
future versions to preserve the failure. The test uses click-derived beats, not local
performance-beat truth, scores only after five seconds, assumes constant target
tempo, and provides neither downbeats nor meter. The strongest-peak rule is not calibrated and may select
noise, subdivisions or the wrong metrical level. Accordingly **no automatic
tracker or new default was enabled at the time of this experiment**. A later
explicit opt-in CLI flag composes this rule with MIDI export; it remains
experimental, and the default is unchanged. The conditioned path API still
allows callers to choose tempo and tightness explicitly.

## Reproduction and raw reports

Data preparation, published archive MD5s, software versions and code hashes
are described in [the original 50-recording report](m4-guitarset-beat-path.md).
The experiment used the same three release binaries built from `fddde9c` and
the evaluation script at `76b975d`; Python 3.13, mir_eval 0.8.2, NumPy 2.5.5,
SciPy 1.18.1. The validation WAVs were read from the same Zenodo ZIP using
HTTP-range extraction with ZIP CRC checks, not a complete archive-MD5 check.
All per-WAV and per-JAMS SHA-256 values are in the raw reports; hashes and CRC
are not independent authenticity proofs. Downloading and checking the full
archive provides stronger verification.

The unmodified evaluator outputs are:

- Development: [1](beat-tightness-results/development-1.md),
  [3](beat-tightness-results/development-3.md),
  [10](beat-tightness-results/development-10.md),
  [30](beat-tightness-results/development-30.md),
  [100](beat-tightness-results/development-100.md).
- Validation: [3](beat-tightness-results/validation-3.md) and
  [30](beat-tightness-results/validation-30.md).

After preparing data as in the earlier report, for each development tightness
run `tools/evaluate_guitarset_beat_paths.py` with `--take-index 0 --tightness N`;
for validation use `--take-index 1` with `--tightness 3` and `--tightness 30`.
All runs also require the same explicit `--annotation-zip`, `--audio-dir` and
`--examples-dir` paths shown in that report. Evaluations do not use model
weights or MuScriptor inference.
