# GuitarSet end-to-end audio evaluation

Entries: 60; failed entries: 0; total frames: 60000; pairs with both supplied SHA-256 hashes verified: 60. Model architecture: small; greedy decoding, seed 0, guitar-family mask, strict EOS, token limit 2000, standard tuning, capo 0, 22 frets, fingering: strict, then span-four partial retry (outputs can omit notes); omitted generated notes in successful entries: 25. The model is loaded once. Each manifest duration is rounded down to 16 kHz sample boundaries; only that prefix is transcribed and scored. No source separation, technique inference, or dynamics estimation.

| Micro-aggregated metric | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-pitch frames | 66266 | 32971 | 24091 | 66.78% | 73.34% | 69.90% |
| Tab position frames | 45589 | 53651 | 44988 | 45.94% | 50.33% | 48.03% |
| Pitch + onset notes | 2446 | 946 | 623 | 72.11% | 79.70% | 75.72% |
| Pitch + onset + offset notes | 1266 | 2126 | 1803 | 37.32% | 41.25% | 39.19% |
| Tab + onset notes | 1635 | 1757 | 1434 | 48.20% | 53.27% | 50.61% |
| Tab + onset + offset notes | 829 | 2563 | 2240 | 24.44% | 27.01% | 25.66% |

Note-event rows use same MIDI pitch and onset within 50 ms; the offset row also requires offset within max(50 ms, 20% of reference duration). Tab note rows additionally require the same string (and therefore the same fret in standard tuning). Each uses independent maximum-cardinality matching. Notes crossing the window end are clipped.

TDR-style ratio (correct tab positions / correct unique pitches): 45589 / 66266 = 68.80%. Simultaneous unisons can produce a ratio above 100%; it is not the complement of wrong-string rate.

Wrong-string among pitch-matched string instances: 20677 / 66266 = 31.20%.

| Manifest entry | Frames | Reference notes | Scored predicted notes | Omitted generated notes | Multi-pitch F1 | Tab-position F1 | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1000 | 72 | 75 | 0 | 75.61% | 24.57% | ok |
| 2 | 1000 | 31 | 32 | 0 | 74.63% | 42.79% | ok |
| 3 | 1000 | 81 | 104 | 0 | 73.03% | 72.43% | ok |
| 4 | 1000 | 27 | 25 | 0 | 83.98% | 34.96% | ok |
| 5 | 1000 | 79 | 70 | 0 | 66.49% | 46.26% | ok |
| 6 | 1000 | 17 | 24 | 0 | 77.94% | 57.33% | ok |
| 7 | 1000 | 86 | 76 | 0 | 75.96% | 15.25% | ok |
| 8 | 1000 | 31 | 32 | 0 | 80.00% | 62.12% | ok |
| 9 | 1000 | 61 | 97 | 0 | 64.75% | 34.46% | ok |
| 10 | 1000 | 24 | 26 | 0 | 77.19% | 74.79% | ok |
| 11 | 1000 | 80 | 60 | 0 | 61.88% | 35.61% | ok |
| 12 | 1000 | 23 | 39 | 0 | 63.79% | 41.52% | ok |
| 13 | 1000 | 69 | 74 | 0 | 69.62% | 51.59% | ok |
| 14 | 1000 | 35 | 38 | 0 | 72.75% | 65.17% | ok |
| 15 | 1000 | 76 | 98 | 0 | 65.70% | 54.54% | ok |
| 16 | 1000 | 37 | 35 | 0 | 63.38% | 42.80% | ok |
| 17 | 1000 | 68 | 87 | 0 | 72.38% | 71.23% | ok |
| 18 | 1000 | 35 | 36 | 0 | 73.27% | 46.27% | ok |
| 19 | 1000 | 82 | 140 | 15 | 52.61% | 47.83% | ok |
| 20 | 1000 | 29 | 33 | 0 | 64.08% | 53.66% | ok |
| 21 | 1000 | 74 | 85 | 0 | 68.77% | 42.29% | ok |
| 22 | 1000 | 32 | 39 | 0 | 63.26% | 33.33% | ok |
| 23 | 1000 | 103 | 133 | 0 | 67.77% | 67.43% | ok |
| 24 | 1000 | 31 | 32 | 0 | 69.18% | 50.64% | ok |
| 25 | 1000 | 48 | 67 | 0 | 55.24% | 47.92% | ok |
| 26 | 1000 | 33 | 34 | 0 | 83.86% | 17.11% | ok |
| 27 | 1000 | 54 | 50 | 0 | 52.88% | 51.02% | ok |
| 28 | 1000 | 32 | 33 | 0 | 68.85% | 64.25% | ok |
| 29 | 1000 | 39 | 75 | 0 | 62.69% | 13.17% | ok |
| 30 | 1000 | 28 | 25 | 0 | 81.35% | 55.82% | ok |
| 31 | 1000 | 51 | 51 | 1 | 67.79% | 57.98% | ok |
| 32 | 1000 | 37 | 38 | 0 | 71.96% | 51.74% | ok |
| 33 | 1000 | 38 | 37 | 0 | 71.28% | 53.26% | ok |
| 34 | 1000 | 40 | 39 | 0 | 85.25% | 75.49% | ok |
| 35 | 1000 | 37 | 40 | 0 | 56.25% | 34.93% | ok |
| 36 | 1000 | 34 | 39 | 0 | 71.49% | 53.81% | ok |
| 37 | 1000 | 98 | 98 | 0 | 77.47% | 1.77% | ok |
| 38 | 1000 | 27 | 27 | 0 | 74.63% | 9.92% | ok |
| 39 | 1000 | 59 | 92 | 9 | 78.80% | 73.33% | ok |
| 40 | 1000 | 23 | 19 | 0 | 69.66% | 25.06% | ok |
| 41 | 1000 | 81 | 87 | 0 | 70.56% | 52.41% | ok |
| 42 | 1000 | 28 | 30 | 0 | 69.12% | 44.12% | ok |
| 43 | 1000 | 52 | 87 | 0 | 71.60% | 71.60% | ok |
| 44 | 1000 | 19 | 22 | 0 | 80.88% | 58.28% | ok |
| 45 | 1000 | 78 | 74 | 0 | 76.16% | 35.29% | ok |
| 46 | 1000 | 29 | 33 | 0 | 76.06% | 55.54% | ok |
| 47 | 1000 | 109 | 101 | 0 | 71.66% | 26.36% | ok |
| 48 | 1000 | 29 | 38 | 0 | 69.79% | 45.88% | ok |
| 49 | 1000 | 126 | 96 | 0 | 58.08% | 57.45% | ok |
| 50 | 1000 | 32 | 33 | 0 | 77.47% | 56.93% | ok |
| 51 | 1000 | 74 | 51 | 0 | 61.46% | 44.69% | ok |
| 52 | 1000 | 24 | 24 | 0 | 89.32% | 88.38% | ok |
| 53 | 1000 | 119 | 131 | 0 | 68.44% | 63.44% | ok |
| 54 | 1000 | 18 | 23 | 0 | 65.12% | 3.53% | ok |
| 55 | 1000 | 89 | 116 | 0 | 76.78% | 76.78% | ok |
| 56 | 1000 | 21 | 20 | 0 | 72.74% | 63.47% | ok |
| 57 | 1000 | 39 | 41 | 0 | 64.87% | 4.03% | ok |
| 58 | 1000 | 22 | 18 | 0 | 84.04% | 72.19% | ok |
| 59 | 1000 | 99 | 97 | 0 | 73.07% | 25.25% | ok |
| 60 | 1000 | 20 | 20 | 0 | 84.03% | 73.66% | ok |

Omitted generated notes are counted before clipping and excluded from scoring; they may have been true or false positives, so omission can change both precision and recall. Omission counts are unknown for failed entries and shown as n/a. Failures (including strict-EOS, decoding, assignment or short audio) count as empty predictions, never excluded; thus failures contribute reference false negatives but cannot estimate their hypothetical false positives. Pitch and tab metrics are 10 ms note-derived fixed-pitch frame sets, not continuous F0. The TDR-style ratio follows Wiggins and Kim (2019), Eq. 8, but the reference and frame grid differ from their protocol. No held-out split, tab-labeler-v3 baseline, or independent ground-truth pairing is established by the manifest. Keep the manifest and document data selection separately; supplied hashes verify identity against the manifest, not provenance or license. GuitarSet 1.1.0, Xi et al., CC BY 4.0; annotations can contain errors.

## Reproducibility identifiers

- Audio manifest SHA-256: `f1c44bc3fc69b070ad43db053dfab077140abb1fc52c30d8b35fb952a08e4633`
- Model weights SHA-256: `bbd482c786b895cf7d8f44185073d951adae2ebb8a66f82ca84cd1f84569549c`

Hashes identify bytes, not authorization, license, dataset completeness, or split independence.
