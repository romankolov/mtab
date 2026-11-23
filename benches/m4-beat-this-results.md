# External Beat This! Rust small-model diagnostic

The [precommitted protocol](m4-beat-this-protocol.md) evaluated the external
`beat-this-rs` 1.1.0 release-mode executable, not an implementation inside
mtab. Full [development](beat-this-results/development.md) and
[second-take](beat-this-results/second-take.md) reports include per-recording
counts and hashes. Linux x86-64, Rust 1.100.0, Python 3.13.18, NumPy 2.3.5,
SciPy 1.16.3 and mir_eval 0.8.2 were used. The second-take set was *not blind*:
one file was inspected for a smoke test before the protocol commit.

| Selection | Beat TP/FP/FN | Beat micro F1 | Downbeat TP/FP/FN | Downbeat micro F1 |
|---|---|---:|---|---:|
| Development, 50 | 1548/308/332 | 82.87% | 359/370/91 | 60.90% |
| Second take, 50 | 1432/486/438 | 75.61% | 301/516/139 | 47.89% |

For orientation, mtab's earlier experimental strongest-autocorrelation
candidate with tightness 30 scored beat micro F1 48.87% and 53.31% on the
same selections. Both methods use identical trimmed references and ±70 ms
matching, but this is **not a fair held-out comparison**:

- [The Beat This! paper](https://arxiv.org/abs/2407.21658) says it trained
  on GuitarSet **comping** recordings, excluding solos. The
  [official model description](https://github.com/CPJKU/beat_this#available-models)
  says `small1` was trained on all the paper datasets except GTZAN and warns
  that results on training datasets can be unfairly good. The third-party port
  claims to export `small1`, but our tests have not independently established
  weight or preprocessing parity with that checkpoint. Whether the exact input
  audio channel here matches the training audio was not established.
- A **post-hoc**, descriptive breakdown shows comp vs solo beat micro F1 of
  98.73% vs 66.74% on development and 94.75% vs 56.65% on second takes.
  Corresponding downbeat F1 is 95.74% vs 37.80%, and 87.78% vs 23.00%.
  Solo performances share players, styles and musical context with comping;
  they are not a performer- or project-held-out set either.
- References are click-derived and constant-tempo; these results do not test
  rubato, pickups, non-guitar mixes, robust meter or reliable tempo maps.

The large comp/solo gap makes the aggregate unsuitable for choosing a default
tracker. These experiments identify a candidate for further work, not a
replacement for the required built-in automatic tracking. In particular,
downbeat precision is low on the solo subset, Rust 1.89 exceeds the project's
1.75 MSRV, the ONNX runtime/models are not integrated, and automatic output
must not be silently promoted to notation. Independent parity and disjoint
real-audio evaluation are still needed.
