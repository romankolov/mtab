# GuitarSet symbolic chord diagnostic results

This is an **oracle-note, sheet-progression comparison**, not an audio chord
recognition benchmark. Reference notes were supplied to the experimental
library API; no MuScriptor weights or transcribed audio were used. Protocol:
[m7-guitarset-chord-protocol.md](m7-guitarset-chord-protocol.md).
The first GuitarSet chord annotation describes the sheet progression and does
not necessarily equal the guitar's instantaneous pitch set, especially during
solos. `mir_eval.chord.triads` compares root and triad reduction, ignoring
seventh differences and bass inversion. These numbers do not establish chord
recognition quality or release acceptance.

| Group | Files | Comparable frames | Suggested frames | Correct frames | Coverage | Overall agreement | Conditional agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 360 | 1,090,141 | 171,861 | 149,595 | 15.77% | 13.72% | 87.04% |
| Comp | 180 | 546,882 | 167,621 | 148,183 | 30.65% | 27.10% | 88.40% |
| Solo | 180 | 543,259 | 4,240 | 1,412 | 0.78% | 0.26% | 33.30% |

Coverage is suggestions / comparable frames. Overall agreement counts missing
suggestions as `N`; conditional agreement is correct suggestions / suggested
frames. Every frame was comparable, and in this dataset none of the correct
frames came from a no-chord suggestion. No file was skipped. There were 6,836
suggestion intervals, 6,721 in comp and 115 in solo. **Coverage is low**,
particularly in solo; the high conditional comp number must not be reported as
full-track accuracy. Chord-set predictions can also disagree with sheet labels
for legitimate voicing and musical reasons. No automatic CLI integration or
notation export is claimed.

Per-file counts and input hashes:
[chord-results/guitarset-oracle.json](chord-results/guitarset-oracle.json).
A repeated run produced byte-identical JSON. GuitarSet 1.1.0 is CC BY 4.0;
archive MD5 `b39b78e63d3446f2e54ddb7a54df9b10`. The example executable was
built on Linux x86-64 using `cargo build --locked --release -p mtab-guitar
--example chord_dump`, Rust 1.100.0; SHA-256
`91c95b66dc8748e8f64fa29906332d7179b30f4089137f6275e6e4f69a6c44dc`.
Python 3.13.18, NumPy 2.3.5 and mir_eval 0.8.2 were used. Reproduce with
`python tools/evaluate_guitarset_chords.py --annotations annotation.zip
--executable target/release/examples/chord_dump --output report.json`.
