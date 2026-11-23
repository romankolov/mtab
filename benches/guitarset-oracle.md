# Reproducible GuitarSet oracle-note evaluation (2026-11-22)

This evaluates **only string/fret assignment given ground-truth notes**, not
MuScriptor transcription, tuning detection, end-to-end audio-to-tab quality,
TabCNN metrics, or the tab-labeler-v3 model. These results do not satisfy the
project's full GuitarSet acceptance criterion. This same corpus was used for
failure diagnosis and method design, so results are not held-out. In particular, the wrong-string
rate conditions on *assigned* notes, and the strict beam search failed 40 complete
excerpts; use the all-notes accuracy and unassigned counts alongside it. The
naive greedy baseline ignores playability and has 7,623 same-string collisions.

Dataset: GuitarSet v1.1.0 by Qingyang Xi, Rachel M. Bittner, Xuzhou Ye, Johan
Pauwels and Juan P. Bello, [Zenodo 3371780](https://zenodo.org/records/3371780),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset authors
note known annotation errors; this unfiltered run includes all 360 excerpts.
The downloaded `annotation.zip` MD5 was `b39b78e63d3446f2e54ddb7a54df9b10`
(published by the dataset) and SHA-256 was
`8daa02e6417ccca1685feb44b135e95928ad7037e5032ecb326b5791856fda99`.
This updated run used the 360 full, unmodified JAMS files from the verified
archive, with code at commit `3c90fd1` in release mode. No dataset files are
included in the repository. Earlier versions of this report used note-only
copies, which matched an earlier strict run over the original files.

To reproduce with the full original JAMS files (no Python or model weights):

```sh
curl -fL -o annotation.zip 'https://zenodo.org/records/3371780/files/annotation.zip?download=1'
echo 'b39b78e63d3446f2e54ddb7a54df9b10  annotation.zip' | md5sum -c -
mkdir -p annotations
unzip -q -j annotation.zip '*.jams' -d annotations
cargo run --locked --release -p mtab-cli -- eval --corpus guitarset \
  --oracle-notes --annotations annotations --report oracle-report.md
```

On GitHub Actions, manually run `Nightly` with `run_oracle=true` for this
annotation-only evaluation. The separate end-to-end `run_eval` covers only the
checked-in five-second development excerpt and requires authorized small-model
weights; it has not been run on GitHub. Evaluation uses the annotation's string index (low E = 0), rounds
reference MIDI pitch to the nearest semitone, preserves annotation times, and
assumes standard tuning, capo 0. The adaptive mode first tries a 4-fret span,
then retries with a 7-fret limit; every recovered excerpt actually used a span
above four frets. Such grips may be uncomfortable or unplayable in practice.
The independent greedy baseline chooses the
lowest fret per note without preventing overlaps. Failed beam-search excerpts
count every note as unassigned, rather than excluding those tracks.


The diagnostic reports the state of the *reference* at the first solver failure:
38 excerpts have an active fretted span greater than four, and two have a
same-string overlap. These conditions can overlap and are not proof of the
search failure's cause. The solver clips sequential overlaps of at most one nanosecond at note
boundaries to remove arithmetic jitter, and preserves other timing. The
four-fret constraint itself can exclude the reference fingering.


The new partial retry kept the four-fret span and omitted 59 of 62,476 notes,
while recovering some notes in every strict-failure excerpt. Its all-note
string accuracy was 68.50% versus 59.65% strict, but the comparison primarily
reflects strict search counting *entire failed excerpts* as unassigned. The
seven-fret adaptive retry retained all notes for 38 failed excerpts and had
67.04% all-note accuracy; two excerpts still failed completely. These are
alternative trade-offs, not evidence that dropped notes are acceptable or that
this method will improve audio-derived predictions. Search design was informed
by this corpus, and the result is not a held-out test or v3 comparison.

---

# GuitarSet oracle-note string assignment

Reference pitches and onset/offsets, not model predictions, are provided to all methods.
String accuracy is correct assignments divided by **all** reference notes; unassigned notes count as incorrect.
Wrong-string rate is incorrect assigned notes divided by assigned notes. Beam-search failures mark the entire excerpt unassigned.
Adaptive search retries failed excerpts with a seven-fret span, which may be uncomfortable and needs review.
Partial search retries strict failures at span four, explicitly counts skipped notes as incorrect, and has no optimality guarantee.
Fingering clips sequential note-boundary overlaps of at most 1 ns; reference times otherwise remain unchanged.
The greedy baseline chooses the lowest fret per note independently and does not enforce overlaps.
This is not end-to-end accuracy, a TabCNN/TDR metric, or a tab-labeler-v3 comparison.

- Annotation files: 360
- Reference notes: 62476
- Beam-search failed excerpts: 40
- Adaptive retry recovered excerpts: 38
- Recovered excerpts with actual span above four: 38
- Largest actual span in recovered excerpts: 7 frets
- Adaptive retry still failed excerpts: 2
- Partial retry recovered at least one note: 40 excerpts
- Partial retry errors: 0 excerpts
- Greedy notes overlapping another note on the same string: 7623
- At failure, reference fretted span exceeds four: 38 excerpts
- At failure, reference has a same-string overlap: 2 excerpts
- At failure, reference has neither condition: 0 excerpts

These reference diagnostics may overlap and are not proof of failure cause.

| Method | Assigned | Unassigned | Correct strings | String accuracy (all notes) | Wrong-string rate (assigned) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bounded beam | 54666 | 7810 | 37269 | 59.65% | 31.82% |
| Adaptive span 4→7 | 61638 | 838 | 41883 | 67.04% | 32.05% |
| Partial retry span 4 | 62417 | 59 | 42798 | 68.50% | 31.43% |
| Per-note lowest-fret greedy | 62476 | 0 | 28365 | 45.40% | 54.60% |

Settings: standard EADGBE, capo 0, frets 0–22, normal span 4, adaptive retry span 7, beam width 512.
GuitarSet (Xi et al., 2018), Zenodo 3371780, version 1.1.0, CC BY 4.0; annotations are user-provided and known dataset errors may affect results.

## Strict-search failures and adaptive retry

- `00_BN1-129-Eb_solo.jams`: note 6: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_BN1-147-Gb_comp.jams`: note 26: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference same-string overlap); recovered with wider span
- `00_BN2-166-Ab_solo.jams`: note 77: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_BN3-154-E_comp.jams`: note 101: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_Funk1-114-Ab_solo.jams`: note 50: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_Funk2-108-Eb_comp.jams`: note 409: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_Funk2-108-Eb_solo.jams`: note 39: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_Funk3-98-A_solo.jams`: note 30: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_Jazz2-187-F#_solo.jams`: note 48: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_Jazz3-137-Eb_solo.jams`: note 19: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_Jazz3-150-C_solo.jams`: note 31: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_Rock1-130-A_solo.jams`: note 55: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_Rock1-90-C#_comp.jams`: note 522: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_Rock3-117-Bb_comp.jams`: note 295: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_SS1-68-E_comp.jams`: note 136: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_SS2-88-F_comp.jams`: note 50: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_SS2-88-F_solo.jams`: note 15: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_SS3-84-Bb_comp.jams`: note 84: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `01_Funk3-112-C#_solo.jams`: note 50: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `01_Rock1-130-A_solo.jams`: note 80: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `01_SS1-100-C#_comp.jams`: note 280: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `02_BN2-131-B_comp.jams`: note 36: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference same-string overlap); recovered with wider span
- `02_BN2-166-Ab_comp.jams`: note 60: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `02_Rock1-130-A_solo.jams`: note 39: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `02_Rock1-90-C#_comp.jams`: note 173: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `02_Rock2-142-D_solo.jams`: note 28: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `02_SS1-100-C#_solo.jams`: note 40: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `02_SS1-68-E_solo.jams`: note 39: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `02_SS2-107-Ab_comp.jams`: note 186: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `02_SS3-84-Bb_comp.jams`: note 182: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `03_Funk1-97-C_solo.jams`: note 87: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); retry failed: note 87: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path)
- `03_Funk2-119-G_solo.jams`: note 35: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `03_Rock3-117-Bb_solo.jams`: note 37: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `04_Rock1-130-A_comp.jams`: note 156: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `04_SS1-68-E_comp.jams`: note 219: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); retry failed: note 219: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path)
- `04_SS2-107-Ab_comp.jams`: note 200: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `05_BN3-119-G_comp.jams`: note 123: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `05_Rock2-85-F_solo.jams`: note 10: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `05_SS1-100-C#_solo.jams`: note 35: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `05_SS1-68-E_solo.jams`: note 69: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
