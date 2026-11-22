# Reproducible GuitarSet oracle-note evaluation (2026-11-22)

This evaluates **only string/fret assignment given ground-truth notes**, not
MuScriptor transcription, tuning detection, end-to-end audio-to-tab quality,
TabCNN metrics, or the tab-labeler-v3 model. These results do not satisfy the
project's full GuitarSet acceptance criterion. In particular, the wrong-string
rate conditions on *assigned* notes, and the strict beam search failed 46 complete
excerpts; use the all-notes accuracy and unassigned counts alongside it. The
naive greedy baseline ignores playability and has 7,623 same-string collisions.

Dataset: GuitarSet v1.1.0 by Qingyang Xi, Rachel M. Bittner, Xuzhou Ye, Johan
Pauwels and Juan P. Bello, [Zenodo 3371780](https://zenodo.org/records/3371780),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset authors
note known annotation errors; this unfiltered run includes all 360 excerpts.
The downloaded `annotation.zip` MD5 was `b39b78e63d3446f2e54ddb7a54df9b10`
(published by the dataset) and SHA-256 was
`8daa02e6417ccca1685feb44b135e95928ad7037e5032ecb326b5791856fda99`.
For local space efficiency, non-`note_midi` annotations were removed from the
local JAMS copies; all six `note_midi` objects and note data were unchanged.
An earlier strict-search run over all 360 full original files produced a
byte-identical report to the note-only copies; this expanded adaptive report
was generated from the same note-only copies. No dataset files are included in
the repository.

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
annotation-only evaluation. The separate end-to-end `run_eval` is not
implemented. Evaluation uses the annotation's string index (low E = 0), rounds
reference MIDI pitch to the nearest semitone, preserves annotation times, and
assumes standard tuning, capo 0. The adaptive mode first tries a 4-fret span,
then retries with a 7-fret limit; every recovered excerpt actually used a span
above four frets. Such grips may be uncomfortable or unplayable in practice.
The independent greedy baseline chooses the
lowest fret per note without preventing overlaps. Failed beam-search excerpts
count every note as unassigned, rather than excluding those tracks.


The diagnostic reports the state of the *reference* at the first solver failure:
38 excerpts have an active fretted span greater than four, and eight have a
same-string overlap. These conditions can overlap and are not proof of the
search failure's cause. Several overlaps are at floating-point boundary scale;
the evaluator currently retains the source times rather than silently altering
them. The four-fret constraint itself can exclude the reference fingering.

---

# GuitarSet oracle-note string assignment

Reference pitches and onset/offsets, not model predictions, are provided to all methods.
String accuracy is correct assignments divided by **all** reference notes; unassigned notes count as incorrect.
Wrong-string rate is incorrect assigned notes divided by assigned notes. Beam-search failures mark the entire excerpt unassigned.
Adaptive search retries failed excerpts with a seven-fret span, which may be uncomfortable and needs review.
The greedy baseline chooses the lowest fret per note independently and does not enforce overlaps.
This is not end-to-end accuracy, a TabCNN/TDR metric, or a tab-labeler-v3 comparison.

- Annotation files: 360
- Reference notes: 62476
- Beam-search failed excerpts: 46
- Adaptive retry recovered excerpts: 41
- Recovered excerpts with actual span above four: 41
- Largest actual span in recovered excerpts: 7 frets
- Adaptive retry still failed excerpts: 5
- Greedy notes overlapping another note on the same string: 7623
- At failure, reference fretted span exceeds four: 38 excerpts
- At failure, reference has a same-string overlap: 8 excerpts
- At failure, reference has neither condition: 0 excerpts

These reference diagnostics may overlap and are not proof of failure cause.

| Method | Assigned | Unassigned | Correct strings | String accuracy (all notes) | Wrong-string rate (assigned) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bounded beam | 53176 | 9300 | 35945 | 57.53% | 32.40% |
| Adaptive span 4→7 | 60929 | 1547 | 41174 | 65.90% | 32.42% |
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
- `00_Funk3-98-A_comp.jams`: note 170: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference same-string overlap); recovered with wider span
- `00_Funk3-98-A_solo.jams`: note 30: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `00_Jazz1-130-D_comp.jams`: note 147: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference same-string overlap); retry failed: note 147: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path)
- `00_Jazz2-110-Bb_solo.jams`: note 63: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference same-string overlap); recovered with wider span
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
- `01_SS1-68-E_comp.jams`: note 118: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference same-string overlap); retry failed: note 118: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path)
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
- `03_Rock3-148-C_comp.jams`: note 131: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference same-string overlap); retry failed: note 131: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path)
- `04_Rock1-130-A_comp.jams`: note 156: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `04_SS1-68-E_comp.jams`: note 219: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); retry failed: note 219: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path)
- `04_SS2-107-Ab_comp.jams`: note 200: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `05_BN3-119-G_comp.jams`: note 123: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `05_Funk2-108-Eb_comp.jams`: note 150: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference same-string overlap); recovered with wider span
- `05_Rock2-85-F_solo.jams`: note 10: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `05_SS1-100-C#_solo.jams`: note 35: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
- `05_SS1-68-E_solo.jams`: note 69: no feasible string remains under overlap/span constraints (or the beam pruned the feasible path) (reference span >4); recovered with wider span
