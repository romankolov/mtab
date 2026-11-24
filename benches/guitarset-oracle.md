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

## Feasibility decomposition

An excerpt is **infeasible** when its annotations ask for something no left hand could produce at some instant: a fretted span over four frets between simultaneously sounding notes (open strings excluded — a ringing open string costs no finger), or two notes overlapping in time on the same string. GuitarSet was recorded with a hexaphonic pickup, so a note on one string can also appear on a neighbouring string's annotation; the duplicate-pitch two-string case that dominates the span failures is most likely that bleed, and no fingering could satisfy it.

- Infeasible excerpts: 111 of 360 (30.83%)
- ...span over four only: 101
- ...same-string overlap only: 8
- ...both: 2
- Feasible excerpts: 249
- Reference notes in feasible excerpts: 41138

The same three methods, scored only on the feasible excerpts:

| Method | Feasible notes | Correct strings | String accuracy | Wrong-string rate (assigned) |
| --- | ---: | ---: | ---: | ---: |
| Strict beam (span 4) | 41138 | 28088 | 68.28% | 31.72% |
| Partial retry (span 4) | 41138 | 28088 | 68.28% | 31.72% |
| Per-note lowest-fret greedy | 41138 | 19021 | 46.24% | 53.76% |

Comparing the two tables shows how much of the headline gap is the reference data and how much is the solver.

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
