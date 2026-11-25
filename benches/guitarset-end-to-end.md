# GuitarSet end-to-end: audio → transcription → tablature (held-out players)

**What this measures.** The whole pipeline, not the fingerer in isolation: a
10-second mono-microphone GuitarSet excerpt goes into the `small` MuScriptor
checkpoint through `mtab`'s own frontend and decoder, then through the string/
fret solver, and the resulting tablature is scored against the annotation.

**What it is not.** TabCNN/TDR protocol compliance, a `tab-labeler-v3`
comparison, or a measurement of `medium`/`large`. Decoding is greedy, seed 0, a
guitar-family mask and a 2000-token cap, standard tuning, capo 0, 22 frets. No
source separation, no dynamics, no technique inference.

## Provenance

| Item | Value |
| --- | --- |
| Corpus | GuitarSet 1.1.0 (Xi et al.), Zenodo 3371780, CC BY 4.0 |
| Split | Sessions 03–05 = **unseen players** (180 excerpts); sessions 00–02 are the tuning half |
| Excerpts | 40 consecutive entries of the 180, 10 s each, trimmed at the first annotated onset |
| Audio | `*_mic.wav` mono microphone take, 44.1 kHz 16-bit, audio SHA-256 verified per entry |
| Annotation | per-clip `*.excerpt.jams`, note times rebased onto the clip and clipped to the window |
| Model | `MuScriptor/muscriptor-small` safetensors, 411,888,600 B, SHA-256 verified |
| Runner | GitHub Actions `nightly.yml` → `end-to-end-eval`, run [45490499979](https://github.com/romankolov/mtab/actions/runs/45490499979), commit `fc98709` |

## Results

40 excerpts, 42,000 reference note-frames, 1,961 reference notes.

| Micro-aggregated metric | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-pitch frames | 35,190 | 20,777 | 16,616 | 62.88% | 67.93% | **65.30%** |
| Tab position frames | 21,707 | 34,260 | 30,183 | 38.79% | 41.83% | **40.25%** |
| Pitch + onset notes | 1,468 | 499 | 493 | 74.63% | 74.86% | **74.75%** |
| Pitch + onset + offset notes | 753 | 1,214 | 1,208 | 38.28% | 38.40% | 38.34% |
| Tab + onset notes | 906 | 1,061 | 1,055 | 46.06% | 46.20% | **46.13%** |
| Tab + onset + offset notes | 440 | 1,527 | 1,521 | 22.37% | 22.44% | 22.40% |

TDR-style ratio 21,707 / 35,190 = **61.69%**. Wrong-string among pitch-matched
string instances 13,483 / 35,190 = **38.31%**.

The pitch rows are identical to the earlier run, which is the check that the
open-string retune only touched string choice: multi-pitch frames are 35,190 TP
in both. Tablature moved +1.17 points on pitch+onset (44.96% → 46.13%) and
wrong-string fell 0.91, from that retune (see "the open-string term" in
[`guitarset-oracle.md`](guitarset-oracle.md)). The oracle experiment predicted a
much larger gain (65.90% → 73.40% given ground-truth notes), and the gap between
that prediction and the +1.2 measured here is the honest measure of how much of
the end-to-end loss is the solver's *policy* versus the notes it is handed.

## The fingering solver is the bottleneck, not the model

The per-entry spread is what says where the loss is. Across the 37 entries that
completed:

| | Median | Min | Max |
| --- | ---: | ---: | ---: |
| Multi-pitch F1 (pitch content) | 71.3% | 52.9% | 85.2% |
| Tab-position F1 (string + fret) | 42.8% | 1.8% | 84.0% |

The mean gap between the two is **28.2 points**, and it is not uniform. Four of
the five worst entries are **solo** takes — a single monophonic line with no
simultaneous pitches to anchor a shape. The single exception is entry 37, a
comping take at 98 reference notes in ten seconds, which is a dense strummed
pattern where every note is a separate attack:

| Entry | Take | Reference notes | Distinct pitches | Multi-pitch F1 | Tab F1 | Gap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 37 | `03_Rock1-130-A_comp` | 98 | 10 | 77.5% | 1.8% | 75.7 |
| 38 | `03_Rock1-130-A_solo` | 27 | 10 | 74.6% | 9.9% | 64.7 |
| 26 | `03_Jazz1-130-D_solo` | 33 | 16 | 83.9% | 12.0% | 71.8 |
| 10 | `03_BN3-119-G_solo` | 24 | 11 | 77.2% | 12.9% | 64.3 |

Pitch is essentially right and the string choice still fails. Split by take
type, the failure is on the melodic side:

| Take | Entries scored | Mean multi-pitch F1 | Mean tab F1 | Gap |
| --- | ---: | ---: | ---: | ---: |
| `comp` (chords) | 17 of 20 | 66.9% | 42.5% | 24.4 |
| `solo` (single line) | 20 of 20 | 73.5% | 42.0% | 31.5 |

Solo takes are *easier* for the model — 73.5% pitch against 66.9% — and still
land lower on tab. A single melodic line removes every constraint the fingerer
relies on: there are no simultaneous notes to force a shape, and a recurring
pitch says nothing about which of its three or four positions the player used.
The cost of *leaving* a position is only the reach to the next note, so with
nothing to hold the hand steady the frame drifts.

That is the concrete case for a sequence-level or melodic prior — the current
reward for staying in a position is too weak when a take is a bare melody. It also explains why the oracle-note result (70.76% string accuracy on
held-out players, given ground-truth notes) does not carry over: predicted notes
include 11% more than the reference and spurious pitches, so the solver has both
weaker pitch cues and more collision pressure than the oracle setting.

The 29.8-point gap is *not* primarily a tuning problem. The solver is tuned on
sessions 00–02 and evaluated here on 03–05, but on oracle notes it reaches
70.76% on this exact split; the loss is in what the model hands it.

## Three entries fail outright

3 of 40 recordings (19, 31, 39 — 192 notes, 9.8% of the reference notes) end with
the strict bounded beam unable to place a note (`no feasible string remains
under overlap/span constraints`). They are scored as empty predictions rather
than dropped, which is the honest default, but it means 9.8% of the reference
notes contribute guaranteed false negatives. `mtab eval --partial-fingering`
exists to place what it can and report the omissions; it is off in this run so
that the number is not inflated by silently omitted notes.

## Correction history — read this before quoting an older number

The first version of the manifest tool trimmed each clip at the first annotated
onset and then paired it with the **untrimmed** JAMS file. Clips with a lead-in
(24 of the first 40, median 551 ms) were therefore scored against annotations
that still counted from the recording's t=0, and correct notes were counted as
half a second early. The first published run reported **39.33%** multi-pitch F1
and **23.92%** tab-position F1 for that reason. It is withdrawn; the numbers
above supersede it, and the tool now writes rebased annotations and five offline
tests pin the behaviour (`tools/test_make_guitarset_excerpt_manifest.py`).

A same-audio A/B on 7 excerpts, scored both ways from identical predictions:

| Pairing | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| old, unrebased | 67.98% | 70.23% | 69.09% |
| rebased | 83.99% | 86.77% | **85.36%** |

Worst single excerpt: a solo take at 4.44% → 88.89%. That scale of error is why
the pairing is now hashed per entry and asserted in tests.

## Reproducing

```sh
python3 tools/make_guitarset_excerpt_manifest.py \
  --audio <dir of *_mic.wav> --annotations <dir of *.jams> \
  --out eval/heldout.json --clips eval/clips --seconds 10 --sessions 03,04,05

mtab eval --corpus guitarset --audio-manifest eval/heldout.json \
  --weights muscriptor-small.safetensors --size small \
  --max-gen-len 2000 --report eval/report.md
```

Or dispatch `nightly.yml` with `run_end_to_end=true` and `eval_entries=40`. The
job uploads `report.md` unconditionally so a partially failing run still
publishes its numbers.

GuitarSet 1.1.0, Xi et al., CC BY 4.0; annotations can contain errors. The
reference is fixed MIDI pitches from `note_midi`, not continuous F0.
