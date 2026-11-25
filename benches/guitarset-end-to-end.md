# GuitarSet end-to-end: audio → transcription → tablature (held-out players)

**What this measures.** The whole pipeline, not the fingerer in isolation: a
10-second mono-microphone GuitarSet excerpt goes into the `small` MuScriptor
checkpoint through `mtab`'s own frontend and decoder, then through the string/
fret solver, and the resulting tablature is scored against the annotation.

**What it is not.** TabCNN/TDR protocol compliance, or a measurement of
`medium`/`large`. Decoding is greedy, seed 0, a guitar-family mask and a
2000-token cap, standard tuning, capo 0, 22 frets. No source separation, no
dynamics, no technique inference. The `tab-labeler-v3` comparison is a
*separate* experiment that supplies both fingerers with these same predicted
notes — see [`m5-tab-labeler-v3-head-to-head.md`](m5-tab-labeler-v3-head-to-head.md);
this document does not measure AutoTab's audio frontend.

## Provenance

| Item | Value |
| --- | --- |
| Corpus | GuitarSet 1.1.0 (Xi et al.), Zenodo 3371780, CC BY 4.0 |
| Split | Sessions 03–05 = **unseen players** (180 excerpts); sessions 00–02 are the tuning half |
| Excerpts | 60 consecutive entries of the 180, 10 s each, trimmed at the first annotated onset (the earlier 40 are a prefix of these) |
| Audio | `*_mic.wav` mono microphone take, 44.1 kHz 16-bit, audio SHA-256 verified per entry |
| Annotation | per-clip `*.excerpt.jams`, note times rebased onto the clip and clipped to the window |
| Model | `MuScriptor/muscriptor-small` safetensors, 411,888,600 B, SHA-256 verified |
| Runner | Local run on Linux x86-64 at commit `a0a3993`; the same job is wired into `nightly.yml` → `end-to-end-eval`, which is where the original 40-entry figures came from |
| Manifest | [`eval/guitarset-e2e-60.json`](../eval/guitarset-e2e-60.json) — 60 entries with audio and annotation SHA-256 |
| Report | [`eval/guitarset-e2e-60-report.md`](../eval/guitarset-e2e-60-report.md) — the run these numbers come from |

Regenerate the clips and score them:

```sh
python3 tools/make_guitarset_excerpt_manifest.py \
  --audio audio --annotations annotations \
  --out eval/heldout.json --clips eval/clips \
  --seconds 10 --sessions 03,04,05
python3 -c 'import json;m=json.load(open("eval/heldout.json"));json.dump(m[:60],open("eval/run.json","w"))'
./target/release/mtab eval --corpus guitarset --audio-manifest eval/run.json \
  --weights "$RUNNER_TEMP/mtab-small.safetensors" --size small \
  --max-gen-len 2000 --partial-fingering --report eval/report.md
```

## Results

60 excerpts, 60,000 reference note-frames, 2,807 reference notes. Fingering is
**strict, then span-four partial retry**, so every excerpt is scored and none is
zeroed by one unplaceable note.

| Micro-aggregated metric | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-pitch frames | 66,266 | 32,971 | 24,091 | 66.78% | 73.34% | **69.90%** |
| Tab position frames | 45,589 | 53,651 | 44,988 | 45.94% | 50.33% | **48.03%** |
| Pitch + onset notes | 2,446 | 946 | 623 | 72.11% | 79.70% | **75.72%** |
| Pitch + onset + offset notes | 1,266 | 2,126 | 1,803 | 37.32% | 41.25% | 39.19% |
| Tab + onset notes | 1,635 | 1,757 | 1,434 | 48.20% | 53.27% | **50.61%** |
| Tab + onset + offset notes | 829 | 2,563 | 2,240 | 24.44% | 27.01% | 25.66% |

TDR-style ratio 45,589 / 66,266 = **68.80%**. Wrong-string among pitch-matched
string instances 20,677 / 66,266 = **31.20%**.

### The same 60 excerpts, three ways

The first 40 excerpts are a prefix of these 60, and the 57 excerpts whose
fingering succeeds are byte-identical across the last two rows — only the pitch
counts of the three failures change, so the decomposition below is exact.

| Build | Fingering | Excerpts | Multi-pitch F1 | Tab-position F1 |
| --- | --- | ---: | ---: | ---: |
| Pre-seed, blind start | strict | 40 | 65.30% | **40.25%** |
| Seeded at fret 7 | strict | 40 | 66.34% | 42.58% |
| Seeded at fret 7 | strict | 60 | 68.04% | 45.34% |
| **Seeded at fret 7** | **strict + partial** | **60** | **69.90%** | **48.03%** |

Read down the tab-position column and the three improvements are separable: the
seed is worth **+2.33** on the original 40, the larger and more representative
excerpt set **+2.76** on top of that (the extra 20 excerpts are easier), and the
partial retry **+2.69** more by scoring the three entries that used to count as
complete losses instead of partial ones.

Multi-pitch moves for a different reason: it first rises 65.30% → 66.34%
because entries 19, 31 and 39 are still scored as empty in a 60-entry run — the
new twenty excerpts are easier — and only reaches 69.90% when the three
recover. **The pitch rows are not independent evidence of better transcription;
the notes are the same notes.**

The pitch rows are identical to the earlier run, which is the check that the
open-string retune only touched string choice: multi-pitch frames are 35,190 TP
in both. Tablature moved +1.17 points on pitch+onset (44.96% → 46.13%) and
wrong-string fell 0.91, from that retune (see "the open-string term" in
[`guitarset-oracle.md`](guitarset-oracle.md)). The oracle experiment predicted a
much larger gain (65.90% → 73.40% given ground-truth notes), and the gap between
that prediction and the +1.2 measured here is the honest measure of how much of
the end-to-end loss is the solver's *policy* versus the notes it is handed.

## The fingering solver is the bottleneck, not the model

The per-entry spread is what says where the loss is. Across all 60 entries:

| | Median | Min | Max |
| --- | ---: | ---: | ---: |
| Multi-pitch F1 (pitch content) | 71.5% | 52.6% | 89.3% |
| Tab-position F1 (string + fret) | 51.3% | 1.8% | 88.4% |

The mean gap between the two is **23.2 points**, down from 28.2 on the first
37 scored entries. Six entries still lose 60 points or more to string choice.
The worst is a comping take at 98 reference notes in ten seconds — a dense
strummed pattern where every note is a separate attack — and the rest are now
an even mix of solo and comping takes rather than the solo-heavy picture the
40-entry run suggested:

| Entry | Take | Reference notes | Multi-pitch F1 | Tab F1 | Gap |
| ---: | --- | ---: | ---: | ---: | ---: |
| 37 | `03_Rock1-130-A_comp` | 98 | 77.5% | 1.8% | 75.7 |
| 26 | `03_Jazz1-130-D_solo` | 33 | 83.9% | 17.1% | 66.8 |
| 38 | `03_Rock1-130-A_solo` | 27 | 74.6% | 9.9% | 64.7 |
| 54 | `03_SS2-107-Ab_solo` | 18 | 65.1% | 3.5% | 61.6 |
| 57 | `03_SS3-84-Bb_comp` | 39 | 64.9% | 4.0% | 60.8 |

Pitch is essentially right and the string choice still fails. Split by take
type, the failure is on the melodic side:

| Take | Entries | Mean multi-pitch F1 | Mean tab F1 | Gap |
| --- | ---: | ---: | ---: | ---: |
| `comp` (chords) | 30 | 67.7% | 45.1% | 22.6 |
| `solo` (single line) | 30 | 74.6% | 50.7% | 23.9 |

Solo takes are *easier* for the model — 74.6% pitch against 67.7% — and that
carries through to tab, where they are also 5.6 points higher. **The gap is
essentially the same on both take types** (22.6 against 23.9), so the earlier
reading of the 40-entry run, that solo takes are where string choice collapses,
does not survive the larger sample. What looked like a melodic-specific failure
was mostly the three entries that failed outright, two of them solo.

Both take types lose the same ~23 points to string choice, which is what the
4×4 hand-window beam and the two-string reach penalty predict: the solver has
one mechanism for keeping a hand in place and it works equally badly on chords
and on melodies. The narrowed sequencing problem is
[in the head-to-head](m5-tab-labeler-v3-head-to-head.md): mtab keeps a coherent
position, but on some excerpts it is the *wrong* position.

The 23-point gap is *not* primarily a tuning problem. The solver is tuned on
sessions 00–02 and evaluated here on 03–05, but on oracle notes it reaches
76.23% on this exact split; the loss is in what the model hands it.

## Three entries fail outright

3 of the 60 recordings (19, 31, 39 — 192 notes, 6.8% of the reference notes)
cannot be assigned by the strict bounded beam at all (`no feasible string
remains under overlap/span constraints`). All three are `comp` takes whose
predicted notes overlap heavily on one string; two have no notes outside the
guitar's range.

The published run above uses `mtab eval --partial-fingering`, which retries a
failed excerpt with per-note omissions. It does not quietly discard the excerpt:
the omitted notes are counted in the per-entry *Omitted generated notes* column
(15, 1 and 9 for the three), excluded from true positives, and the entry is
still scored against every reference note. Scored that way the three entries
are 47.8%, 58.0% and 73.3% on tab position instead of 0%. Strict scoring is the
harsher reading and both are defensible; what is not defensible is pretending
the notes were never generated.

One more class needed handling: the decoder occasionally promotes a pitch
outside the instrument entirely (MIDI 22, 117 and 127 have been observed on
real excerpts). A note like that can never be a true positive on any string, so
`assign_for_evaluation` drops it and counts it as an omission — before that,
one such note made an entire recording unassignable.

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
