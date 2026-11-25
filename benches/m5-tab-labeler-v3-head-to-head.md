# Head to head with `tab-labeler-v3`, on mtab's own notes

PROMPT.md §9 asks for two things the project has never had: tablature accuracy
≥ autotab-rust's `tab-labeler-v3`, and wrong-string rate below a per-note greedy
baseline. §6.6 describes the gap in mechanism terms — the solver is a per-note
beam search over string/fret choices, and what a player does is choose a
*shape*, holding the hand in one position across a phrase.

This document answers both criteria with one experiment, and the answer is that
**the first one fails and the second one passes**.

## Method

Both fingerers are given *the same notes* — mtab's own transcriptions of 57
GuitarSet excerpts — and scored by the same code.

1. `diag/nN.jsonl` holds notes that `mtab eval` already produced from audio.
2. `tools/autotab_head_to_head.py` writes those notes to MIDI (PPQ 25,000 at
   500,000 µs/beat, 20 µs grid, note-off before note-on) and runs the
   **unmodified** `autotab` CLI at commit
   `a1eebbebbd74db669a6a5f74079eed907717cfb3` with `--tab-engine v3 --device
   cpu --threads 1 --midi-track all`.
3. AutoTab's assignment is converted to mtab's `TabDocument` IR — string
   1 (high E) … 6 (low E) remapped onto GuitarSet's 0 (low E) … 5 (high E),
   extra-guitar notes dropped and counted.
4. `mtab eval --tab-manifest` scores the converted IR against the same
   annotation it scored mtab's own IR against.

Because step 1 is shared, **the pitch rows are identical by construction and
serve as the control**: multi-pitch TP/FP come out at 61,451 / 29,310 for both
sides. Any difference below is string choice, not note recognition.

`--emit-ir-dir` was added to the audio harness to make step 4 possible; it dumps
each entry's IR even when scoring fails.

## Result — 57 scored excerpts, 2,448 reference notes of 84,323 frames

| Metric | mtab | AutoTab v3 | Difference |
| --- | ---: | ---: | ---: |
| Multi-pitch frames F1 | 70.20% | 70.20% | control, 0.00 |
| Pitch + onset note F1 | 77.55% | 77.55% | control, 0.00 |
| **Tab-position frames F1** | **47.05%** | **58.92%** | **+11.87** |
| **Tab + onset note F1** | **50.65%** | **64.62%** | **+13.97** |

Both tab rows are computed on the 57 entries that scored; the three entries
whose fingering failed are common to both sides and are excluded from *both*,
so neither is credited for them.

Per entry, AutoTab's tab-position F1 is higher on 33 of 54 comparable entries,
lower on 17, and within a point on 4. The mean per-entry advantage is 9.8
points and the median 8.8.

## Why the §9 wrong-string criterion still passes

The criterion is "wrong-string rate below the per-note greedy baseline", and
that baseline is not AutoTab. On the oracle-note corpus the greedy baseline
scores **53.76%** wrong-string; mtab's strict beam scores **23.77%**, and
end-to-end at 10 s, **32.89%**. Both are far below greedy and the criterion is
met — it was met before this run.

AutoTab's own wrong-string rate on the shared notes is **43.10%** (39,115 of
90,761 pitch-matched frame instances). That is *worse* than mtab's 32.89% on
the same transcripts. Yet its tab-position F1 is 11.87 points higher.

That combination is the interesting part, and it says something specific about
which errors matter.

## What AutoTab is doing differently

On a single excerpt (entry 1) the players stay high on the neck: the reference
mean fret is **6.62** (median 7, range 4–8), and AutoTab reproduces that with
mean fret **6.33** (median 7). mtab plays the same notes at mean fret **3.11**.

The specific shift is down and toward thin strings. The reference takes 67 on
string 4 fret 8, 62 on string 3 fret 7 and 51 on string 1 fret 6; mtab takes
67 on string 5 fret 3, 62 on string 4 fret 3 and 51 on string 2 fret 1 — a
coherent, playable shape, displaced about five frets toward the headstock.

So mtab's error is a *global position shift* that is internally consistent:
every note of the excerpt is played as a shape a hand could actually make. It
is simply the wrong shape, and a framewise metric reads that as a total loss.
Entry 1 scores 24.6% for mtab and 75.6% for AutoTab.

AutoTab loses points differently. Its 43% wrong-string rate is spread thin:
it puts individual notes in places a hand would not go, but it lands the global
position right, so the notes it does get right score. Partial credit everywhere
beats total credit nowhere, and F1 rewards the first.

The correlation between AutoTab's mean fret on an entry and its margin over
mtab is **−0.10** — essentially zero — so this is not a matter of "play higher
and you win"; an excerpt where AutoTab sits high and one where it sits low
contribute equally. What separates them is calibration: AutoTab's position
choices line up with the players' on each excerpt, while its within-position
choices are worse than mtab's.

## What this implies for the solver

§6.6's "shape-aware, not per-note" framing is the right diagnosis but the
current implementation has the priority backwards. mtab already has what
AutoTab lacks — a sequence-level search that keeps one hand position coherent
(hence the much better wrong-string rate within a position) — and lacks what
AutoTab has: a prior over *which* position a passage is in.

The seed at fret 7 was a global version of that prior and bought +4.0/+3.2
points on the two halves. This run shows what a global constant cannot do: the
corpus *median* is 6, but this excerpt sits at 4–8 with its centre at 7, and
mtab answers 3.1. A constant seed prices the first note's distance from fret 7
and then lets the beam drift, and on a repeated figure that repeats an
already-wrong shape it never pays to leave it.

Two candidate fixes, both testable against this harness in about 20 minutes:

1. **Per-excerpt position estimate.** Before the beam, choose the window from
   the pitch content (e.g. the lowest sounding pitch of the first chord, or a
   histogram over the whole excerpt) instead of a constant 7. This is the
   direct answer to the failure above and would have put entry 1 at frets 0–3.
2. **Score the whole excerpt, not per onset.** Let the beam keep multiple hand
   positions alive across the excerpt and choose at the end by total cost,
   with a term for how much the hand must travel. That is closer to what a
   player optimises and to what the tab-labeler is doing.

## The obvious fix does not work, and that is informative

The first fix to try is to estimate the position from the excerpt's own pitches:
take each note's lowest playable fret and use the median. It makes the solver
**worse** — 73.55% → 63.48% on the tuning half — and the reason is visible
before writing any solver code:

| Quantity | Mean across 180 excerpts |
| --- | ---: |
| Reference fretted median | **6.03** |
| Median of each note's lowest playable fret | **2.30** |
| Median of each note's lowest playable fret, open strings excluded | 2.84 |
| Reference minus the lowest-fret estimate | +2.76 (sd 2.30) |

The gap is positive and *variable*. No constant correction repairs it, and that
is the point: which position a player used for a note is **not a function of the
note's pitch**. It depends on the chord shape being held, on where the phrase
came from, on string choice for tone — none of which are in a pitch list. A
pitch histogram answers "how low *could* this be played", never "how high *did*
they play it".

That is exactly the information `tab-labeler` gets from training on GuitarSet's
string labels, and exactly what mtab's hand-window beam is trying to reconstruct
online. The seed at fret 7 beats the pitch-derived estimate because the corpus
prior (median 6) is a better predictor of an excerpt's position than that
excerpt's own pitches are.

`MTAB_SEED_FROM_PITCHES` and `seed_from_pitches` were written, measured and
removed; the numbers above are why.

### The fret prior is not the culprit either

The other obvious candidate is the low-fret prior, `0.015 × fret` per note,
which does bias every choice downward. Removing it **loses** 1.65 points
(73.55% → 71.90% on the tuning half), and it turns out to be propping up the
seeded start rather than fighting it:

| Start window | Fret prior | Tuning-half accuracy |
| --- | ---: | ---: |
| seeded at 7 (shipped) | 0.015 (shipped) | **73.55%** |
| seeded at 7 | 0.0 | 71.90% |
| blind (pre-seed) | 0.015 | 68.90% |
| blind | 0.0 | 62.14% |

Both knobs earn their place and neither substitutes for the other: the seed
prices the *first* note's distance from the window, the prior taxes every
*fretted* note, and on a passage that stays in one position the prior is what
pays for holding it. Removing either one alone loses more than the other
gains. That also rules out "just delete the bias" as the fix for the wrong-
position failures.

## Caveats — read before quoting

- **AutoTab's v3 model was trained on GuitarSet** and its model card says
  guitarist 05 was its validation player. Its advantage here is therefore
  not a generalization estimate; mtab's side is uncontaminated by comparison.
  It is still the number §9 asks to be compared against.
- AutoTab receives clean note lists. Its audio frontend was not run, and this
  measures only the fingerer.
- The manifest is 57 of 60 excerpts from **players 03–05**, all `small` model,
  greedy decoding. It is not the whole corpus.
- 20 µs MIDI rounding is shared by both sides and cannot favour either.
- AutoTab produced 0 extra-guitar notes, 0 unassigned notes and 0 unmatched
  assignments across all 57 entries, so the conversion had no losses to
  report.

## Reproduction

```sh
python3 tools/autotab_head_to_head.py \
  --transcripts <dir of nN.jsonl> --annotations <dir of eN.jams> \
  --entries 1,2,3,... \
  --executable <autotab> --autotab-home <dir with models/tab-labeler-v3-egset.onnx> \
  --runtime-dir <onnxruntime lib dir> --mtab <mtab> --work <scratch dir>
```

The tool verifies the model SHA-256
`93114786c6f4781e7a0b69abd7e93a0d4966873f8e54908f97a1de8e8cf788af` before
running and records the executable hash in its output.
