# Tuning and capo detection

`mtab detect-tuning song.wav` and the automatic default in `mtab transcribe`
answer one question: over which set of open strings was this recorded?

It is a **heuristic ranking**, not a classifier, and it is not calibrated.

## Method

1. Pitch evidence is collected as a 128-bin histogram of MIDI pitch weight from
   the decoded/generated notes or a monophonic pitch track. Each bin is the
   amount of evidence at that absolute pitch.
2. Candidate hypotheses are the built-in tuning library crossed with capo
   positions 0–12. A capo position raises every open-string pitch by that many
   semitones.
3. A candidate scores the **fraction of all pitch weight that lands on its
   sounding open strings**. It matches better when the phrase actually used
   open strings.
4. `sufficient_evidence` requires all of: at least 8 total weight, at least 4
   distinct matched open pitches, at least 75% open fraction, at least two
   candidates, and a lead of at least 0.10 over the runner-up. Otherwise the
   detector abstains and the CLI warns and falls back.

The candidate list keeps acoustically equivalent templates separate (for
example standard tuning with capo 2 and a tuning that sounds the same), so a
tie cannot be reported as confidence.

`mtab detect-tuning --list` prints the library. `--tuning` and `--capo` pin one
half of the hypothesis, and the detector then ranks the other half.

## Why it can be wrong

- **Fretted notes look like open strings.** A phrase without open strings can
  score well on the wrong tuning.
- **Equivalent setups sound identical.** Capo 2 in E is capo 0 in F# as far as
  pitch is concerned; only context or the player tells them apart.
- **Chords, bass and other instruments blur the histogram.** The evidence is
  not separated by instrument.
- **The pitch evidence itself can be wrong.** Octave errors and the
  monophonic tracker's assumptions propagate directly into the score.

Use it as a ranked suggestion. When it matters, set `--tuning` and `--capo`
explicitly — the assignment then guarantees fret numbers are consistent by
construction.

## Where the code lives

- `crates/mtab-guitar/src/tuning.rs`: the tuning type (open-string MIDI
  pitches, low string first) and the library (`standard`, `drop-d`, `drop-c`,
  `dadgad`, `open-g`, `open-d`, `open-e`, half-step down, seven- and
  eight-string variants).
- `crates/mtab-guitar/src/detection.rs`: scoring, ranking and the sufficiency
  thresholds, plus `standard_candidates()`.
- `crates/mtab-cli/src/main.rs`: the `detect-tuning` command and the
  transcribe-time fallback.

Tests cover silence, ties, equivalent templates, insufficient evidence and the
round trip through the CLI.
