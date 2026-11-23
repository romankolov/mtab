# M7 pitch-tracking check: one real GuitarSet excerpt

The test `cargo test --locked -p mtab-cli --test guitarset_pitch -- --nocapture`
uses the existing five-second GuitarSet mono-mic clip and **reference** note
times and string labels. It tests only notes with no overlapping annotated
note on any string and only F0 frames fully inside that note. Each frame center
is paired with the nearest voiced per-string GuitarSet pitch-contour annotation
within 10 ms. Frames are correlated because analysis windows overlap.

On the present Linux run: one isolated reference note (string index 1,
1.027–1.434 s, MIDI 53); 18 matching reference frames, 18 voiced estimates,
18 within 50 cents, median absolute error 2.23 cents. The bend-suggestion API
returned zero candidates for all 20 reference notes; most were too short or
had overlapping annotations. There are **no verified bend labels** here, so
zero suggestions cannot establish either precision or recall.

This is a tiny, non-held-out single-note sanity check, not corpus-level F0
accuracy, polyphonic tracking, transcription accuracy, or bend detection
quality. The scorer uses ground-truth note intervals/strings, not model
predictions. GuitarSet pitch annotations have their own errors, and undetected
sounds in the mono microphone can interfere. Evaluate a much larger independent
set, including labeled bends, before using this to infer production techniques.

Attribution, original data hashes, and regeneration instructions are in
`crates/mtab-cli/tests/fixtures/guitarset/README.md`. The derived 5-second
contours are CC BY 4.0, separate from the project code and model weights.

## Opt-in MIDI bend attachment

`--experimental-bends` now attaches candidates from the same conservative
analysis to assigned notes in MIDI and IR. A synthetic 100-cent rising sine
with one omitted unplayable predecessor checks source-index mapping through
partial fingering; Mido independently validates note timing, nonzero pitch
wheels and reset. Overlapping supplied notes across instrument tracks are
skipped in a separate test. No official MuScriptor checkpoint or labeled
real-guitar bend is involved in these tests, so they do not establish precision
or recall. False positives from undetected overlapping sounds, nonstandard
tuning or vibrato remain possible; the option is off by default, MIDI-only,
and requires review. `--techniques` remains unsupported.
