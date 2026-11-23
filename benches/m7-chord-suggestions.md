# Symbolic chord suggestions: current scope

`mtab_guitar::suggest_chords(notes, minimum_duration)` is a preliminary library
API, not automatic audio chord recognition. It takes note intervals supplied
by the caller and recognizes exact pitch-class sets for major, minor and
diminished triads and major-seventh, minor-seventh and dominant-seventh chords.
It merges adjacent identical labels, filters short regions, and abstains on
incomplete, extra-tone, out-of-vocabulary or ambiguous sets. Fixed pitch-name
spelling is not key-aware; inversion, functional harmony, note confidence and
percussion filtering are not modeled. A transcription error or an overlapping
instrument can change the set or label. An exact match is **not** calibrated
confidence: chords may be voiced incompletely or interpreted differently.

The library does not yet attach these suggestions to the transcription CLI or
export chord symbols, and `--chords` remains unsupported. No annotated chord
corpus or audio accuracy evaluation has been performed. Synthetic tests check
exact examples, abstention, same-time transitions and sweep consistency across
500 deterministic randomized symbolic fixtures. These tests establish algorithm
behavior, not musical effectiveness or milestone acceptance.
