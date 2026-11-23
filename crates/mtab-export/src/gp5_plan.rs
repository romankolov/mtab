//! Experimental, loss-detecting rhythmic planning for a future GP5 writer.
//!
//! This module does **not** write Guitar Pro files. It converts a constrained
//! [`TabDocument`] to bar-contained beats with ties and rests. It intentionally
//! rejects input it cannot yet preserve, including tempo/meter changes,
//! eight-string tuning, unsupported bends and techniques; it
//! never silently drops them. Dead-note, palm-mute, let-ring and vibrato flags,
//! hammer-ons, pull-offs and adjacent-note slides are retained; vibrato depth
//! and rate are not encodable in GP5. Timing is
//! quantized to 1/32 notes at an integer, constant BPM.
//!
//! GP5 uses two voices per measure. This initial planner uses a single voice,
//! tying notes across beat boundaries when another string changes. Notes whose
//! quantized intervals collide on one string are rejected. A future encoder must
//! still be validated with an independent GP5 parser and a notation program.

use std::collections::BTreeMap;

use mtab_core::{MtabError, Result};
use mtab_guitar::{BendKind, ChordSymbol, GuitarTrack, TabDocument, TabNote, Technique};

/// Pulses per quarter note in the proposed notation plan.
pub const PPQ: u64 = 960;
/// The 1/32-note grid in pulses.
pub const GRID: u64 = PPQ / 8;
const BAR: u64 = PPQ * 4;
const MAX_BARS: u64 = 4096;
const MAX_TRACKS: usize = 60;
const MAX_NOTES: usize = 50_000;
const MAX_BEATS: usize = 300_000;
const MAX_CHORDS: usize = 20_000;

/// A rhythmic score plan; no GP5 bytes have been encoded.
#[derive(Debug, Clone, PartialEq)]
pub struct ScorePlan {
    /// Constant integer tempo used to map seconds to notation.
    pub bpm: u16,
    /// The largest absolute onset or offset shift caused by quantization.
    pub max_boundary_shift_seconds: f64,
    /// One planned track for each input track, in input order.
    pub tracks: Vec<TrackPlan>,
}

/// Beats for one input guitar. Bars share a common four-quarter duration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackPlan {
    /// One beat list per bar, including rests.
    pub bars: Vec<Vec<BeatPlan>>,
}

/// An indivisible notated beat. An empty `notes` vector represents a rest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeatPlan {
    /// Absolute position in PPQ pulses, measured from the beginning.
    pub start: u64,
    /// Notated duration, in pulses.
    pub duration: u64,
    /// Written note denominator: 1, 2, 4, 8, 16 or 32.
    pub denominator: u8,
    /// Whether the notated duration has a dot.
    pub dotted: bool,
    /// Notes sounding in this beat, sorted low-string-first.
    pub notes: Vec<PlannedNote>,
    /// Chord symbol anchored at this beat, if one was supplied.
    pub chord: Option<PlannedChord>,
}

/// A supplied chord symbol anchored to a planned beat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedChord {
    /// Original label, e.g. `Em7`.
    pub name: String,
    /// Root pitch class, `C = 0`.
    pub root: u8,
    /// Chord quality from the fixed supported vocabulary.
    pub kind: ChordKind,
}

/// The chord qualities `mtab_guitar::suggest_chords` can emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChordKind {
    Major,
    Minor,
    Diminished,
    DominantSeventh,
    MajorSeventh,
    MinorSeventh,
}

const CHORD_ROOTS: [&str; 12] = [
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
];
const CHORD_SUFFIXES: [(&str, ChordKind); 6] = [
    ("", ChordKind::Major),
    ("m", ChordKind::Minor),
    ("dim", ChordKind::Diminished),
    ("7", ChordKind::DominantSeventh),
    ("maj7", ChordKind::MajorSeventh),
    ("m7", ChordKind::MinorSeventh),
];

/// Parse one of the fixed symbolic chord labels. No other spelling is accepted,
/// so an exporter never writes a chord it cannot name exactly.
pub fn parse_chord_label(label: &str) -> Option<(u8, ChordKind)> {
    // Prefer the longest matching root without subtracting lengths: a label
    // longer than the root table must not underflow (that panics in debug and
    // silently picks the wrong root in release).
    let (root, suffix) = CHORD_ROOTS
        .iter()
        .enumerate()
        .filter_map(|(index, name)| label.strip_prefix(name).map(|rest| (index, rest)))
        .min_by_key(|(_, rest)| rest.len())?;
    let kind = CHORD_SUFFIXES
        .iter()
        .find(|(name, _)| *name == suffix)
        .map(|(_, kind)| *kind)?;
    Some((root as u8, kind))
}

/// A fretted note inside a planned beat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedNote {
    /// Source note index within the original track.
    pub source_index: usize,
    /// Zero-based string index, low string first.
    pub string: u8,
    /// Fret relative to the capo.
    pub fret: u8,
    /// Original velocity; actual GP5 encoding may quantize velocity levels.
    pub velocity: u8,
    /// A supported bend contour, if supplied.
    pub bend: Option<PlannedBend>,
    /// Whether this is a continuation rather than a new attack.
    pub tied: bool,
    /// The source marks this as a percussive dead note.
    pub dead_note: bool,
    /// The source marks this as palm-muted.
    pub palm_mute: bool,
    /// The source marks the string to ring beyond its written duration.
    pub let_ring: bool,
    /// The source marks vibrato; GP5 stores only a yes/no flag.
    pub vibrato: bool,
    /// The source marks a hammer-on onto the next note on this string.
    pub hammer: bool,
    /// The source marks a pull-off onto the next note on this string.
    pub pull_off: bool,
    /// An adjacent-note slide marker, if supplied and validated.
    pub slide: Option<PlannedSlide>,
}

/// A slide encoded in GP5 as a one-byte marker; the target fret is not stored
/// by the format, so the planner requires a directly adjacent target note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedSlide {
    /// The supplied slide kind.
    pub kind: mtab_guitar::SlideKind,
    /// Fret the slide moves to (out) or arrives at (in).
    pub to_fret: u8,
    /// True for a picked slide into this note; false for a slide out to the
    /// next note.
    pub into: bool,
}

/// A bend on a single unsplit beat. Positions are twelfths of its duration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedBend {
    /// Shape of the bend.
    pub kind: BendKind,
    /// Time of the maximum deviation, in twelfths of the note duration.
    pub peak_position: u8,
    /// Positive bend height in whole semitones.
    pub semitones: u8,
}

#[derive(Clone)]
struct QuantizedNote {
    start: u64,
    end: u64,
    index: usize,
    string: u8,
    fret: u8,
    velocity: u8,
    bend: Option<PlannedBend>,
    dead_note: bool,
    palm_mute: bool,
    let_ring: bool,
    vibrato: bool,
    hammer: bool,
    pull_off: bool,
    slide: Option<PlannedSlide>,
    techniques: Vec<Technique>,
}

fn invalid(reason: &str) -> MtabError {
    MtabError::DecodeFailure {
        chunk_index: 0,
        reason: format!("cannot plan GP5 notation: {reason}"),
    }
}

/// Plan 4/4 notation at a constant integer BPM on a straight 1/32 grid.
///
/// A note shorter than the grid is extended to one grid unit. The maximum
/// onset/offset displacement is reported. Sequential notes that quantize to
/// overlapping intervals on the same string are rejected rather than lost.
/// Supported tunings contain six or seven strings; this planner does not
/// support techniques other than dead notes, palm muting, let-ring, vibrato,
/// hammer-on/pull-off onto the next attack and slides between adjacent notes,
/// explicit source ties, chords, changing tempo/meter, or empty documents. Bends must start at
/// the attack, peak on an interior twelfth of the original note, have a positive
/// whole-semitone height of at most two semitones, and fit a single planned beat.
/// The peak is mapped proportionally onto the quantized beat. Errors describe the unsupported input. Input order is preserved
/// as a source-note index; ties are created only for splitting sounding notes.
///
/// This is an internal-style planning API, not a GP5 exporter and not a beat
/// tracker; the caller is responsible for supplying an appropriate tempo.
pub fn plan(doc: &TabDocument) -> Result<ScorePlan> {
    let tempo = &doc.meta.tempo_map.changes;
    if tempo.len() != 1
        || tempo[0].time != 0.0
        || !tempo[0].bpm.is_finite()
        || tempo[0].bpm < 1.0
        || tempo[0].bpm > 255.0
        || tempo[0].bpm.fract() != 0.0
    {
        return Err(invalid(
            "a single integer tempo from 1 to 255 BPM starting at zero is required",
        ));
    }
    if !doc.meta.time_signatures.is_empty() && doc.meta.time_signatures != [(0, 4, 4)] {
        return Err(invalid("only a constant 4/4 time signature is supported"));
    }
    for symbol in &doc.chord_symbols {
        if !symbol.start.is_finite()
            || !symbol.end.is_finite()
            || symbol.start < 0.0
            || symbol.end <= symbol.start
            || symbol.end - symbol.start > 3600.0
        {
            return Err(invalid(
                "chord symbols need finite ordered times up to one hour",
            ));
        }
        if parse_chord_label(&symbol.label).is_none() {
            return Err(invalid(
                "chord symbols must use the supported exact spellings (C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B with no suffix, m, dim, 7, maj7 or m7)",
            ));
        }
    }
    if doc.chord_symbols.len() > MAX_CHORDS {
        return Err(invalid("too many chord symbols (limit 20000)"));
    }
    if doc.tracks.is_empty() || doc.tracks.len() > MAX_TRACKS {
        return Err(invalid("provide between 1 and 60 guitar tracks"));
    }
    let count = doc
        .tracks
        .iter()
        .try_fold(0usize, |total, track| total.checked_add(track.notes.len()));
    if count.map_or(true, |n| n > MAX_NOTES) {
        return Err(invalid("too many notes (limit 50000)"));
    }
    let bpm = tempo[0].bpm;
    let mut quantized = Vec::with_capacity(doc.tracks.len());
    let mut max_end = 0;
    let mut max_shift = 0.0_f64;
    for track in &doc.tracks {
        let notes = quantize_track(track, doc.meta.frets, bpm, &mut max_shift)?;
        max_end = max_end.max(notes.iter().map(|n| n.end).max().unwrap_or(0));
        quantized.push(notes);
    }
    let bars = (max_end.saturating_add(BAR - 1) / BAR).max(1);
    if bars > MAX_BARS {
        return Err(invalid("score exceeds the 4096-bar planning limit"));
    }
    let mut beat_count = 0;
    let mut tracks = Vec::with_capacity(quantized.len());
    for notes in quantized {
        tracks.push(plan_track(&notes, bars, &mut beat_count)?);
    }
    attach_chords(&mut tracks[0], &doc.chord_symbols, bpm)?;
    Ok(ScorePlan {
        bpm: bpm as u16,
        max_boundary_shift_seconds: max_shift,
        tracks,
    })
}

fn quantize_track(
    track: &GuitarTrack,
    frets: u8,
    bpm: f64,
    max_shift: &mut f64,
) -> Result<Vec<QuantizedNote>> {
    let strings = &track.tuning.strings;
    if !(6..=7).contains(&strings.len())
        || !strings.windows(2).all(|w| w[0] < w[1])
        || strings.iter().any(|&p| p > 127)
    {
        return Err(invalid(
            "tuning must contain six or seven ascending MIDI pitches",
        ));
    }
    let seconds_per_grid = GRID as f64 / PPQ as f64 * 60.0 / bpm;
    let max_tick = MAX_BARS * BAR;
    let mut out = Vec::with_capacity(track.notes.len());
    for (index, note) in track.notes.iter().enumerate() {
        if !note.start.is_finite()
            || !note.end.is_finite()
            || note.start < 0.0
            || note.end <= note.start
        {
            return Err(invalid(
                "note times must be finite, nonnegative and strictly ordered",
            ));
        }
        if note.tie_from_prev
            || note.technique.iter().any(|technique| {
                !matches!(
                    technique,
                    Technique::DeadNote
                        | Technique::PalmMute
                        | Technique::LetRing
                        | Technique::Vibrato { .. }
                        | Technique::HammerOn
                        | Technique::PullOff
                        | Technique::Slide { .. }
                )
            })
        {
            return Err(invalid(
                "techniques other than dead notes, palm muting, let-ring, vibrato, hammer-on, pull-off, slides and explicit source ties are not yet supported",
            ));
        }
        let Some(&open) = strings.get(note.string as usize) else {
            return Err(invalid("note string is outside its tuning"));
        };
        if note.fret > frets
            || u16::from(open) + u16::from(track.capo.position()) + u16::from(note.fret)
                != u16::from(note.midi)
            || note.midi > 127
            || !(1..=127).contains(&note.velocity)
        {
            return Err(invalid("invalid fret, pitch, capo, or velocity in a note"));
        }
        let start_grid = (note.start / seconds_per_grid).round();
        let end_grid = (note.end / seconds_per_grid).round();
        if !start_grid.is_finite()
            || !end_grid.is_finite()
            || start_grid >= (max_tick / GRID) as f64
            || end_grid > (max_tick / GRID) as f64
        {
            return Err(invalid("note lies beyond the 4096-bar planning limit"));
        }
        let start = start_grid as u64 * GRID;
        let end = (end_grid as u64 * GRID).max(start + GRID);
        *max_shift = max_shift.max((start_grid * seconds_per_grid - note.start).abs());
        *max_shift = max_shift.max((end as f64 / GRID as f64 * seconds_per_grid - note.end).abs());
        out.push(QuantizedNote {
            start,
            end,
            index,
            string: note.string,
            fret: note.fret,
            velocity: note.velocity,
            bend: plan_bend(note)?,
            dead_note: note.technique.contains(&Technique::DeadNote),
            palm_mute: note.technique.contains(&Technique::PalmMute),
            let_ring: note.technique.contains(&Technique::LetRing),
            vibrato: note
                .technique
                .iter()
                .any(|t| matches!(t, Technique::Vibrato { .. })),
            hammer: note.technique.contains(&Technique::HammerOn),
            pull_off: note.technique.contains(&Technique::PullOff),
            slide: None,
            techniques: note.technique.clone(),
        });
    }
    validate_legato(&mut out, frets)?;
    Ok(out)
}

/// Validate and attach hammer-on/pull-off and slide markers.
///
/// GP5 stores legato and slide flags on the note itself without an explicit
/// target, so every such technique must describe a real, directly adjacent
/// note on the same string: hammer-on/pull-off onto the next attack, a slide
/// out to the next attack, or a picked slide in from the previous attack.
/// Dead notes cannot be legato ends and a note may not both slide out and hold
/// a picked slide-in marker in the opposite direction.
fn validate_legato(notes: &mut [QuantizedNote], frets: u8) -> Result<()> {
    let mut order: Vec<usize> = (0..notes.len()).collect();
    order.sort_by_key(|&i| (notes[i].string, notes[i].start, i));
    for position in 0..order.len() {
        let index = order[position];
        let next = order.get(position + 1).copied().filter(|&next| {
            notes[next].string == notes[index].string && notes[next].start > notes[index].start
        });
        let previous = position
            .checked_sub(1)
            .and_then(|p| order.get(p).copied())
            .filter(|&previous| {
                notes[previous].string == notes[index].string
                    && notes[previous].end <= notes[index].start
            });
        let mut slide: Option<PlannedSlide> = None;
        // GP5 stores a single legato flag; the direction (hammer-on vs
        // pull-off) is implied by the fret movement, so a mark that
        // contradicts it would be silently rewritten.
        let hammer_on = notes[index]
            .techniques
            .iter()
            .any(|technique| matches!(technique, Technique::HammerOn));
        let pull_off = notes[index]
            .techniques
            .iter()
            .any(|technique| matches!(technique, Technique::PullOff));
        if hammer_on || pull_off {
            if hammer_on && pull_off {
                return Err(invalid(
                    "one note cannot be both a hammer-on and a pull-off",
                ));
            }
            let Some(next) = next else {
                return Err(invalid(
                    "a hammer-on or pull-off needs a following attack on the same string",
                ));
            };
            if notes[next].start != notes[index].end
                || notes[next].dead_note
                || notes[next].fret == notes[index].fret
            {
                return Err(invalid(
                    "a hammer-on or pull-off must connect directly adjacent attacks at different frets",
                ));
            }
            let rises = notes[next].fret > notes[index].fret;
            if hammer_on && !rises {
                return Err(invalid(
                    "a hammer-on must move to a higher fret than the note it starts on",
                ));
            }
            if pull_off && rises {
                return Err(invalid(
                    "a pull-off must move to a lower fret than the note it starts on",
                ));
            }
        }
        for technique in notes[index].techniques.clone() {
            if let Technique::Slide { to_fret, kind } = technique {
                if to_fret > frets {
                    return Err(invalid("slide target fret exceeds the fretboard"));
                }
                let into = matches!(kind, mtab_guitar::SlideKind::Picked);
                if into {
                    // Picked slides into a note: `to_fret` is this note's own
                    // fret and the previous note must end exactly at it with
                    // a different fret to slide from.
                    let Some(previous) = previous else {
                        return Err(invalid(
                            "a picked slide-in needs a preceding note on the same string",
                        ));
                    };
                    if notes[previous].end != notes[index].start
                        || to_fret != notes[index].fret
                        || notes[previous].fret == to_fret
                    {
                        return Err(invalid(
                            "a picked slide-in needs the previous note to end at this attack and move from its fret",
                        ));
                    }
                } else {
                    let Some(next) = next else {
                        return Err(invalid(
                            "a slide-out needs a following attack on the same string",
                        ));
                    };
                    if notes[next].start != notes[index].end
                        || notes[next].fret != to_fret
                        || notes[next].dead_note
                    {
                        return Err(invalid(
                            "a slide-out target must be a directly adjacent real note with the declared fret",
                        ));
                    }
                }
                if slide.is_some() {
                    return Err(invalid("at most one slide marker per note is supported"));
                }
                slide = Some(PlannedSlide {
                    kind,
                    to_fret,
                    into,
                });
            }
        }
        notes[index].slide = slide;
    }
    Ok(())
}

fn plan_bend(note: &TabNote) -> Result<Option<PlannedBend>> {
    let Some(bend) = note.bend else {
        return Ok(None);
    };
    if !bend.start.is_finite()
        || !bend.peak.is_finite()
        || !bend.depth_cents.is_finite()
        || bend.start != note.start
        || bend.peak <= note.start
        || bend.peak >= note.end
        || (bend.depth_cents != 100.0 && bend.depth_cents != 200.0)
        || note.technique.contains(&Technique::DeadNote)
    {
        return Err(invalid("bend must start at the note attack, have an interior peak and a positive height of 100 or 200 cents, and not be a dead note"));
    }
    let position = (bend.peak - note.start) / (note.end - note.start) * 12.0;
    let rounded = position.round();
    if !(1.0..=11.0).contains(&rounded) || (position - rounded).abs() > 1e-9 {
        return Err(invalid(
            "bend peak must be exactly on an interior twelfth of the source note",
        ));
    }
    Ok(Some(PlannedBend {
        kind: bend.kind,
        peak_position: rounded as u8,
        semitones: (bend.depth_cents / 100.0) as u8,
    }))
}

/// Anchor each supplied symbol at the last beat starting at or before it.
///
/// Chord symbols are chart annotations, not sounding events, so a symbol may
/// fall between grid positions; it is attached to the containing beat of the
/// first track, which is how notation programs display it. Symbols before the
/// first beat attach to the first beat. The caller has already validated times
/// and labels.
fn attach_chords(track: &mut TrackPlan, symbols: &[ChordSymbol], bpm: f64) -> Result<()> {
    if symbols.is_empty() {
        return Ok(());
    }
    let seconds_per_grid = GRID as f64 / PPQ as f64 * 60.0 / bpm;
    let mut previous_start = None;
    for symbol in symbols {
        let grid = (symbol.start / seconds_per_grid).round();
        if !grid.is_finite() || grid < 0.0 {
            return Err(invalid("chord symbol lies outside the score"));
        }
        let tick = grid as u64 * GRID;
        if previous_start.is_some_and(|previous| tick < previous) {
            return Err(invalid(
                "chord symbols must be supplied in nondecreasing time order",
            ));
        }
        previous_start = Some(tick);
        let (root, kind) = parse_chord_label(&symbol.label)
            .ok_or_else(|| invalid("chord symbols must use the supported exact spellings"))?;
        // Locate the containing beat (or the last beat for a trailing symbol)
        // in a read-only pass, then mutate exactly that slot.
        let mut target = None;
        let mut last = None;
        for (bar_index, bar) in track.bars.iter().enumerate() {
            for (beat_index, beat) in bar.iter().enumerate() {
                last = Some((bar_index, beat_index));
                if beat.start <= tick && tick < beat.start + beat.duration {
                    target = Some((bar_index, beat_index));
                    break;
                }
            }
            if target.is_some() {
                break;
            }
        }
        let (bar_index, beat_index) = target
            .or(last)
            .ok_or_else(|| invalid("the score has no beats to anchor a chord symbol"))?;
        let beat = &mut track.bars[bar_index][beat_index];
        if beat.chord.is_some() {
            return Err(invalid("two chord symbols map to one notated beat"));
        }
        beat.chord = Some(PlannedChord {
            name: symbol.label.clone(),
            root,
            kind,
        });
    }
    Ok(())
}

fn plan_track(notes: &[QuantizedNote], bars: u64, beat_count: &mut usize) -> Result<TrackPlan> {
    // Event buckets are processed with all note-offs before all note-ons so
    // a new attack on the same string may start exactly at the old offset.
    let mut events: BTreeMap<u64, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
    for (i, note) in notes.iter().enumerate() {
        events.entry(note.start).or_default().1.push(i);
        events.entry(note.end).or_default().0.push(i);
    }
    let mut active: BTreeMap<u8, usize> = BTreeMap::new();
    let mut result = Vec::with_capacity(bars as usize);
    for bar in 0..bars {
        let mut beats = Vec::new();
        let mut tick = bar * BAR;
        let end = tick + BAR;
        while tick < end {
            if let Some((offs, ons)) = events.get(&tick) {
                for &index in offs {
                    active.remove(&notes[index].string);
                }
                for &index in ons {
                    if active.insert(notes[index].string, index).is_some() {
                        return Err(invalid("quantized notes overlap on the same string; choose a finer grid or edit the notes"));
                    }
                }
            }
            let next = events
                .range((std::ops::Bound::Excluded(tick), std::ops::Bound::Unbounded))
                .next()
                .map_or(end, |(&t, _)| t.min(end));
            let (duration, denominator, dotted) = choose_duration(tick, next - tick);
            if active
                .values()
                .any(|&i| notes[i].bend.is_some() && notes[i].start < tick)
            {
                return Err(invalid("a bent note must fit one notated beat without a tie or an interleaved note boundary"));
            }
            let sounding = active
                .values()
                .map(|&i| {
                    let n = &notes[i];
                    PlannedNote {
                        source_index: n.index,
                        string: n.string,
                        fret: n.fret,
                        velocity: n.velocity,
                        bend: n.bend,
                        tied: n.start < tick,
                        dead_note: n.dead_note,
                        palm_mute: n.palm_mute,
                        let_ring: n.let_ring,
                        vibrato: n.vibrato,
                        hammer: n.hammer,
                        pull_off: n.pull_off,
                        slide: n.slide,
                    }
                })
                .collect();
            *beat_count += 1;
            if *beat_count > MAX_BEATS {
                return Err(invalid("score exceeds the 300000-beat planning limit"));
            }
            beats.push(BeatPlan {
                start: tick,
                duration,
                denominator,
                dotted,
                notes: sounding,
                chord: None,
            });
            tick += duration;
        }
        result.push(beats);
    }
    Ok(TrackPlan { bars: result })
}

fn choose_duration(tick: u64, remaining: u64) -> (u64, u8, bool) {
    for denominator in [1_u8, 2, 4, 8, 16, 32] {
        let base = PPQ * 4 / u64::from(denominator);
        for dotted in [true, false] {
            let duration = if dotted { base + base / 2 } else { base };
            if duration <= remaining
                && duration % GRID == 0
                && tick % base == 0
                && (!dotted || tick % (base * 2) == 0)
            {
                return (duration, denominator, dotted);
            }
        }
    }
    unreachable!("grid-aligned positions always allow a 1/32 duration")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mtab_guitar::{Capo, GuitarTrack, TabNote, TempoMap, TrackMeta, Tuning};

    fn note(start: f64, end: f64, string: u8, fret: u8) -> TabNote {
        let tuning = Tuning::standard();
        TabNote {
            start,
            end,
            midi: tuning.strings[string as usize] + fret,
            string,
            fret,
            velocity: 80,
            bend: None,
            technique: vec![],
            tie_from_prev: false,
        }
    }

    fn document(notes: Vec<TabNote>) -> TabDocument {
        TabDocument {
            meta: TrackMeta {
                title: None,
                tempo_map: TempoMap::constant(120.0),
                time_signatures: vec![(0, 4, 4)],
                frets: 22,
                source_hash: None,
            },
            tracks: vec![GuitarTrack {
                tuning: Tuning::standard(),
                capo: Capo::NONE,
                notes,
                instrument: None,
            }],
            chord_symbols: vec![],
        }
    }

    #[test]
    fn ties_across_attacks_and_bar_lines_without_losing_notes() {
        let doc = document(vec![
            note(0.0, 2.5, 0, 3),
            note(0.5, 1.0, 1, 2),
            note(2.0, 2.5, 2, 0),
        ]);
        let planned = plan(&doc).unwrap();
        assert_eq!(planned.bpm, 120);
        assert_eq!(planned.max_boundary_shift_seconds, 0.0);
        let track = &planned.tracks[0];
        assert_eq!(track.bars.len(), 2);
        for bar in &track.bars {
            assert_eq!(bar.iter().map(|b| b.duration).sum::<u64>(), BAR);
        }
        let beats: Vec<_> = track.bars.iter().flatten().collect();
        assert_eq!(beats[0].start, 0);
        assert_eq!(
            beats[0]
                .notes
                .iter()
                .map(|n| (n.source_index, n.tied))
                .collect::<Vec<_>>(),
            vec![(0, false)]
        );
        let chord = beats.iter().find(|b| b.start == PPQ).unwrap();
        assert_eq!(
            chord
                .notes
                .iter()
                .map(|n| (n.source_index, n.tied))
                .collect::<Vec<_>>(),
            vec![(0, true), (1, false)]
        );
        let next_bar = beats.iter().find(|b| b.start == BAR).unwrap();
        assert_eq!(
            next_bar
                .notes
                .iter()
                .map(|n| (n.source_index, n.tied))
                .collect::<Vec<_>>(),
            vec![(0, true), (2, false)]
        );
        assert!(beats.last().unwrap().notes.is_empty());
    }

    #[test]
    fn dotted_duration_and_minimum_note_length_are_reported() {
        let doc = document(vec![note(0.0, 0.75, 0, 0), note(0.76, 0.77, 1, 0)]);
        let p = plan(&doc).unwrap();
        let beats = &p.tracks[0].bars[0];
        assert_eq!(
            (beats[0].denominator, beats[0].dotted, beats[0].duration),
            (4, true, 1440)
        );
        assert!((p.max_boundary_shift_seconds - 0.0425).abs() < 1e-10);
    }

    #[test]
    fn rejects_quantization_collision_but_allows_touching_notes() {
        let collision = document(vec![note(0.0, 0.01, 0, 0), note(0.02, 0.03, 0, 2)]);
        assert!(plan(&collision)
            .unwrap_err()
            .to_string()
            .contains("overlap"));
        let touching = document(vec![note(0.0, 0.5, 0, 0), note(0.5, 1.0, 0, 2)]);
        let p = plan(&touching).unwrap();
        let beats = &p.tracks[0].bars[0];
        assert_eq!(beats[1].start, PPQ);
        assert_eq!(beats[1].notes[0].source_index, 1);
        assert!(!beats[1].notes[0].tied);
    }

    #[test]
    fn rejects_unsupported_or_malformed_inputs() {
        let mut doc = document(vec![note(0.0, 0.5, 0, 0)]);
        doc.meta.tempo_map = TempoMap::constant(120.5);
        assert!(plan(&doc).is_err());
        doc.meta.tempo_map = TempoMap::constant(120.0);
        doc.meta.time_signatures = vec![(0, 3, 4)];
        assert!(plan(&doc).is_err());
        doc.meta.time_signatures = vec![];
        doc.tracks[0].notes[0]
            .technique
            .push(mtab_guitar::Technique::Harmonic);
        assert!(plan(&doc).is_err());
        // A hammer-on with no following attack on the string is rejected, and
        // so is a slide whose target does not match the next note.
        doc.tracks[0].notes[0].technique = vec![mtab_guitar::Technique::HammerOn];
        assert!(plan(&doc).is_err());
        doc.tracks[0].notes[0].technique = vec![mtab_guitar::Technique::Slide {
            to_fret: 3,
            kind: mtab_guitar::SlideKind::Legato,
        }];
        assert!(plan(&doc).is_err());
        doc.tracks[0].notes[0].technique.clear();
        doc.tracks[0].notes[0].midi = 41;
        assert!(plan(&doc).is_err());
        doc.tracks[0].notes[0].midi = 40;
        doc.tracks[0].notes[0].start = f64::NAN;
        assert!(plan(&doc).is_err());
        doc.tracks[0].notes.clear();
        doc.tracks[0].tuning = Tuning::eight_string();
        assert!(plan(&doc).is_err());
        doc.tracks.clear();
        assert!(plan(&doc).is_err());
    }

    #[test]
    fn attaches_chord_symbols_to_the_containing_beat() {
        let symbol = |start: f64, end: f64, label: &str| mtab_guitar::ChordSymbol {
            start,
            end,
            label: label.into(),
        };
        // 120 BPM: one 1/32 grid step is 0.015625 s.
        let mut doc = document(vec![
            note(0.0, 0.5, 0, 0),
            note(0.5, 1.0, 0, 0),
            note(1.0, 1.5, 0, 0),
        ]);
        doc.chord_symbols = vec![symbol(0.0, 0.5, "C"), symbol(1.2, 2.0, "Ebmaj7")];
        let p = plan(&doc).unwrap();
        let beats = &p.tracks[0].bars[0];
        // Symbol 2 starts mid-beat and still anchors to that beat.
        assert_eq!(beats[0].chord.as_ref().unwrap().name, "C");
        assert_eq!(beats[0].chord.as_ref().unwrap().root, 0);
        assert_eq!(beats[1].chord, None);
        assert_eq!(beats[2].chord.as_ref().unwrap().name, "Ebmaj7");
        assert_eq!(beats[2].chord.as_ref().unwrap().root, 3);
        assert_eq!(
            beats[2].chord.as_ref().unwrap().kind,
            ChordKind::MajorSeventh
        );
        // A symbol starting before the first beat anchors to it.
        doc.chord_symbols = vec![symbol(0.0, 1.0, "Am"), symbol(0.001, 1.0, "D7")];
        assert!(plan(&doc).is_err());
        // A symbol past the last beat anchors to the last beat.
        let mut trailing = document(vec![note(0.0, 0.5, 0, 0)]);
        trailing.chord_symbols = vec![symbol(5.0, 6.0, "G")];
        let beats = &plan(&trailing).unwrap().tracks[0].bars[0];
        assert_eq!(beats.last().unwrap().chord.as_ref().unwrap().name, "G");
    }

    #[test]
    fn legato_direction_must_match_the_fret_movement() {
        // GP5 writes only one legato flag, so the direction a mark implies has
        // to agree with the frets on either side of it.
        let legato = |first: Vec<Technique>, second: Vec<Technique>, frets: (u8, u8, u8)| {
            let mut notes = vec![
                note(0.0, 0.5, 0, frets.0),
                note(0.5, 1.0, 0, frets.1),
                note(1.0, 1.5, 0, frets.2),
            ];
            notes[0].technique = first;
            notes[1].technique = second;
            plan(&document(notes))
        };
        // Rises then falls: the middle note is a pull-off, the first a hammer-on.
        legato(
            vec![Technique::HammerOn],
            vec![Technique::PullOff],
            (3, 5, 2),
        )
        .unwrap();
        // A hammer-on that goes down is a contradiction, and so is a pull-off
        // that goes up.
        assert!(legato(vec![Technique::HammerOn], vec![], (5, 3, 2)).is_err());
        assert!(legato(vec![Technique::PullOff], vec![], (3, 5, 7)).is_err());
        // One note cannot claim both directions.
        assert!(legato(
            vec![Technique::HammerOn, Technique::PullOff],
            vec![],
            (3, 5, 7)
        )
        .is_err());
    }

    #[test]
    fn overlong_chord_labels_are_rejected_without_underflow() {
        // A label whose remainder is longer than the root table would
        // underflow "roots - rest" and, in release, wrap into the wrong root.
        for label in [
            "Em7G".repeat(40),
            format!("C{}", "m".repeat(64)),
            format!("B{}", "7".repeat(200)),
        ] {
            assert_eq!(parse_chord_label(&label), None, "{label:?}");
            let mut doc = document(vec![note(0.0, 1.0, 0, 0)]);
            doc.chord_symbols = vec![mtab_guitar::ChordSymbol {
                start: 0.0,
                end: 1.0,
                label,
            }];
            assert!(plan(&doc).is_err());
        }
        // The longest root still wins over a shorter prefix of it.
        assert_eq!(
            parse_chord_label("C#m7").unwrap(),
            (1, ChordKind::MinorSeventh)
        );
        assert_eq!(
            parse_chord_label("Cm7").unwrap(),
            (0, ChordKind::MinorSeventh)
        );
        assert_eq!(
            parse_chord_label("Ebmaj7").unwrap(),
            (3, ChordKind::MajorSeventh)
        );
    }

    #[test]
    fn rejects_invalid_chord_symbols() {
        let symbol = |start: f64, end: f64, label: &str| mtab_guitar::ChordSymbol {
            start,
            end,
            label: label.into(),
        };
        let doc_with = |symbols| {
            let mut doc = document(vec![note(0.0, 0.5, 0, 0), note(0.5, 1.0, 0, 0)]);
            doc.chord_symbols = symbols;
            doc
        };
        // Only the fixed root spellings and suffixes are accepted.
        for label in ["Hm7", "Cmaj9", "c", "Cm ", "", "CM7", "C mi", "Dbm7", "G#"] {
            assert!(
                plan(&doc_with(vec![symbol(0.0, 0.5, label)])).is_err(),
                "{label:?} must be rejected"
            );
        }
        // Every supported spelling, including accented roots, is accepted.
        // Enharmonic spellings outside the fixed vocabulary (Db, G#, ...) are
        // not, because the vocabulary mirrors mtab_guitar's chord roots.
        for label in ["C#", "Ebmaj7", "Bbm7", "Cdim", "B7", "Abm"] {
            assert!(
                plan(&doc_with(vec![symbol(0.0, 0.5, label)])).is_ok(),
                "{label:?} must be accepted"
            );
        }
        // Two symbols cannot share one notated beat.
        assert!(plan(&doc_with(vec![
            symbol(0.0, 0.2, "C"),
            symbol(0.2, 0.4, "G")
        ]))
        .is_err());
        // Order must be nondecreasing once quantized to the grid.
        assert!(plan(&doc_with(vec![
            symbol(0.5, 1.0, "C"),
            symbol(0.0, 0.5, "G")
        ]))
        .is_err());
        // Times must be finite, nonnegative and within the exportable range.
        for (start, end) in [
            (f64::NAN, 0.5),
            (-1.0, 0.5),
            (0.0, f64::INFINITY),
            (0.0, -0.5),
            (0.0, 3600.1),
        ] {
            assert!(
                plan(&doc_with(vec![symbol(start, end, "C")])).is_err(),
                "({start}, {end}) must be rejected"
            );
        }
        // The symbol count is bounded.
        let many = (0..=MAX_CHORDS)
            .map(|i| symbol(i as f64 * 0.5, i as f64 * 0.5 + 0.5, "C"))
            .collect();
        assert!(plan(&doc_with(many)).is_err());
    }

    #[test]
    fn plans_silence_and_multiple_independent_tracks() {
        let mut doc = document(vec![note(0.0, 2.5, 0, 0)]);
        let mut second = doc.tracks[0].clone();
        second.notes.clear();
        doc.tracks.push(second);
        let p = plan(&doc).unwrap();
        assert_eq!(p.tracks.len(), 2);
        assert_eq!(p.tracks[1].bars.len(), 2);
        for (bar, beats) in p.tracks[1].bars.iter().enumerate() {
            assert_eq!(beats.len(), 1);
            assert_eq!(beats[0].start, bar as u64 * BAR);
            assert_eq!(beats[0].duration, BAR);
            assert!(beats[0].notes.is_empty());
        }
        assert_eq!(p, plan(&doc).unwrap());
    }

    #[test]
    fn all_grid_intervals_decompose_and_cover_exactly() {
        for start in 0..32 {
            for end in start + 1..=32 {
                let mut current = start * GRID;
                while current < end * GRID {
                    let (duration, denominator, dotted) =
                        choose_duration(current, end * GRID - current);
                    let base = PPQ * 4 / u64::from(denominator);
                    assert_eq!(duration, if dotted { base * 3 / 2 } else { base });
                    assert_eq!(current % base, 0);
                    assert_eq!(duration % GRID, 0);
                    current += duration;
                }
                assert_eq!(current, end * GRID);
            }
        }
    }
    #[test]
    fn reconstructs_many_interleaved_notes_from_tied_beats() {
        let mut notes = Vec::new();
        for string in 0..6 {
            let mut grid = u64::from(string % 3);
            for step in 0..30 {
                let length = 1 + (step * 7 + u64::from(string)) % 6;
                let start = grid as f64 * 0.0625;
                let end = (grid + length) as f64 * 0.0625;
                notes.push(note(start, end, string, (step % 5) as u8));
                grid += length + (step + u64::from(string)) % 3;
            }
        }
        // The input note order is intentionally not chronological.
        notes.reverse();
        let doc = document(notes);
        let p = plan(&doc).unwrap();
        let mut segments: Vec<Vec<(u64, u64, bool)>> = vec![vec![]; doc.tracks[0].notes.len()];
        for beat in p.tracks[0].bars.iter().flatten() {
            for n in &beat.notes {
                segments[n.source_index].push((beat.start, beat.start + beat.duration, n.tied));
            }
        }
        for (index, source) in doc.tracks[0].notes.iter().enumerate() {
            let parts = &segments[index];
            assert!(!parts.is_empty());
            assert_eq!(parts[0].0, (source.start / 0.0625).round() as u64 * GRID);
            assert_eq!(
                parts.last().unwrap().1,
                (source.end / 0.0625).round() as u64 * GRID
            );
            assert!(!parts[0].2);
            assert!(parts.iter().skip(1).all(|part| part.2));
            assert!(parts.windows(2).all(|pair| pair[0].1 == pair[1].0));
        }
    }

    #[test]
    fn validates_capo_and_bar_limit() {
        let mut doc = document(vec![note(0.0, 0.5, 0, 2)]);
        doc.tracks[0].capo = Capo::new(2).unwrap();
        assert!(plan(&doc).is_err());
        doc.tracks[0].notes[0].midi += 2;
        assert!(plan(&doc).is_ok());
        doc.tracks[0].notes[0].start = 8192.0;
        doc.tracks[0].notes[0].end = 8192.5;
        assert!(plan(&doc).unwrap_err().to_string().contains("4096-bar"));
    }
    #[test]
    fn carries_dead_and_mute_marks_across_generated_ties() {
        let mut doc = document(vec![note(0.0, 2.5, 0, 0), note(0.5, 1.0, 1, 0)]);
        doc.tracks[0].notes[0].technique = vec![Technique::DeadNote, Technique::PalmMute];
        doc.tracks[0].notes[1].technique = vec![Technique::PalmMute];
        let score = plan(&doc).unwrap();
        let notes: Vec<_> = score.tracks[0]
            .bars
            .iter()
            .flatten()
            .flat_map(|b| &b.notes)
            .collect();
        let dead: Vec<_> = notes.iter().filter(|n| n.source_index == 0).collect();
        assert_eq!(dead.len(), 4);
        assert!(!dead[0].tied && dead.iter().skip(1).all(|n| n.tied));
        assert!(dead.iter().all(|n| n.dead_note && n.palm_mute));
        let muted = notes.iter().find(|n| n.source_index == 1).unwrap();
        assert!(muted.palm_mute && !muted.dead_note && !muted.tied);
    }
    #[test]
    fn carries_let_ring_across_tied_segments() {
        let mut doc = document(vec![note(0.0, 2.5, 0, 0), note(0.5, 1.0, 1, 0)]);
        doc.tracks[0].notes[0].technique.push(Technique::LetRing);
        let plan = plan(&doc).unwrap();
        let notes: Vec<_> = plan.tracks[0]
            .bars
            .iter()
            .flatten()
            .flat_map(|beat| &beat.notes)
            .collect();
        let held: Vec<_> = notes.iter().filter(|n| n.source_index == 0).collect();
        assert_eq!(held.len(), 4);
        assert!(held.iter().all(|n| n.let_ring));
        assert!(notes
            .iter()
            .filter(|n| n.source_index == 1)
            .all(|n| !n.let_ring));
    }

    #[test]
    fn bends_accept_only_representable_unsplit_notes() {
        use mtab_guitar::{BendKind, PitchBend};
        let mut doc = document(vec![note(0.0, 0.5, 0, 0)]);
        let valid = PitchBend {
            start: 0.0,
            peak: 0.25,
            depth_cents: 100.0,
            kind: BendKind::BendRelease,
        };
        doc.tracks[0].notes[0].bend = Some(valid);
        let p = plan(&doc).unwrap();
        assert_eq!(
            p.tracks[0].bars[0][0].notes[0].bend,
            Some(PlannedBend {
                kind: BendKind::BendRelease,
                peak_position: 6,
                semitones: 1
            })
        );
        for invalid in [
            PitchBend {
                start: 0.01,
                ..valid
            },
            PitchBend {
                start: f64::NAN,
                ..valid
            },
            PitchBend {
                peak: f64::NAN,
                ..valid
            },
            PitchBend { peak: 0.0, ..valid },
            PitchBend { peak: 0.5, ..valid },
            PitchBend {
                peak: 0.251,
                ..valid
            },
            PitchBend {
                depth_cents: 150.0,
                ..valid
            },
            PitchBend {
                depth_cents: -100.0,
                ..valid
            },
            PitchBend {
                depth_cents: f32::NAN,
                ..valid
            },
        ] {
            doc.tracks[0].notes[0].bend = Some(invalid);
            assert!(plan(&doc).is_err(), "{invalid:?}");
        }
        doc.tracks[0].notes[0].bend = Some(valid);
        doc.tracks[0].notes[0].technique.push(Technique::DeadNote);
        assert!(plan(&doc).is_err());
        doc.tracks[0].notes[0].technique.clear();
        doc.tracks[0].notes.push(note(0.25, 0.5, 1, 0));
        assert!(plan(&doc)
            .unwrap_err()
            .to_string()
            .contains("bent note must fit one"));
        doc.tracks[0].notes.pop();
        doc.tracks[0].notes[0].end = 2.5;
        doc.tracks[0].notes[0].bend = Some(PitchBend {
            peak: 1.25,
            ..valid
        });
        assert!(plan(&doc)
            .unwrap_err()
            .to_string()
            .contains("bent note must fit one"));
    }
}
