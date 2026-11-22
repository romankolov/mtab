//! Experimental, loss-detecting rhythmic planning for a future GP5 writer.
//!
//! This module does **not** write Guitar Pro files. It converts a constrained
//! [`TabDocument`] to bar-contained beats with ties and rests. It intentionally
//! rejects input it cannot yet preserve, including tempo/meter changes,
//! eight-string tuning, chord symbols and techniques; it never silently drops
//! them. Timing is quantized to 1/32 notes at an integer, constant BPM.
//!
//! GP5 uses two voices per measure. This initial planner uses a single voice,
//! tying notes across beat boundaries when another string changes. Notes whose
//! quantized intervals collide on one string are rejected. A future encoder must
//! still be validated with an independent GP5 parser and a notation program.

use std::collections::BTreeMap;

use mtab_core::{MtabError, Result};
use mtab_guitar::{GuitarTrack, TabDocument};

/// Pulses per quarter note in the proposed notation plan.
pub const PPQ: u64 = 960;
/// The 1/32-note grid in pulses.
pub const GRID: u64 = PPQ / 8;
const BAR: u64 = PPQ * 4;
const MAX_BARS: u64 = 4096;
const MAX_TRACKS: usize = 60;
const MAX_NOTES: usize = 50_000;
const MAX_BEATS: usize = 300_000;

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
    /// Whether this is a continuation rather than a new attack.
    pub tied: bool,
}

#[derive(Clone)]
struct QuantizedNote {
    start: u64,
    end: u64,
    index: usize,
    string: u8,
    fret: u8,
    velocity: u8,
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
/// support techniques, explicit ties, chords, changing tempo/meter, or empty
/// documents. Errors describe the unsupported input. Input order is preserved
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
    if !doc.chord_symbols.is_empty() {
        return Err(invalid("chord symbols are not supported"));
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
        if note.bend.is_some() || !note.technique.is_empty() || note.tie_from_prev {
            return Err(invalid(
                "bends, techniques and explicit source ties are not yet supported",
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
        });
    }
    Ok(out)
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
            let sounding = active
                .values()
                .map(|&i| {
                    let n = &notes[i];
                    PlannedNote {
                        source_index: n.index,
                        string: n.string,
                        fret: n.fret,
                        velocity: n.velocity,
                        tied: n.start < tick,
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
            .push(mtab_guitar::Technique::PalmMute);
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
}
