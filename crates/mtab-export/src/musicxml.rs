//! Experimental MusicXML 3.1 score with standard and tablature staves.
//!
//! Explicit let-ring marks use a visual `tied` let-ring on the final segment.
//! Playback sustain is application-dependent and is not encoded as a sound tie.
//! Uses the restricted notation plan. Bends are rejected for now because a
//! MusicXML bend mark alone cannot preserve the supplied timed contour.
//! MuseScore 3.2.3 changed string/fret assignments and dropped capo and palm
//! mutes on a tested re-export; do not assume application round-trip fidelity.
use crate::gp5_plan::{self, BeatPlan, ChordKind, PlannedChord, PlannedNote, ScorePlan, PPQ};
use mtab_core::{MtabError, Result};
use mtab_guitar::{GuitarTrack, TabDocument};
use std::fmt::Write as _;
use std::path::Path;

fn invalid(reason: &str) -> MtabError {
    MtabError::DecodeFailure {
        chunk_index: 0,
        reason: format!("cannot export MusicXML: {reason}"),
    }
}
/// Split a supported chord label into its MusicXML root step and alteration.
///
/// Only labels accepted by `gp5_plan::parse_chord_label` reach this function, so
/// the root is always one of the fixed spellings; flats are reported as a
/// negative `root-alter` rather than a re-spelled sharp.
fn chord_root(chord: &PlannedChord) -> (&'static str, Option<i8>) {
    match chord.root {
        0 => ("C", None),
        1 => ("C", Some(1)),
        2 => ("D", None),
        3 => ("E", Some(-1)),
        4 => ("E", None),
        5 => ("F", None),
        6 => ("F", Some(1)),
        7 => ("G", None),
        8 => ("A", Some(-1)),
        9 => ("A", None),
        10 => ("B", Some(-1)),
        11 => ("B", None),
        _ => unreachable!("validated chord root"),
    }
}

fn chord_kind(chord: &PlannedChord) -> &'static str {
    match chord.kind {
        ChordKind::Major => "major",
        ChordKind::Minor => "minor",
        ChordKind::Diminished => "diminished",
        ChordKind::DominantSeventh => "dominant",
        ChordKind::MajorSeventh => "major-seventh",
        ChordKind::MinorSeventh => "minor-seventh",
    }
}

/// Render a `<harmony>` chord symbol; the caller places it before the beat.
fn harmony_xml(out: &mut String, chord: &PlannedChord) {
    let (step, alter) = chord_root(chord);
    write!(out, "<harmony><root><root-step>{step}</root-step>",).unwrap();
    if let Some(alter) = alter {
        write!(out, "<root-alter>{alter}</root-alter>").unwrap();
    }
    writeln!(
        out,
        "</root><kind>{}</kind><staff>1</staff></harmony>",
        chord_kind(chord)
    )
    .unwrap();
}

fn escape_xml(text: &str) -> Result<String> {
    if text.chars().any(|c| {
        !matches!(c, '\t' | '\n' | '\r') && (c < ' ' || c == '\u{fffe}' || c == '\u{ffff}')
    }) {
        return Err(invalid("text contains characters forbidden by XML 1.0"));
    }
    Ok(text
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;"))
}
fn pitch(midi: u8) -> (&'static str, bool, i16) {
    let (step, sharp) = [
        ("C", false),
        ("C", true),
        ("D", false),
        ("D", true),
        ("E", false),
        ("F", false),
        ("F", true),
        ("G", false),
        ("G", true),
        ("A", false),
        ("A", true),
        ("B", false),
    ][usize::from(midi % 12)];
    (step, sharp, i16::from(midi / 12) - 1)
}
fn pitch_xml(out: &mut String, midi: u8, prefix: &str) {
    let (step, sharp, octave) = pitch(midi);
    write!(out, "<{prefix}step>{step}</{prefix}step>").unwrap();
    if sharp {
        write!(out, "<{prefix}alter>1</{prefix}alter>").unwrap();
    }
    write!(out, "<{prefix}octave>{octave}</{prefix}octave>").unwrap();
}
/// Per-note rendering state: which staff, whether this beat continues a chord
/// or tie, and where it sits in a vibrato.
#[derive(Debug, Clone, Copy)]
struct NoteContext {
    staff: u8,
    chord: bool,
    tie_start: bool,
    vibrato: VibratoStage,
}

fn note_xml(
    out: &mut String,
    source: &GuitarTrack,
    beat: &BeatPlan,
    note: Option<&PlannedNote>,
    context: NoteContext,
) {
    let NoteContext {
        staff,
        chord,
        tie_start,
        vibrato: vibrato_stage,
    } = context;
    out.push_str("<note>");
    if chord {
        out.push_str("<chord/>");
    }
    if let Some(note) = note {
        out.push_str("<pitch>");
        pitch_xml(out, source.notes[note.source_index].midi, "");
        out.push_str("</pitch>");
    } else {
        out.push_str("<rest/>");
    }
    write!(out, "<duration>{}</duration>", beat.duration).unwrap();
    if let Some(note) = note {
        if note.tied {
            out.push_str("<tie type=\"stop\"/>");
        }
        if tie_start {
            out.push_str("<tie type=\"start\"/>");
        }
    }
    let kind = match beat.denominator {
        1 => "whole",
        2 => "half",
        4 => "quarter",
        8 => "eighth",
        16 => "16th",
        32 => "32nd",
        _ => unreachable!(),
    };
    write!(out, "<voice>{staff}</voice><type>{kind}</type>").unwrap();
    if beat.dotted {
        out.push_str("<dot/>");
    }
    if note.is_some_and(|n| n.dead_note) {
        out.push_str("<notehead>x</notehead>");
    }
    write!(out, "<staff>{staff}</staff>").unwrap();
    if let Some(note) = note {
        let has_notation = note.tied
            || tie_start
            || staff == 2
            || note.palm_mute
            || note.hammer
            || note.pull_off
            || note.slide.is_some()
            || vibrato_stage != VibratoStage::None
            || (note.let_ring && !tie_start);
        if has_notation {
            out.push_str("<notations>");
            if note.tied {
                out.push_str("<tied type=\"stop\"/>");
            }
            if tie_start {
                out.push_str("<tied type=\"start\"/>");
            }
            if note.let_ring && !tie_start {
                out.push_str("<tied type=\"let-ring\"/>");
            }
            // MusicXML orders `slide` before `ornaments` and `technical` in
            // `notations`, and it is not a `technical` child.
            if let Some(slide) = note.slide {
                if slide.into {
                    out.push_str("<slide type=\"stop\" number=\"1\"/>");
                } else {
                    out.push_str("<slide type=\"start\" number=\"1\"/>");
                }
            }
            if let Some(types) = vibrato_stage.types() {
                out.push_str("<ornaments>");
                for kind in types {
                    write!(out, "<wavy-line type=\"{kind}\" number=\"1\"/>").unwrap();
                }
                out.push_str("</ornaments>");
            }
            if staff == 2 || note.palm_mute || note.hammer || note.pull_off || note.slide.is_some()
            {
                out.push_str("<technical>");
                if staff == 2 {
                    write!(
                        out,
                        "<string>{}</string><fret>{}</fret>",
                        source.tuning.strings.len() - usize::from(note.string),
                        note.fret
                    )
                    .unwrap();
                }
                if note.palm_mute {
                    out.push_str("<other-technical>palm mute</other-technical>");
                }
                // MusicXML separates the two directions: a hammer-on starts on
                // the note being hammered *to*, and a pull-off starts on the
                // note that is pulled *off*.
                if note.hammer {
                    out.push_str("<hammer-on type=\"start\"/>");
                }
                if note.pull_off {
                    out.push_str("<pull-off type=\"start\"/>");
                }
                out.push_str("</technical>");
            }
            out.push_str("</notations>");
        }
    }
    out.push_str("</note>\n");
}

/// Where a beat sits inside a note's vibrato, so the wavy line gets a
/// `start`, optional `continue`s and one `stop` instead of restarting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VibratoStage {
    None,
    StartStop,
    Start,
    Continue,
    Stop,
}

impl VibratoStage {
    fn types(self) -> Option<&'static [&'static str]> {
        match self {
            Self::None => None,
            Self::StartStop => Some(&["start", "stop"]),
            Self::Start => Some(&["start"]),
            Self::Continue => Some(&["continue"]),
            Self::Stop => Some(&["stop"]),
        }
    }
}

fn vibrato_stage(note: &PlannedNote, first: u64, last: u64, start: u64) -> VibratoStage {
    if !note.vibrato {
        return VibratoStage::None;
    }
    match (start == first, start == last) {
        (true, true) => VibratoStage::StartStop,
        (true, false) => VibratoStage::Start,
        (false, true) => VibratoStage::Stop,
        (false, false) => VibratoStage::Continue,
    }
}

/// Render MusicXML and return the timing plan. Velocities are not encoded;
/// notes use concert pitch, and unsupported timed bends are rejected.
pub fn musicxml(doc: &TabDocument) -> Result<(String, ScorePlan)> {
    let plan = gp5_plan::plan(doc)?;
    if doc
        .tracks
        .iter()
        .flat_map(|t| &t.tuning.strings)
        .any(|&pitch| pitch < 12)
        || doc
            .tracks
            .iter()
            .flat_map(|t| &t.notes)
            .any(|n| n.midi < 12)
    {
        return Err(invalid("MusicXML 3.1 requires pitch octaves 0 through 9"));
    }
    if doc
        .tracks
        .iter()
        .flat_map(|t| &t.notes)
        .any(|n| n.bend.is_some())
    {
        return Err(invalid("timed bends are not yet supported"));
    }
    let title = escape_xml(doc.meta.title.as_deref().unwrap_or("Untitled"))?;
    let mut out = String::new();
    writeln!(out, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<score-partwise version=\"3.1\">\n<work><work-title>{title}</work-title></work>\n<part-list>").unwrap();
    for i in 0..doc.tracks.len() {
        writeln!(
            out,
            "<score-part id=\"P{}\"><part-name>Guitar {}</part-name></score-part>",
            i + 1,
            i + 1
        )
        .unwrap();
    }
    out.push_str("</part-list>\n");
    for (i, (source, track)) in doc.tracks.iter().zip(&plan.tracks).enumerate() {
        // First and last planned beat of each source note, used for tie and
        // vibrato continuation state.
        let mut first_start = vec![u64::MAX; source.notes.len()];
        let mut last_start = vec![0; source.notes.len()];
        for beat in track.bars.iter().flatten() {
            for note in &beat.notes {
                first_start[note.source_index] = first_start[note.source_index].min(beat.start);
                last_start[note.source_index] = beat.start;
            }
        }
        writeln!(out, "<part id=\"P{}\">", i + 1).unwrap();
        for (bar_index, bar) in track.bars.iter().enumerate() {
            writeln!(out, "<measure number=\"{}\">", bar_index + 1).unwrap();
            if bar_index == 0 {
                write!(out, "<attributes><divisions>{PPQ}</divisions><time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves><clef number=\"1\"><sign>G</sign><line>2</line></clef><clef number=\"2\"><sign>TAB</sign><line>5</line></clef><staff-details number=\"2\"><staff-lines>{}</staff-lines>", source.tuning.strings.len()).unwrap();
                for (line, &midi) in source.tuning.strings.iter().enumerate() {
                    write!(out, "<staff-tuning line=\"{}\">", line + 1).unwrap();
                    pitch_xml(&mut out, midi, "tuning-");
                    out.push_str("</staff-tuning>");
                }
                write!(out, "<capo>{}</capo></staff-details></attributes>\n<direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>{}</per-minute></metronome></direction-type><staff>1</staff><sound tempo=\"{}\"/></direction>\n", source.capo.position(), plan.bpm, plan.bpm).unwrap();
            }
            for staff in 1..=2 {
                if staff == 2 {
                    writeln!(out, "<backup><duration>{}</duration></backup>", PPQ * 4).unwrap();
                }
                for beat in bar {
                    // Chord symbols belong to the first staff only.
                    if staff == 1 {
                        if let Some(chord) = &beat.chord {
                            harmony_xml(&mut out, chord);
                        }
                    }
                    if beat.notes.is_empty() {
                        note_xml(
                            &mut out,
                            source,
                            beat,
                            None,
                            NoteContext {
                                staff,
                                chord: false,
                                tie_start: false,
                                vibrato: VibratoStage::None,
                            },
                        );
                    }
                    for (j, note) in beat.notes.iter().enumerate() {
                        note_xml(
                            &mut out,
                            source,
                            beat,
                            Some(note),
                            NoteContext {
                                staff,
                                chord: j > 0,
                                tie_start: last_start[note.source_index] > beat.start,
                                vibrato: vibrato_stage(
                                    note,
                                    first_start[note.source_index],
                                    last_start[note.source_index],
                                    beat.start,
                                ),
                            },
                        );
                    }
                }
            }
            out.push_str("</measure>\n");
        }
        out.push_str("</part>\n");
    }
    out.push_str("</score-partwise>\n");
    Ok((out, plan))
}

/// Validate and render before creating or overwriting the output file.
pub fn write_musicxml(doc: &TabDocument, path: &Path) -> Result<()> {
    let (text, _) = musicxml(doc)?;
    std::fs::write(path, text)?;
    Ok(())
}
