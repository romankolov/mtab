//! Experimental MusicXML 3.1 score with standard and tablature staves.
//!
//! Uses the restricted notation plan. Bends are rejected for now because a
//! MusicXML bend mark alone cannot preserve the supplied timed contour.
use crate::gp5_plan::{self, BeatPlan, PlannedNote, ScorePlan, PPQ};
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
fn note_xml(
    out: &mut String,
    source: &GuitarTrack,
    beat: &BeatPlan,
    note: Option<&PlannedNote>,
    chord: bool,
    staff: u8,
    tie_start: bool,
) {
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
        let has_notation = note.tied || tie_start || staff == 2 || note.palm_mute;
        if has_notation {
            out.push_str("<notations>");
            if note.tied {
                out.push_str("<tied type=\"stop\"/>");
            }
            if tie_start {
                out.push_str("<tied type=\"start\"/>");
            }
            if staff == 2 || note.palm_mute {
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
                out.push_str("</technical>");
            }
            out.push_str("</notations>");
        }
    }
    out.push_str("</note>\n");
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
        let mut last_start = vec![0; source.notes.len()];
        for beat in track.bars.iter().flatten() {
            for note in &beat.notes {
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
                    if beat.notes.is_empty() {
                        note_xml(&mut out, source, beat, None, false, staff, false);
                    }
                    for (j, note) in beat.notes.iter().enumerate() {
                        note_xml(
                            &mut out,
                            source,
                            beat,
                            Some(note),
                            j > 0,
                            staff,
                            last_start[note.source_index] > beat.start,
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
