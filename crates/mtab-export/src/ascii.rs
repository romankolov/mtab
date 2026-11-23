//! Deterministic, beat-aligned ASCII tablature for the restricted notation plan.
//!
//! This is a readable notation view, not a lossless performance format: timing
//! is quantized and velocities are not displayed. The plan rejects unsupported
//! meter, techniques, source ties and bend contours rather than hiding them.
use crate::gp5_plan::{self, ScorePlan};
use mtab_core::Result;
use mtab_guitar::{BendKind, TabDocument};
use std::fmt::Write as _;
use std::path::Path;

/// Render one block per bar with time values, high string first, and a legend.
/// Returns the timing plan so callers can report quantization displacement.
pub fn ascii_tab(doc: &TabDocument) -> Result<(String, ScorePlan)> {
    let plan = gp5_plan::plan(doc)?;
    let mut out = String::new();
    writeln!(out, "mtab ASCII tab | {} BPM | 4/4 | 1/32 grid", plan.bpm).unwrap();
    writeln!(out, "Legend: ~ continuation; x dead note; p palm mute; l let ring; b/r/br bend/release/bend-release; @ peak in twelfths").unwrap();
    writeln!(
        out,
        "Durations: denominator, optional dot; velocities omitted; fret is relative to capo."
    )
    .unwrap();
    for (track_number, (source, track)) in doc.tracks.iter().zip(&plan.tracks).enumerate() {
        writeln!(
            out,
            "\nTrack {} | capo {} | tuning MIDI {:?}",
            track_number + 1,
            source.capo.position(),
            source.tuning.strings
        )
        .unwrap();
        for (bar_index, bar) in track.bars.iter().enumerate() {
            writeln!(out, "Bar {}", bar_index + 1).unwrap();
            let mut rows = vec![String::new(); source.tuning.strings.len()];
            let mut duration_row = String::new();
            for beat in bar {
                let duration =
                    format!("{}{}", beat.denominator, if beat.dotted { "." } else { "" });
                let mut cells = vec![String::new(); rows.len()];
                for note in &beat.notes {
                    let mut cell = String::new();
                    if note.palm_mute {
                        cell.push('p');
                    }
                    if note.let_ring {
                        cell.push('l');
                    }
                    if note.tied {
                        cell.push('~');
                    }
                    if note.dead_note {
                        cell.push('x');
                    }
                    write!(cell, "{}", note.fret).unwrap();
                    if let Some(bend) = note.bend {
                        let kind = match bend.kind {
                            BendKind::Bend => "b",
                            BendKind::Release => "r",
                            BendKind::BendRelease => "br",
                        };
                        write!(cell, "{kind}{}@{}", bend.semitones, bend.peak_position).unwrap();
                    }
                    cells[note.string as usize] = cell;
                }
                let width = cells
                    .iter()
                    .map(String::len)
                    .max()
                    .unwrap_or(0)
                    .max(duration.len())
                    + 1;
                write!(duration_row, "{duration:-<width$}").unwrap();
                for (row, cell) in rows.iter_mut().zip(cells) {
                    write!(row, "{cell:-<width$}").unwrap();
                }
            }
            writeln!(out, "dur |{duration_row}|").unwrap();
            for i in (0..rows.len()).rev() {
                writeln!(out, "{:>3} |{}|", source.tuning.strings[i], rows[i]).unwrap();
            }
        }
    }
    Ok((out, plan))
}

/// Validate and render before creating or overwriting the output file.
pub fn write_ascii(doc: &TabDocument, path: &Path) -> Result<()> {
    let (text, _) = ascii_tab(doc)?;
    std::fs::write(path, text)?;
    Ok(())
}
