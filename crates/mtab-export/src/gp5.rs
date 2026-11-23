//! Experimental Guitar Pro 5.00 encoding for the constrained notation planner.
//!
//! This writer supports the documented subset of [`crate::gp5_plan`]. In
//! particular, it rejects unsupported bends and techniques, source ties,
//! variable tempo, non-4/4 meters, eight-string tracks and quantization
//! collisions rather than losing them. Explicit dead-note and palm-mute marks
//! and let-ring marks are supported. GP5 velocities are rounded to its eight dynamic levels.
//! This is not a replacement for manual verification in a notation program.

use std::path::Path;

use guitarpro::{
    beat::{Beat, Voice},
    effects::{BendEffect, BendPoint},
    enums::{BeatStatus, BendType, NoteType},
    gp::Song,
    headers::MeasureHeader,
    key_signature::Duration,
    measure::Measure,
    midi::MidiChannel,
    note::Note,
    track::Track,
};
use mtab_core::{MtabError, Result};
use mtab_guitar::{BendKind, TabDocument};

use crate::gp5_plan::{self, ScorePlan};

fn invalid(reason: &str) -> MtabError {
    MtabError::DecodeFailure {
        chunk_index: 0,
        reason: format!("cannot export GP5: {reason}"),
    }
}

/// Encode GP5 5.00 bytes and return the plan used to create them.
///
/// Timing is quantized to the 1/32 grid; inspect the plan's maximum timing
/// displacement. To avoid unrepresentable or misleading output, this initial
/// encoder permits no more than 15 tracks, a control-free Windows-1252 title of
/// at most 255 encoded bytes, and a fretboard size no greater than 99. Other constraints are
/// documented by [`gp5_plan::plan`]. Identical input yields identical bytes.
pub fn gp5_bytes(doc: &TabDocument) -> Result<(Vec<u8>, ScorePlan)> {
    let plan = gp5_plan::plan(doc)?;
    if doc.tracks.len() > 15 {
        return Err(invalid("at most 15 tracks are supported"));
    }
    if doc.meta.frets == 0 || doc.meta.frets > 99 {
        return Err(invalid("fretboard size must be 1–99"));
    }
    let title = doc.meta.title.as_deref().unwrap_or("Untitled");
    let (title_bytes, _, unmappable) = encoding_rs::WINDOWS_1252.encode(title);
    if unmappable || title.chars().any(char::is_control) || title_bytes.len() > 255 {
        return Err(invalid(
            "title must contain no control characters, be representable in Windows-1252, and fit in 255 encoded bytes; rename the input or edit its title",
        ));
    }
    let mut song = Song {
        name: title.to_owned(),
        writer: "mtab".into(),
        tempo: plan.bpm as i16,
        ..Song::default()
    };
    // The old upstream library assumes these defaults are populated.
    song.lyrics.lines = (0..5).map(|i| (i, 0, String::new())).collect();
    song.page_setup.copyright = "\n".into();
    song.channels = (0..64)
        .map(|number| {
            let mut channel = MidiChannel::default();
            channel.channel = number;
            channel.effect_channel = number;
            channel
        })
        .collect();
    let channel_indices: Vec<u8> = (0..9).chain(10..16).collect();
    for (i, input) in doc.tracks.iter().enumerate() {
        song.channels[channel_indices[i] as usize]
            .set_program(gm_program(input.instrument.as_deref()));
    }
    let bar_count = plan.tracks[0].bars.len();
    song.measure_headers = (0..bar_count)
        .map(|i| MeasureHeader {
            number: (i + 1) as u16,
            ..MeasureHeader::default()
        })
        .collect();
    for (track_index, (source, planned)) in doc.tracks.iter().zip(&plan.tracks).enumerate() {
        let mut track = Track {
            number: (track_index + 1) as i32,
            name: format!("Guitar {}", track_index + 1),
            channel_index: channel_indices[track_index] as usize,
            fret_count: doc.meta.frets,
            offset: i32::from(source.capo.position()),
            ..Track::default()
        };
        track.strings = source
            .tuning
            .strings
            .iter()
            .rev()
            .enumerate()
            .map(|(i, &pitch)| ((i + 1) as i8, pitch as i8))
            .collect();
        for (bar_index, planned_bar) in planned.bars.iter().enumerate() {
            let mut voice = Voice::default();
            for beat in planned_bar {
                let mut target = Beat {
                    duration: Duration {
                        value: u16::from(beat.denominator),
                        dotted: beat.dotted,
                        ..Duration::default()
                    },
                    status: if beat.notes.is_empty() {
                        BeatStatus::Rest
                    } else {
                        BeatStatus::Normal
                    },
                    ..Beat::default()
                };
                for note in &beat.notes {
                    let mut target_note = Note::default();
                    target_note.value = i16::from(note.fret);
                    target_note.string = (source.tuning.strings.len() - note.string as usize) as i8;
                    target_note.velocity = gp5_velocity(note.velocity);
                    target_note.kind = if note.tied {
                        NoteType::Tie
                    } else if note.dead_note {
                        NoteType::Dead
                    } else {
                        NoteType::Normal
                    };
                    target_note.effect.palm_mute = note.palm_mute;
                    target_note.effect.let_ring = note.let_ring;
                    if let Some(bend) = note.bend {
                        // GP5 point values use quarter-tone units: two units
                        // per semitone. TuxGuitar renders a value of one as 1/4.
                        let height = (bend.semitones * 2) as i8;
                        let (kind, values) = match bend.kind {
                            BendKind::Bend => (BendType::Bend, [0, height, height]),
                            BendKind::Release => (BendType::PrebendRelease, [height, height, 0]),
                            BendKind::BendRelease => (BendType::BendRelease, [0, height, 0]),
                        };
                        target_note.effect.bend = Some(BendEffect {
                            kind,
                            value: i16::from(height) * 25,
                            points: [0, bend.peak_position, 12]
                                .into_iter()
                                .zip(values)
                                .map(|(position, value)| BendPoint {
                                    position,
                                    value,
                                    vibrato: false,
                                })
                                .collect(),
                            ..BendEffect::default()
                        });
                    }
                    target.notes.push(target_note);
                }
                voice.beats.push(target);
            }
            let measure = Measure {
                number: bar_index + 1,
                track_index,
                header_index: bar_index,
                voices: vec![voice, Voice::default()],
                ..Measure::default()
            };
            track.measures.push(measure);
        }
        song.tracks.push(track);
    }
    Ok((song.write((5, 0, 0), None), plan))
}

/// Write a GP5 file and return the maximum boundary shift in seconds.
/// No file is written if validation or planning fails.
pub fn write_gp5(doc: &TabDocument, path: &Path) -> Result<f64> {
    let (bytes, plan) = gp5_bytes(doc)?;
    std::fs::write(path, bytes)?;
    Ok(plan.max_boundary_shift_seconds)
}

fn gp5_velocity(velocity: u8) -> i16 {
    let value = i16::from(velocity).clamp(15, 127);
    15 + ((value - 15 + 8) / 16).min(7) * 16
}

fn gm_program(instrument: Option<&str>) -> u8 {
    match instrument {
        Some("acoustic_guitar") => 25,
        Some("clean_electric_guitar") => 27,
        Some("distorted_electric_guitar") => 30,
        Some(name) if name.starts_with("program_") => {
            name[8..].parse::<u8>().unwrap_or(27).min(127)
        }
        _ => 27,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn velocity_rounds_to_gp_levels() {
        assert_eq!(gp5_velocity(1), 15);
        assert_eq!(gp5_velocity(80), 79);
        assert_eq!(gp5_velocity(100), 95);
        assert_eq!(gp5_velocity(127), 127);
    }
}
