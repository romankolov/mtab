//! MIDI export (PROMPT.md §6.10, §6.7): SMF type 1 with tempo map, time
//! signatures, per-guitar tracks, real velocity dynamics and pitch bends.
//!
//! Conventions follow upstream `notes.py::note_event2midi` (MIT): type-1
//! multi-track, ppq 480, meta track carrying time signatures + tempo changes
//! (a tempo change lands BEFORE notes on the same tick), programs claim
//! channels 0–8 then 10–15 in order of first appearance. Product upgrades
//! over upstream: per-note velocities (upstream exports a fixed 100) and
//! pitch bends from the IR. MIDI pitch bend is channel-wide, so overlapping
//! notes on the same track share bends; per-note isolation is not yet implemented.

use std::path::Path;

use midly::{
    num::{u15, u24, u28, u4, u7},
    Format, Header, MetaMessage, MidiMessage, PitchBend, Smf, Timing, TrackEvent, TrackEventKind,
};

use mtab_core::{MtabError, Result};
use mtab_guitar::{BendKind, TabDocument, TabNote};

/// Ticks per quarter note (upstream default 480).
pub const PPQ: u16 = 480;

/// GM program for a track's instrument name (product mapping; the model's
/// representative programs 24/26/29 are MIDI-legal GM numbers but 25/27/30
/// sound closer to the recorded timbres — documented in docs/weights.md).
fn gm_program_for(instrument: Option<&str>) -> u8 {
    match instrument {
        Some("acoustic_guitar") => 25,           // Steel-string acoustic
        Some("clean_electric_guitar") => 27,     // Electric guitar (clean)
        Some("distorted_electric_guitar") => 30, // Electric guitar (distortion)
        Some(name) if name.starts_with("program_") => name
            .trim_start_matches("program_")
            .parse::<u8>()
            .unwrap_or(27)
            .min(127),
        _ => 27,
    }
}

/// Absolute-tick MIDI event for the merge step.
struct AbsEvent<'a> {
    tick: u64,
    /// Equal-tick ordering: metadata, existing bend, note-off, reset, new bend, note-on.
    prio: u8,
    order: u32,
    kind: TrackEventKind<'a>,
}

/// Tempo map → (tick, microseconds per beat) list.
fn build_tempo_tick_map(
    tempo: &mtab_guitar::TempoMap,
    ppq: u16,
) -> Vec<(u64 /*tick*/, u32 /*us per beat*/)> {
    let mut out = Vec::new();
    let mut tick_acc = 0.0f64;
    let mut last_time = 0.0f64;
    for (i, change) in tempo.changes.iter().enumerate() {
        if i > 0 {
            tick_acc += (change.time - last_time) * (tempo.changes[i - 1].bpm / 60.0) * ppq as f64;
        }
        last_time = change.time;
        out.push((
            tick_acc.round() as u64,
            (60_000_000.0 / change.bpm).round() as u32,
        ));
    }
    out
}

fn time_to_tick(tempo: &mtab_guitar::TempoMap, ppq: u16, t: f64) -> u64 {
    let mut tick_acc = 0.0f64;
    let mut last_time = 0.0f64;
    let mut last_bpm = tempo.changes.first().map(|c| c.bpm).unwrap_or(120.0);
    for change in &tempo.changes {
        if change.time >= t {
            break;
        }
        tick_acc += (change.time - last_time) * (last_bpm / 60.0) * ppq as f64;
        last_time = change.time;
        last_bpm = change.bpm;
    }
    tick_acc += (t - last_time) * (last_bpm / 60.0) * ppq as f64;
    tick_acc.round() as u64
}

/// Bar boundaries in ticks are determined by meter, not elapsed seconds;
/// therefore they remain exact even if tempo changes within a bar.
fn time_sig_ticks(doc: &TabDocument, ppq: u16) -> Vec<(u64, u8, u8)> {
    let mut sigs = doc.meta.time_signatures.clone();
    sigs.sort_by_key(|(bar, _, _)| *bar);
    let mut out = Vec::new();
    let mut tick = 0u64;
    let mut bar = 0u32;
    let mut num = 4u32;
    let mut den = 4u32;
    for &(sig_bar, s_num, s_den) in &sigs {
        tick += (sig_bar - bar) as u64 * num as u64 * 4 * ppq as u64 / den as u64;
        bar = sig_bar;
        out.push((tick, s_num as u8, s_den.trailing_zeros() as u8));
        num = s_num;
        den = s_den;
    }
    out
}

/// Pitch-bend control points (tick, wheel value) for a note.
/// Wheel: 8192 = neutral, ±8191 max; default sensitivity 2 semitones →
/// value = 8192 + cents / 200 * 8192.
fn bend_events(note: &TabNote, ppq: u16, tempo: &mtab_guitar::TempoMap) -> Vec<(u64, i16)> {
    let Some(bend) = note.bend else {
        return Vec::new();
    };
    let wheel = |cents: f32| -> i16 {
        let v = 8192.0 + cents / 200.0 * 8192.0;
        v.round().clamp(0.0, 16_383.0) as i16
    };
    let depth = wheel(bend.depth_cents);
    let mut pts = match bend.kind {
        BendKind::Bend => vec![(bend.start, 8192), (bend.peak, depth), (note.end, depth)],
        BendKind::Release => vec![(bend.start, depth), (bend.peak, depth), (note.end, 8192)],
        BendKind::BendRelease => vec![(bend.start, 8192), (bend.peak, depth), (note.end, 8192)],
    };
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pts.into_iter()
        .map(|(t, v)| (time_to_tick(tempo, ppq, t), v))
        .collect()
}

fn midi_event(channel: u4, message: MidiMessage) -> TrackEventKind<'static> {
    TrackEventKind::Midi { channel, message }
}

/// Render a [`TabDocument`] to SMF type-1 bytes. Deterministic: identical
/// input ⇒ byte-identical output (§10.7).
pub fn midi_bytes(doc: &TabDocument) -> Result<Vec<u8>> {
    let ppq = PPQ;
    let tempo = &doc.meta.tempo_map;
    let invalid = |reason: &str| MtabError::DecodeFailure {
        chunk_index: 0,
        reason: format!("invalid MIDI document: {reason}"),
    };
    if doc.tracks.len() > 15 {
        return Err(invalid(
            "at most 15 independent melodic MIDI channels are available",
        ));
    }
    if tempo.changes.is_empty()
        || tempo.changes[0].time != 0.0
        || tempo.changes.iter().any(|c| {
            !c.time.is_finite()
                || !c.bpm.is_finite()
                || c.bpm <= 0.0
                || 60_000_000.0 / c.bpm > 16_777_215.0
                || 60_000_000.0 / c.bpm < 1.0
        })
        || tempo.changes.windows(2).any(|p| p[0].time >= p[1].time)
    {
        return Err(invalid("tempo changes must start at zero, be strictly ordered, and have finite MIDI-representable BPM"));
    }
    if doc.meta.time_signatures.iter().any(|(_, num, den)| {
        *num == 0 || *num > 255 || *den == 0 || *den > 128 || !den.is_power_of_two()
    }) {
        return Err(invalid(
            "time signatures need numerator 1–255 and power-of-two denominator 1–128",
        ));
    }
    if doc.tracks.iter().flat_map(|t| &t.notes).any(|n| {
        n.midi > 127
            || !n.start.is_finite()
            || !n.end.is_finite()
            || n.start < 0.0
            || n.end < n.start
            || n.bend.is_some_and(|b| {
                !b.start.is_finite()
                    || !b.peak.is_finite()
                    || !b.depth_cents.is_finite()
                    || b.start < n.start
                    || b.peak < b.start
                    || b.peak > n.end
            })
    }) {
        return Err(invalid(
            "notes and bends need MIDI pitches 0–127 and finite, ordered times within the note",
        ));
    }

    // --- Meta track: time signatures + tempo changes ---
    let mut meta_events: Vec<AbsEvent> = Vec::new();
    let mut order = 0u32;
    for (tick, num, den) in time_sig_ticks(doc, ppq) {
        meta_events.push(AbsEvent {
            tick,
            prio: 0,
            order,
            kind: TrackEventKind::Meta(MetaMessage::TimeSignature(num, den, 24, 8)),
        });
        order += 1;
    }
    for (tick, us) in build_tempo_tick_map(tempo, ppq) {
        meta_events.push(AbsEvent {
            tick,
            prio: 0,
            order,
            kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::from(us))),
        });
        order += 1;
    }

    // --- One track per guitar ---
    let names: Vec<String> = doc
        .tracks
        .iter()
        .enumerate()
        .map(|(i, t)| {
            t.instrument
                .clone()
                .unwrap_or_else(|| format!("guitar {}", i + 1))
        })
        .collect();
    let mut track_events: Vec<Vec<AbsEvent<'_>>> = Vec::new();
    let mut channel_pool: Vec<u4> = (0..9u8).chain(10..16u8).map(u4::from).collect();
    for (ti, track) in doc.tracks.iter().enumerate() {
        let channel = channel_pool.first().copied().unwrap_or(u4::from(15u8));
        if channel_pool.len() > 1 {
            channel_pool.remove(0);
        }
        let mut events = Vec::new();
        let mut order = 0u32;
        let name = &names[ti];
        events.push(AbsEvent {
            tick: 0,
            prio: 0,
            order,
            kind: TrackEventKind::Meta(MetaMessage::TrackName(name.as_bytes())),
        });
        order += 1;
        events.push(AbsEvent {
            tick: 0,
            prio: 0,
            order,
            kind: midi_event(
                channel,
                MidiMessage::ProgramChange {
                    program: u7::from(gm_program_for(track.instrument.as_deref())),
                },
            ),
        });
        order += 1;

        for note in &track.notes {
            let on_tick = time_to_tick(tempo, ppq, note.start);
            let min_off_tick = on_tick
                .checked_add(1)
                .ok_or_else(|| invalid("note tick overflow"))?;
            let off_tick = time_to_tick(tempo, ppq, note.end).max(min_off_tick);
            let vel = note.velocity.clamp(1, 127);
            for (tick, wheel) in bend_events(note, ppq, tempo) {
                events.push(AbsEvent {
                    tick,
                    prio: if tick == on_tick { 4 } else { 1 },
                    order,
                    kind: midi_event(
                        channel,
                        MidiMessage::PitchBend {
                            bend: PitchBend::from_int(wheel - 8192),
                        },
                    ),
                });
                order += 1;
            }
            events.push(AbsEvent {
                tick: off_tick,
                prio: 2,
                order,
                kind: midi_event(
                    channel,
                    MidiMessage::NoteOff {
                        key: u7::from(note.midi),
                        vel: u7::from(0u8),
                    },
                ),
            });
            order += 1;
            if note.bend.is_some_and(|bend| bend.kind == BendKind::Bend) {
                events.push(AbsEvent {
                    tick: off_tick,
                    prio: 3,
                    order,
                    kind: midi_event(
                        channel,
                        MidiMessage::PitchBend {
                            bend: PitchBend::from_int(0),
                        },
                    ),
                });
                order += 1;
            }
            events.push(AbsEvent {
                tick: on_tick,
                prio: 5,
                order,
                kind: midi_event(
                    channel,
                    MidiMessage::NoteOn {
                        key: u7::from(note.midi),
                        vel: u7::from(vel),
                    },
                ),
            });
            order += 1;
        }
        track_events.push(events);
    }

    // --- Assemble: deltas ---
    fn to_delta<'a>(events: &mut Vec<AbsEvent<'a>>) -> Result<Vec<TrackEvent<'a>>> {
        events.sort_by_key(|e| (e.tick, e.prio, e.order));
        let mut last = 0u64;
        let mut out = Vec::with_capacity(events.len());
        for e in events.drain(..) {
            let delta = e.tick - last;
            if delta > 0x0fff_ffff {
                return Err(MtabError::DecodeFailure {
                    chunk_index: 0,
                    reason: "MIDI event delta exceeds the 28-bit SMF limit".into(),
                });
            }
            last = e.tick;
            out.push(TrackEvent {
                delta: u28::from(delta as u32),
                kind: e.kind,
            });
        }
        out.push(TrackEvent {
            delta: u28::from(0u32),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        });
        Ok(out)
    }

    let mut tracks = Vec::with_capacity(1 + track_events.len());
    tracks.push(to_delta(&mut meta_events)?);
    for mut evs in track_events {
        tracks.push(to_delta(&mut evs)?);
    }

    let smf = Smf {
        header: Header::new(Format::Parallel, Timing::Metrical(u15::from(ppq))),
        tracks,
    };
    let mut buf = Vec::new();
    smf.write(&mut buf).map_err(|e| MtabError::DecodeFailure {
        chunk_index: 0,
        reason: format!("midi encode: {e}"),
    })?;
    Ok(buf)
}

/// Write a [`TabDocument`] to a `.mid` file.
pub fn write_midi(doc: &TabDocument, path: &Path) -> Result<()> {
    let bytes = midi_bytes(doc)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mtab_guitar::{
        BendKind, Capo, GuitarTrack, PitchBend as TabPitchBend, TabNote, TempoChange, TempoMap,
        TrackMeta, Tuning,
    };

    fn doc_with_tempo_change() -> TabDocument {
        TabDocument {
            meta: TrackMeta {
                title: Some("t".into()),
                tempo_map: TempoMap {
                    changes: vec![
                        TempoChange {
                            time: 0.0,
                            bpm: 120.0,
                        },
                        TempoChange {
                            time: 2.0,
                            bpm: 140.0,
                        },
                    ],
                },
                time_signatures: vec![(0, 4, 4)],
                frets: 22,
                source_hash: None,
            },
            tracks: vec![GuitarTrack {
                tuning: Tuning::standard(),
                capo: Capo::NONE,
                instrument: Some("distorted_electric_guitar".into()),
                notes: vec![
                    TabNote {
                        start: 0.5,
                        end: 1.0,
                        midi: 64,
                        string: 2,
                        fret: 2,
                        velocity: 90,
                        bend: None,
                        technique: vec![],
                        tie_from_prev: false,
                    },
                    TabNote {
                        start: 2.25,
                        end: 3.0,
                        midi: 62,
                        string: 3,
                        fret: 0,
                        velocity: 110,
                        bend: Some(TabPitchBend {
                            start: 2.25,
                            peak: 2.6,
                            depth_cents: 200.0,
                            kind: BendKind::BendRelease,
                        }),
                        technique: vec![],
                        tie_from_prev: false,
                    },
                ],
            }],
            chord_symbols: vec![],
        }
    }

    #[test]
    fn time_to_tick_math() {
        let tm = TempoMap::constant(120.0);
        assert_eq!(time_to_tick(&tm, 480, 1.0), 960); // 120 bpm -> 960 ticks/s
        let doc = doc_with_tempo_change();
        // 0..2s @120: 1920 ticks; 0.25s @140: 0.25*140/60*480 = 280.
        assert_eq!(time_to_tick(&doc.meta.tempo_map, 480, 2.25), 2200);
        assert_eq!(build_tempo_tick_map(&doc.meta.tempo_map, 480)[1].0, 1920);
    }

    #[test]
    fn smf_structure_and_dynamics() {
        let doc = doc_with_tempo_change();
        let bytes = midi_bytes(&doc).unwrap();
        let smf = Smf::parse(&bytes).unwrap();
        assert_eq!(smf.header.format, Format::Parallel);
        assert_eq!(smf.tracks.len(), 2); // meta + 1 guitar

        // Meta track: 1 time signature + 2 tempo changes.
        let meta: Vec<MetaMessage> = smf.tracks[0]
            .iter()
            .filter_map(|e| match e.kind {
                TrackEventKind::Meta(m) => Some(m),
                _ => None,
            })
            .collect();
        assert!(meta
            .iter()
            .any(|m| matches!(m, MetaMessage::TimeSignature(4, 2, 24, 8))));
        let tempos: Vec<u32> = meta
            .iter()
            .filter_map(|m| match m {
                MetaMessage::Tempo(t) => Some(t.as_int()),
                _ => None,
            })
            .collect();
        assert_eq!(tempos, vec![500_000, 60_000_000 / 140]); // 120bpm then 140bpm

        // Guitar track: program change 30 (distortion), note events with real
        // velocities, pitch bends present.
        let mut ons = Vec::new();
        let mut offs = Vec::new();
        let mut bends = 0;
        let mut program = None;
        for e in &smf.tracks[1] {
            match e.kind {
                TrackEventKind::Midi {
                    message: MidiMessage::ProgramChange { program: p },
                    ..
                } => program = Some(p.as_int()),
                TrackEventKind::Midi {
                    message: MidiMessage::NoteOn { key, vel },
                    ..
                } => ons.push((key.as_int(), vel.as_int())),
                TrackEventKind::Midi {
                    message: MidiMessage::NoteOff { key, .. },
                    ..
                } => offs.push(key.as_int()),
                TrackEventKind::Midi {
                    message: MidiMessage::PitchBend { .. },
                    ..
                } => bends += 1,
                _ => {}
            }
        }
        assert_eq!(program, Some(30));
        assert_eq!(ons, vec![(64, 90), (62, 110)]); // real dynamics, not flat 100
        assert_eq!(offs, vec![64, 62]);
        assert_eq!(bends, 3); // start/peak/end of BendRelease
    }

    #[test]
    fn deterministic_bytes() {
        let doc = doc_with_tempo_change();
        assert_eq!(midi_bytes(&doc).unwrap(), midi_bytes(&doc).unwrap());
    }

    #[test]
    fn bend_wheel_values() {
        let doc = doc_with_tempo_change();
        let bytes = midi_bytes(&doc).unwrap();
        let smf = Smf::parse(&bytes).unwrap();
        let wheels: Vec<i16> = smf.tracks[1]
            .iter()
            .filter_map(|e| match e.kind {
                TrackEventKind::Midi {
                    message: MidiMessage::PitchBend { bend },
                    ..
                } => Some(bend.as_int()),
                _ => None,
            })
            .collect();
        assert_eq!(wheels, vec![0, 8191, 0]); // 200 cents = full tone = max wheel (+8191)
    }
    #[test]
    fn time_signatures_use_log2_and_bar_ticks_ignore_tempo_changes() {
        let mut doc = doc_with_tempo_change();
        doc.meta.tempo_map.changes[1].time = 1.0;
        doc.meta.time_signatures = vec![(0, 4, 4), (2, 3, 8)];
        assert_eq!(time_sig_ticks(&doc, 480), vec![(0, 4, 2), (3840, 3, 3)]);
        let bytes = midi_bytes(&doc).unwrap();
        let smf = Smf::parse(&bytes).unwrap();
        for track in &smf.tracks {
            assert!(matches!(
                track.last().unwrap().kind,
                TrackEventKind::Meta(MetaMessage::EndOfTrack)
            ));
        }
    }

    #[test]
    fn rejects_invalid_meter_tempo_and_channel_overflow() {
        let mut doc = doc_with_tempo_change();
        doc.meta.time_signatures = vec![(0, 4, 3)];
        assert!(midi_bytes(&doc).is_err());
        doc.meta.time_signatures.clear();
        doc.meta.tempo_map.changes[0].bpm = 0.0;
        assert!(midi_bytes(&doc).is_err());
        doc.meta.tempo_map = TempoMap::constant(120.0);
        doc.tracks = vec![doc.tracks[0].clone(); 16];
        assert!(midi_bytes(&doc).is_err());
    }
    #[test]
    fn resets_unreleased_bend_after_note_off() {
        let mut doc = doc_with_tempo_change();
        doc.tracks[0].notes[1].bend.as_mut().unwrap().kind = BendKind::Bend;
        let bytes = midi_bytes(&doc).unwrap();
        let smf = Smf::parse(&bytes).unwrap();
        let track = &smf.tracks[1];
        let off = track.iter().position(|e| matches!(e.kind, TrackEventKind::Midi { message: MidiMessage::NoteOff { key, .. }, .. } if key.as_int() == 62)).unwrap();
        assert!(
            matches!(track[off + 1].kind, TrackEventKind::Midi { message: MidiMessage::PitchBend { bend }, .. } if bend.as_int() == 0)
        );
    }

    #[test]
    fn rejects_nonfinite_bends_and_unrepresentable_deltas() {
        let mut doc = doc_with_tempo_change();
        doc.tracks[0].notes[1].bend.as_mut().unwrap().peak = f64::NAN;
        assert!(midi_bytes(&doc).is_err());
        doc.tracks[0].notes[1].bend = None;
        doc.tracks[0].notes[1].start = 1.0e10;
        doc.tracks[0].notes[1].end = 1.0e10 + 1.0;
        assert!(midi_bytes(&doc).is_err());
    }
}
