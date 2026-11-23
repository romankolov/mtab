//! MIDI export (PROMPT.md §6.10, §6.7): SMF type 1 with tempo map, time
//! signatures, per-guitar tracks, real velocity dynamics and pitch bends.
//!
//! Conventions follow upstream `notes.py::note_event2midi` (MIT): type-1
//! multi-track, ppq 480, meta track carrying time signatures + tempo changes
//! (a tempo change lands BEFORE notes on the same tick), programs claim
//! channels 0–8 then 10–15, reserved separately for each guitar track.
//! MIDI pitch bend is channel-wide, so bent notes use isolated channels while
//! they overlap other notes; unbent distinct pitches can share a channel.
//! Channel allocation is deterministic and conservative: channels reserved by
//! separate tracks are not reused, even when their notes do not overlap. An
//! error is returned instead of silently corrupting bends if 15 channels are
//! insufficient. Pitch-bend range is explicitly set to ±2 semitones via RPN;
//! unsupported larger bends are rejected. Bend control points are linearly
//! interpolated into bounded-rate pitch-wheel steps; MIDI is not truly continuous.

use std::path::Path;

use midly::{
    num::{u15, u24, u28, u4, u7},
    Format, Header, MetaMessage, MidiMessage, PitchBend, Smf, Timing, TrackEvent, TrackEventKind,
};

use mtab_core::{MtabError, Result};
use mtab_guitar::{BendKind, TabDocument, TabNote, TempoTimeline};

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

/// Convert a validated timeline position to a rounded SMF tick.
fn time_to_tick(timeline: &TempoTimeline, ppq: u16, time: f64) -> Result<u64> {
    let rounded = (timeline.seconds_to_beats(time)? * f64::from(ppq)).round();
    // u64::MAX rounds up to 2^64 in f64, so reject the boundary too.
    if !rounded.is_finite() || rounded < 0.0 || rounded >= u64::MAX as f64 {
        return Err(MtabError::InvalidTempo(
            "MIDI tick position exceeds u64 range".into(),
        ));
    }
    Ok(rounded as u64)
}

/// Tempo map → (tick, microseconds per beat) list.
fn build_tempo_tick_map(
    tempo: &mtab_guitar::TempoMap,
    timeline: &TempoTimeline,
    ppq: u16,
) -> Result<Vec<(u64, u32)>> {
    tempo
        .changes
        .iter()
        .map(|change| {
            Ok((
                time_to_tick(timeline, ppq, change.time)?,
                (60_000_000.0 / change.bpm).round() as u32,
            ))
        })
        .collect()
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
fn bend_events(note: &TabNote, ppq: u16, timeline: &TempoTimeline) -> Result<Vec<(u64, i16)>> {
    let Some(bend) = note.bend else {
        return Ok(Vec::new());
    };
    let wheel = |cents: f32| -> i16 {
        let v = 8192.0 + cents / 200.0 * 8192.0;
        v.round().clamp(0.0, 16_383.0) as i16
    };
    let depth = wheel(bend.depth_cents);
    let pts = match bend.kind {
        BendKind::Bend => [(bend.start, 8192), (bend.peak, depth), (note.end, depth)],
        BendKind::Release => [(bend.start, depth), (bend.peak, depth), (note.end, 8192)],
        BendKind::BendRelease => [(bend.start, 8192), (bend.peak, depth), (note.end, 8192)],
    };
    let mut result: Vec<(u64, i16)> = Vec::new();
    let mut append = |time: f64, value: i16| -> Result<()> {
        let tick = time_to_tick(timeline, ppq, time)?;
        if let Some(last) = result.last_mut() {
            if last.0 == tick {
                last.1 = value;
                return Ok(());
            }
            if last.1 == value {
                return Ok(());
            }
        }
        result.push((tick, value));
        Ok(())
    };
    append(pts[0].0, pts[0].1)?;
    for pair in pts.windows(2) {
        let (start, from) = pair[0];
        let (end, to) = pair[1];
        if from == to {
            continue;
        }
        // Up to 50 Hz, with an explicit cap for very long notes. When capped,
        // the step interval grows rather than allocating unbounded MIDI data.
        let steps = (((end - start) / 0.020).ceil() as usize).clamp(1, 2048);
        for step in 1..=steps {
            let fraction = step as f64 / steps as f64;
            let time = start + (end - start) * fraction;
            let value = (f64::from(from) + f64::from(to - from) * fraction).round() as i16;
            append(time, value)?;
        }
    }
    Ok(result)
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
    if tempo.changes.len() > 100_000 {
        return Err(invalid("tempo map exceeds 100000 changes"));
    }
    if tempo.changes.is_empty()
        || tempo.changes[0].time != 0.0
        || tempo.changes.iter().any(|c| {
            !c.time.is_finite()
                || !c.bpm.is_finite()
                || c.bpm <= 0.0
                || (60_000_000.0 / c.bpm).round() > 16_777_215.0
                || (60_000_000.0 / c.bpm).round() < 1.0
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
            || !(1..=127).contains(&n.velocity)
            || !n.start.is_finite()
            || !n.end.is_finite()
            || n.start < 0.0
            || n.end < n.start
            || n.bend.is_some_and(|b| {
                !b.start.is_finite()
                    || !b.peak.is_finite()
                    || !b.depth_cents.is_finite()
                    || b.depth_cents.abs() > 200.0
                    || b.start < n.start
                    || b.peak < b.start
                    || b.peak > n.end
            })
    }) {
        return Err(invalid(
            "notes need MIDI pitch 0–127 and velocity 1–127; bends need ordered finite times within the note and depth within ±200 cents",
        ));
    }

    let timeline = TempoTimeline::new(tempo)?;

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
    let tempo_events = build_tempo_tick_map(tempo, &timeline, ppq)?;
    if tempo_events.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
        return Err(invalid(
            "distinct tempo changes must occupy distinct MIDI ticks",
        ));
    }
    for (tick, us) in tempo_events {
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
    let channel_pool: Vec<u8> = (0..9u8).chain(10..16u8).collect();
    let mut next_channel = 0usize;
    for (ti, track) in doc.tracks.iter().enumerate() {
        // Assign in tick order rather than input order. SMF note-offs are
        // emitted before note-ons at equal ticks, so adjacent notes may reuse
        // a channel. A bent note needs the channel to itself; two simultaneous
        // notes with the same pitch also need distinct channels so one
        // note-off cannot silence the other.
        struct AllocChannel {
            channel: u8,
            active: Vec<(u64, u8, bool)>, // off tick, key, has bend
            needs_bend_range: bool,
        }
        let mut channels: Vec<AllocChannel> = Vec::new();
        let mut assignments = vec![0u8; track.notes.len()];
        let timings: Vec<(u64, u64)> = track
            .notes
            .iter()
            .map(|note| {
                let on = time_to_tick(&timeline, ppq, note.start)?;
                let off = time_to_tick(&timeline, ppq, note.end)?.max(
                    on.checked_add(1)
                        .ok_or_else(|| invalid("note tick overflow"))?,
                );
                Ok((on, off))
            })
            .collect::<Result<_>>()?;
        let mut note_order: Vec<usize> = (0..track.notes.len()).collect();
        note_order.sort_by_key(|&i| (timings[i].0, i));
        for i in note_order {
            let note = &track.notes[i];
            let (on, off) = timings[i];
            for ch in &mut channels {
                ch.active.retain(|&(end, _, _)| end > on);
            }
            let index = channels.iter().position(|ch| {
                ch.active.is_empty()
                    || (note.bend.is_none()
                        && ch
                            .active
                            .iter()
                            .all(|&(_, key, bent)| !bent && key != note.midi))
            });
            let index = match index {
                Some(index) => index,
                None => {
                    let Some(&channel) = channel_pool.get(next_channel) else {
                        return Err(invalid("not enough independent melodic channels to isolate bends and overlapping equal pitches; at most 15 are available across tracks"));
                    };
                    next_channel += 1;
                    channels.push(AllocChannel {
                        channel,
                        active: Vec::new(),
                        needs_bend_range: false,
                    });
                    channels.len() - 1
                }
            };
            let ch = &mut channels[index];
            ch.active.push((off, note.midi, note.bend.is_some()));
            ch.needs_bend_range |= note.bend.is_some();
            assignments[i] = ch.channel;
        }
        // Preserve a program change on an empty guitar track as well.
        if channels.is_empty() {
            let Some(&channel) = channel_pool.get(next_channel) else {
                return Err(invalid("not enough melodic channels for guitar tracks"));
            };
            next_channel += 1;
            channels.push(AllocChannel {
                channel,
                active: Vec::new(),
                needs_bend_range: false,
            });
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
        for ch in &channels {
            let channel = u4::from(ch.channel);
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
            if ch.needs_bend_range {
                // MIDI RPN 0,0 = pitch bend sensitivity. Set 2 semitones and
                // 0 cents, then null the RPN to avoid accidental later edits.
                for (controller, value) in [
                    (101u8, 0u8),
                    (100, 0),
                    (6, 2),
                    (38, 0),
                    (101, 127),
                    (100, 127),
                ] {
                    events.push(AbsEvent {
                        tick: 0,
                        prio: 0,
                        order,
                        kind: midi_event(
                            channel,
                            MidiMessage::Controller {
                                controller: u7::from(controller),
                                value: u7::from(value),
                            },
                        ),
                    });
                    order += 1;
                }
            }
        }

        for ((note, &channel), &(on_tick, off_tick)) in
            track.notes.iter().zip(&assignments).zip(&timings)
        {
            let channel = u4::from(channel);
            let vel = note.velocity;
            let bends = bend_events(note, ppq, &timeline)?;
            if events.len().saturating_add(bends.len()).saturating_add(3) > 1_000_000 {
                return Err(invalid(
                    "MIDI track exceeds one million events; split the score",
                ));
            }
            for (tick, wheel) in bends {
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
        let timeline = TempoTimeline::new(&tm).unwrap();
        assert_eq!(time_to_tick(&timeline, 480, 1.0).unwrap(), 960); // 120 bpm -> 960 ticks/s
        let doc = doc_with_tempo_change();
        // 0..2s @120: 1920 ticks; 0.25s @140: 0.25*140/60*480 = 280.
        let timeline = TempoTimeline::new(&doc.meta.tempo_map).unwrap();
        assert_eq!(time_to_tick(&timeline, 480, 2.25).unwrap(), 2200);
        assert_eq!(
            build_tempo_tick_map(&doc.meta.tempo_map, &timeline, 480).unwrap()[1].0,
            1920
        );
    }

    #[test]
    fn indexed_timeline_matches_direct_integration_on_random_maps() {
        let mut seed = 0x9f75_41bcu64;
        let mut next = |limit: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 32) % limit
        };
        for _ in 0..100 {
            let mut changes = vec![TempoChange {
                time: 0.0,
                bpm: 60.0 + next(151) as f64,
            }];
            for _ in 0..25 {
                let time = changes.last().unwrap().time + (1 + next(200)) as f64 / 100.0;
                changes.push(TempoChange {
                    time,
                    bpm: 60.0 + next(151) as f64,
                });
            }
            let map = TempoMap { changes };
            let timeline = TempoTimeline::new(&map).unwrap();
            for _ in 0..50 {
                let time = next(5000) as f64 / 100.0;
                let mut direct = 0.0;
                let mut last_time = 0.0;
                let mut bpm = map.changes[0].bpm;
                for change in &map.changes {
                    if change.time >= time {
                        break;
                    }
                    direct += (change.time - last_time) * bpm / 60.0 * f64::from(PPQ);
                    last_time = change.time;
                    bpm = change.bpm;
                }
                direct += (time - last_time) * bpm / 60.0 * f64::from(PPQ);
                let actual = time_to_tick(&timeline, PPQ, time).unwrap();
                assert!(
                    actual.abs_diff(direct.round() as u64) <= 1,
                    "time={time}, actual={actual}, direct={direct}"
                );
            }
        }
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
        assert!(bends > 3); // interpolated BendRelease, not just control points
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
        assert_eq!(*wheels.first().unwrap(), 0);
        assert_eq!(*wheels.last().unwrap(), 0);
        assert!(wheels.contains(&8191)); // 200 cents = full tone = max wheel (+8191)
        assert!(wheels.iter().any(|&w| w > 0 && w < 8191));
    }
    #[test]
    fn interpolated_contours_include_endpoints_and_are_bounded() {
        let tempo = TempoTimeline::new(&TempoMap::constant(120.0)).unwrap();
        let mut note = test_note(60, 0.0, 1.0, true);
        for (kind, first, last) in [
            (BendKind::Bend, 8192, 12288),
            (BendKind::Release, 12288, 8192),
            (BendKind::BendRelease, 8192, 8192),
        ] {
            note.bend.as_mut().unwrap().kind = kind;
            let events = bend_events(&note, PPQ, &tempo).unwrap();
            assert_eq!(events[0], (0, first));
            assert_eq!(events.last().unwrap().1, last);
            assert!(events
                .iter()
                .any(|&(_, value)| value > 8192 && value < 12288));
            assert!(events.windows(2).all(|p| p[0].0 < p[1].0));
            if kind == BendKind::BendRelease {
                let peak = events.iter().position(|&(_, v)| v == 12288).unwrap();
                assert!(events[..=peak].windows(2).all(|p| p[0].1 <= p[1].1));
                assert!(events[peak..].windows(2).all(|p| p[0].1 >= p[1].1));
            }
        }
        note.end = 1000.0;
        note.bend.as_mut().unwrap().peak = 500.0;
        note.bend.as_mut().unwrap().kind = BendKind::BendRelease;
        assert!(bend_events(&note, PPQ, &tempo).unwrap().len() <= 4097);
    }
    #[test]
    fn collapsed_tick_points_keep_last_value_before_next_note() {
        let mut doc = doc_with_tempo_change();
        doc.meta.tempo_map = TempoMap::constant(120.0);
        let mut bent = test_note(60, 0.0, 0.0001, true);
        bent.bend.as_mut().unwrap().kind = BendKind::BendRelease;
        doc.tracks[0].notes = vec![bent, test_note(62, 0.0001, 0.1, false)];
        let bytes = midi_bytes(&doc).unwrap();
        let smf = Smf::parse(&bytes).unwrap();
        let mut wheel = 0;
        let mut active = false;
        for event in &smf.tracks[1] {
            if let TrackEventKind::Midi { message, .. } = event.kind {
                match message {
                    MidiMessage::PitchBend { bend } => wheel = bend.as_int(),
                    MidiMessage::NoteOn { key, .. } if key.as_int() == 62 => {
                        assert_eq!(wheel, 0);
                        active = true;
                    }
                    _ => {}
                }
            }
        }
        assert!(active);
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
    fn supplied_interval_endpoints_encode_as_midi_tempos() {
        let mut doc = doc_with_tempo_change();
        doc.tracks.clear();
        for (end, expected) in [(0.000_001, 1), (16.777_215, 16_777_215)] {
            doc.meta.tempo_map = TempoMap::from_quarter_beat_times(&[0.0, end]).unwrap();
            let bytes = midi_bytes(&doc).unwrap();
            let smf = Smf::parse(&bytes).unwrap();
            assert!(smf.tracks[0].iter().any(|e| matches!(e.kind, TrackEventKind::Meta(MetaMessage::Tempo(value)) if value.as_int() == expected)));
        }
    }

    #[test]
    fn rejects_tempo_changes_collapsed_to_one_midi_tick_and_overflow() {
        let mut doc = doc_with_tempo_change();
        doc.meta.tempo_map.changes[1].time = 0.000_001;
        assert!(midi_bytes(&doc)
            .unwrap_err()
            .to_string()
            .contains("distinct MIDI ticks"));
        doc.meta.tempo_map = TempoMap::constant(120.0);
        doc.tracks[0].notes[0].start = f64::MAX / 2.0;
        doc.tracks[0].notes[0].end = f64::MAX;
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
    fn test_note(midi: u8, start: f64, end: f64, bent: bool) -> TabNote {
        TabNote {
            start,
            end,
            midi,
            string: 0,
            fret: 0,
            velocity: 92,
            bend: bent.then_some(TabPitchBend {
                start,
                peak: start + (end - start) / 2.0,
                depth_cents: 100.0,
                kind: BendKind::Bend,
            }),
            technique: vec![],
            tie_from_prev: false,
        }
    }

    #[test]
    fn overlapping_bends_are_isolated_and_rpn_configured() {
        let mut doc = doc_with_tempo_change();
        doc.meta.tempo_map = TempoMap::constant(120.0);
        doc.tracks[0].notes = vec![
            test_note(64, 0.0, 2.0, false),
            test_note(60, 0.0, 2.0, true),
            test_note(67, 0.0, 2.0, false),
            test_note(62, 0.5, 1.5, true),
        ];
        let bytes = midi_bytes(&doc).unwrap();
        let smf = Smf::parse(&bytes).unwrap();
        let mut notes = std::collections::BTreeMap::new();
        let mut bends = std::collections::BTreeSet::new();
        let mut rpn = std::collections::BTreeMap::<u8, Vec<(u8, u8)>>::new();
        for event in &smf.tracks[1] {
            if let TrackEventKind::Midi { channel, message } = event.kind {
                let channel = channel.as_int();
                match message {
                    MidiMessage::NoteOn { key, .. } => {
                        notes.insert(key.as_int(), channel);
                    }
                    MidiMessage::PitchBend { .. } => {
                        bends.insert(channel);
                    }
                    MidiMessage::Controller { controller, value } => rpn
                        .entry(channel)
                        .or_default()
                        .push((controller.as_int(), value.as_int())),
                    _ => {}
                }
            }
        }
        assert_eq!(notes[&64], notes[&67], "nonbent distinct pitches may share");
        assert_ne!(notes[&60], notes[&64]);
        assert_ne!(notes[&62], notes[&64]);
        assert_ne!(notes[&60], notes[&62]);
        assert_eq!(bends, [notes[&60], notes[&62]].into_iter().collect());
        for channel in bends {
            assert_eq!(
                rpn[&channel],
                vec![(101, 0), (100, 0), (6, 2), (38, 0), (101, 127), (100, 127)]
            );
        }
        assert!(!rpn.contains_key(&notes[&64]));
        assert_eq!(bytes, midi_bytes(&doc).unwrap());
    }

    #[test]
    fn channel_reuse_at_boundary_and_equal_pitch_isolation() {
        let mut doc = doc_with_tempo_change();
        doc.meta.tempo_map = TempoMap::constant(120.0);
        // Intentionally unsorted input exercises time-order allocation.
        doc.tracks[0].notes = vec![
            test_note(65, 2.0, 3.0, false),
            test_note(60, 0.0, 2.0, true),
            test_note(64, 0.5, 1.5, false),
            test_note(64, 0.75, 1.25, false),
        ];
        let bytes = midi_bytes(&doc).unwrap();
        let smf = Smf::parse(&bytes).unwrap();
        let mut ons = Vec::new();
        let mut at_boundary = Vec::new();
        let mut tick = 0;
        for event in &smf.tracks[1] {
            tick += event.delta.as_int();
            if let TrackEventKind::Midi { channel, message } = event.kind {
                if let MidiMessage::NoteOn { key, .. } = message {
                    ons.push((key.as_int(), channel.as_int()));
                }
                if tick == 1920 && channel.as_int() == 0 {
                    at_boundary.push(message);
                }
            }
        }
        assert_eq!(ons.iter().find(|(pitch, _)| *pitch == 60).unwrap().1, 0);
        assert_eq!(ons.iter().find(|(pitch, _)| *pitch == 65).unwrap().1, 0);
        let same: Vec<_> = ons.iter().filter(|(pitch, _)| *pitch == 64).collect();
        assert_ne!(same[0].1, same[1].1);
        let off = at_boundary
            .iter()
            .position(|m| matches!(m, MidiMessage::NoteOff { key, .. } if key.as_int() == 60))
            .unwrap();
        let reset = at_boundary
            .iter()
            .position(|m| matches!(m, MidiMessage::PitchBend { bend } if bend.as_int() == 0))
            .unwrap();
        let next = at_boundary
            .iter()
            .position(|m| matches!(m, MidiMessage::NoteOn { key, .. } if key.as_int() == 65))
            .unwrap();
        assert!(off < reset && reset < next);
    }

    #[test]
    fn channel_exhaustion_errors_instead_of_bend_leak() {
        let mut doc = doc_with_tempo_change();
        doc.meta.tempo_map = TempoMap::constant(120.0);
        doc.tracks[0].notes = (0..15).map(|i| test_note(40 + i, 0.0, 1.0, true)).collect();
        let bytes = midi_bytes(&doc).unwrap();
        let smf = Smf::parse(&bytes).unwrap();
        let channels: std::collections::BTreeSet<u8> = smf.tracks[1]
            .iter()
            .filter_map(|event| match event.kind {
                TrackEventKind::Midi {
                    channel,
                    message: MidiMessage::NoteOn { .. },
                } => Some(channel.as_int()),
                _ => None,
            })
            .collect();
        assert_eq!(channels.len(), 15);
        assert!(!channels.contains(&9), "channel 10 is reserved for drums");
        doc.tracks[0].notes.push(test_note(55, 0.0, 1.0, true));
        assert!(midi_bytes(&doc)
            .unwrap_err()
            .to_string()
            .contains("not enough independent melodic channels"));
    }

    #[test]
    fn channels_reserved_across_tracks_and_programs_repeated() {
        let mut doc = doc_with_tempo_change();
        doc.tracks[0].notes = vec![test_note(60, 0.0, 1.0, true), test_note(64, 0.0, 1.0, true)];
        let mut second = doc.tracks[0].clone();
        second.instrument = Some("acoustic_guitar".into());
        second.notes = vec![test_note(67, 0.0, 1.0, false)];
        doc.tracks.push(second);
        let bytes = midi_bytes(&doc).unwrap();
        let smf = Smf::parse(&bytes).unwrap();
        let programs = |track: usize| {
            smf.tracks[track]
                .iter()
                .filter_map(|event| match event.kind {
                    TrackEventKind::Midi {
                        channel,
                        message: MidiMessage::ProgramChange { program },
                    } => Some((channel.as_int(), program.as_int())),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(programs(1), vec![(0, 30), (1, 30)]);
        assert_eq!(programs(2), vec![(2, 25)]);
    }
    #[test]
    fn rejects_unrepresentable_velocity_and_bend_range() {
        let mut doc = doc_with_tempo_change();
        doc.tracks[0].notes[0].velocity = 0;
        assert!(midi_bytes(&doc).is_err());
        doc.tracks[0].notes[0].velocity = 128;
        assert!(midi_bytes(&doc).is_err());
        doc.tracks[0].notes[0].velocity = 90;
        doc.tracks[0].notes[1].bend.as_mut().unwrap().depth_cents = 201.0;
        assert!(midi_bytes(&doc).unwrap_err().to_string().contains("±200"));
        doc.tracks[0].notes[1].bend.as_mut().unwrap().depth_cents = -200.0;
        assert!(midi_bytes(&doc).is_ok());
    }
}
