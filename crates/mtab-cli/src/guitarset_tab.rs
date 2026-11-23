//! Framewise comparison of one emitted TabDocument with GuitarSet JAMS.
//! Fixed-pitch note events, not measured f0 contours, are sampled at 10 ms.
use crate::guitarset::{parse_jams, Reference};
use mtab_core::{MtabError, Result};
use mtab_guitar::{TabDocument, Tuning};
use std::path::Path;

const HOP: f64 = 0.01;
const MAX_NOTES: usize = 4096;
const MAX_DURATION: f64 = 8.0 * 3600.0;
fn fail(s: impl Into<String>) -> MtabError {
    MtabError::InvalidModelSource(s.into())
}

#[derive(Clone, Copy, Debug)]
struct Event {
    frame: usize,
    reference: bool,
    string: usize,
    pitch: usize,
    delta: i32,
}
#[derive(Default, Debug, PartialEq, Eq)]
struct Counts {
    pitch_tp: u64,
    pitch_ref: u64,
    pitch_pred: u64,
    tab_tp: u64,
    tab_ref: u64,
    tab_pred: u64,
    pitch_string_matches: u64,
    wrong_string: u64,
    note_ref: u64,
    note_pred: u64,
    onset_tp: u64,
    onset_offset_tp: u64,
    tab_onset_tp: u64,
    tab_onset_offset_tp: u64,
}

// Frame centers t=(index+0.5)*10ms; half-open note [start,end).
fn boundary(time: f64, frames: usize) -> usize {
    let (mut lo, mut hi) = (0, frames);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if (mid as f64 + 0.5) * HOP < time {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}
fn add_event(
    events: &mut Vec<Event>,
    reference: bool,
    string: u8,
    pitch: u8,
    start: f64,
    end: f64,
    frames: usize,
) {
    let first = boundary(start, frames);
    let last = boundary(end, frames);
    if first < last {
        events.push(Event {
            frame: first,
            reference,
            string: usize::from(string),
            pitch: usize::from(pitch),
            delta: 1,
        });
        events.push(Event {
            frame: last,
            reference,
            string: usize::from(string),
            pitch: usize::from(pitch),
            delta: -1,
        });
    }
}
fn score(mut events: Vec<Event>, frames: usize) -> Counts {
    events.sort_by_key(|e| e.frame);
    let mut ref_state = [[0_i32; 128]; 6];
    let mut pred_state = [[0_i32; 128]; 6];
    let mut counts = Counts::default();
    let mut index = 0;
    while index < events.len() {
        let frame = events[index].frame;
        while index < events.len() && events[index].frame == frame {
            let e = events[index];
            let state = if e.reference {
                &mut ref_state
            } else {
                &mut pred_state
            };
            state[e.string][e.pitch] += e.delta;
            debug_assert!(state[e.string][e.pitch] >= 0);
            index += 1;
        }
        let next = events.get(index).map_or(frames, |e| e.frame);
        let span = (next - frame) as u64;
        for pitch in 0..128 {
            let mut r = 0u64;
            let mut p = 0u64;
            let mut equal = 0u64;
            for string in 0..6 {
                let rs = ref_state[string][pitch] > 0;
                let ps = pred_state[string][pitch] > 0;
                r += u64::from(rs);
                p += u64::from(ps);
                equal += u64::from(rs && ps);
            }
            counts.pitch_ref += span * u64::from(r > 0);
            counts.pitch_pred += span * u64::from(p > 0);
            counts.pitch_tp += span * u64::from(r > 0 && p > 0);
            counts.tab_ref += span * r;
            counts.tab_pred += span * p;
            counts.tab_tp += span * equal;
            counts.pitch_string_matches += span * r.min(p);
            counts.wrong_string += span * (r.min(p) - equal);
        }
    }
    counts
}
fn percent(n: u64, d: u64) -> String {
    if d == 0 {
        "n/a".into()
    } else {
        format!("{:.2}%", n as f64 * 100.0 / d as f64)
    }
}
// Wiggins–Kim Eq. 8: summed correct string/fret activations divided by
// summed correctly detected *unique* pitches, without clipping. Simultaneous
// unisons can make the numerator exceed the denominator.
fn tdr(counts: &Counts) -> String {
    format!(
        "{} / {} = {}",
        counts.tab_tp,
        counts.pitch_tp,
        percent(counts.tab_tp, counts.pitch_tp)
    )
}
fn row(name: &str, tp: u64, reference: u64, predicted: u64) -> String {
    format!(
        "| {name} | {tp} | {} | {} | {} | {} | {} |",
        predicted - tp,
        reference - tp,
        percent(tp, predicted),
        percent(tp, reference),
        percent(2 * tp, reference + predicted)
    )
}
fn position_matches(
    reference: &[(mtab_core::NoteEvent, u8)],
    predicted: &[(mtab_core::NoteEvent, u8)],
    offsets: bool,
) -> u64 {
    let edges: Vec<Vec<usize>> = reference
        .iter()
        .map(|(r, string)| {
            predicted
                .iter()
                .enumerate()
                .filter_map(|(i, (p, predicted_string))| {
                    (*string == *predicted_string
                        && crate::guitarset_notes::compatible(r, p, offsets))
                    .then_some(i)
                })
                .collect()
        })
        .collect();
    crate::guitarset_notes::maximum_matching(&edges, predicted.len()) as u64
}

#[derive(Debug)]
struct Scored {
    counts: Counts,
    frames: usize,
    reference_notes: usize,
    predicted_notes: usize,
}
fn score_document(refs: &[Reference], doc: &TabDocument, duration: f64) -> Result<Scored> {
    if !duration.is_finite() || duration <= 0.0 || duration > MAX_DURATION {
        return Err(fail(
            "duration must be finite and between 0 and 28800 seconds",
        ));
    }
    if refs.len() > MAX_NOTES || doc.tracks.iter().map(|t| t.notes.len()).sum::<usize>() > MAX_NOTES
    {
        return Err(fail("reference or prediction exceeds 4096 notes"));
    }
    let frames = boundary(duration, (MAX_DURATION / HOP) as usize);
    let mut events = Vec::new();
    for r in refs {
        add_event(
            &mut events,
            true,
            r.string,
            r.note.midi,
            r.note.start,
            r.note.end,
            frames,
        );
    }
    let standard = Tuning::standard();
    for track in &doc.tracks {
        if track.tuning.strings != standard.strings || track.capo.position() != 0 {
            return Err(fail(
                "GuitarSet tab scoring requires standard six-string tuning and capo 0",
            ));
        }
        for n in &track.notes {
            let string = usize::from(n.string);
            if string >= 6
                || n.fret > 22
                || !n.start.is_finite()
                || !n.end.is_finite()
                || n.start < 0.0
                || n.end <= n.start
                || n.midi > 127
                || u16::from(standard.strings.get(string).copied().unwrap_or(0)) + u16::from(n.fret)
                    != u16::from(n.midi)
            {
                return Err(fail(
                    "invalid predicted note timing, pitch, string or fret for standard tuning",
                ));
            }
            add_event(&mut events, false, n.string, n.midi, n.start, n.end, frames);
        }
    }
    let mut counts = score(events, frames);
    let mut reference_notes: Vec<_> = refs.iter().map(|r| r.note).collect();
    let mut predicted_notes: Vec<_> = doc
        .tracks
        .iter()
        .flat_map(|t| &t.notes)
        .map(|n| mtab_core::NoteEvent {
            start: n.start,
            end: n.end,
            midi: n.midi,
            velocity: n.velocity,
            instrument: 24,
        })
        .collect();
    crate::guitarset_notes::clip(&mut reference_notes, Some(duration));
    crate::guitarset_notes::clip(&mut predicted_notes, Some(duration));
    counts.note_ref = reference_notes.len() as u64;
    counts.note_pred = predicted_notes.len() as u64;
    counts.onset_tp =
        crate::guitarset_notes::matching(&reference_notes, &predicted_notes, false) as u64;
    counts.onset_offset_tp =
        crate::guitarset_notes::matching(&reference_notes, &predicted_notes, true) as u64;
    let reference_positions: Vec<_> = refs
        .iter()
        .filter(|r| r.note.start < duration)
        .map(|r| {
            let mut n = r.note;
            n.end = n.end.min(duration);
            (n, r.string)
        })
        .collect();
    let predicted_positions: Vec<_> = doc
        .tracks
        .iter()
        .flat_map(|t| &t.notes)
        .filter(|n| n.start < duration)
        .map(|n| {
            (
                mtab_core::NoteEvent {
                    start: n.start,
                    end: n.end.min(duration),
                    midi: n.midi,
                    velocity: n.velocity,
                    instrument: 24,
                },
                n.string,
            )
        })
        .collect();
    counts.tab_onset_tp = position_matches(&reference_positions, &predicted_positions, false);
    counts.tab_onset_offset_tp = position_matches(&reference_positions, &predicted_positions, true);
    Ok(Scored {
        counts,
        frames,
        reference_notes: refs.len(),
        predicted_notes: doc.tracks.iter().map(|t| t.notes.len()).sum(),
    })
}

pub fn evaluate(annotation: &Path, ir: &Path, duration: f64, report: Option<&Path>) -> Result<()> {
    if let Some(dest) = report.and_then(|p| std::fs::canonicalize(p).ok()) {
        if [annotation, ir]
            .iter()
            .any(|input| std::fs::canonicalize(input).ok().as_ref() == Some(&dest))
        {
            return Err(fail(
                "report must not overwrite annotation or prediction input",
            ));
        }
    }
    let refs = parse_jams(annotation)?;
    let doc = TabDocument::from_json(&std::fs::read_to_string(ir)?)?;
    let scored = score_document(&refs, &doc, duration)?;
    let frames = scored.frames;
    let c = scored.counts;
    let output = format!("# GuitarSet framewise IR evaluation\n\nA user-supplied TabDocument and JAMS file must describe the same recording. Both use standard tuning and no capo. The evaluator uses fixed MIDI note pitches, not continuous F0 annotations; pitch bends, technique, velocity and instrument labels are ignored. Multiple predicted tracks are combined.\n\n- Window: first {duration:.6} seconds; frame centers at 5 ms + 10 ms × k, strictly before the window end\n- Frame count: {frames}\n- Reference notes: {}\n- Predicted notes: {}\n\n| Metric | TP | FP | FN | Precision | Recall | F1 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n{}\n{}\n{}\n{}\n{}\n{}\n\nNote-event rows use same MIDI pitch and onset within 50 ms; the offset row also requires offset within max(50 ms, 20% of reference duration). Tab note rows additionally require the same string (and therefore the same fret in standard tuning). Each uses independent maximum-cardinality matching. Notes crossing the window end are clipped.\n\nTDR-style ratio (correct tab positions / correct unique pitches): {}. Simultaneous unisons can produce a ratio above 100%; it is not the complement of wrong-string rate.\n\nWrong-string among pitch-matched string instances: {} / {} = {}. Per frame, identical pitch/string duplicates are collapsed; matching same-pitch instances maximizes same-string matches first. The multi-pitch row collapses unisons to pitch sets. The tablature row uses (string, MIDI pitch), equivalent to (string, fret) in standard tuning.\n\nThe TDR ratio follows Wiggins and Kim (2019), Eq. 8, but our note-derived reference and frame grid are not the original TabCNN evaluation protocol. This is not a held-out corpus result or a tab-labeler-v3 comparison. GuitarSet 1.1.0, Xi et al., CC BY 4.0; known annotation errors may affect scores.\n", scored.reference_notes, scored.predicted_notes, row("Multi-pitch frames",c.pitch_tp,c.pitch_ref,c.pitch_pred), row("Tab position frames",c.tab_tp,c.tab_ref,c.tab_pred), row("Pitch + onset notes", c.onset_tp,c.note_ref,c.note_pred), row("Pitch + onset + offset notes",c.onset_offset_tp,c.note_ref,c.note_pred), row("Tab + onset notes",c.tab_onset_tp,c.note_ref,c.note_pred), row("Tab + onset + offset notes",c.tab_onset_offset_tp,c.note_ref,c.note_pred), tdr(&c), c.wrong_string,c.pitch_string_matches,percent(c.wrong_string,c.pitch_string_matches));
    if let Some(path) = report {
        std::fs::write(path, output)?;
    } else {
        print!("{output}");
    }
    Ok(())
}

/// Score a user-specified list of precomputed IR files. All entries must be
/// valid; no recording is silently skipped. Paths are relative to the manifest.
pub fn evaluate_manifest(manifest: &Path, report: Option<&Path>) -> Result<()> {
    use std::collections::HashSet;
    let entries: serde_json::Value = serde_json::from_reader(std::fs::File::open(manifest)?)?;
    let entries = entries
        .as_array()
        .ok_or_else(|| fail("tab manifest must be a JSON array"))?;
    if entries.is_empty() || entries.len() > 10000 {
        return Err(fail("tab manifest needs 1–10000 entries"));
    }
    let root = manifest.parent().unwrap_or_else(|| Path::new("."));
    let report_path = report.and_then(|p| std::fs::canonicalize(p).ok());
    if report_path
        .as_ref()
        .is_some_and(|p| std::fs::canonicalize(manifest).ok().as_ref() == Some(p))
    {
        return Err(fail("report must not overwrite the manifest"));
    }
    let mut seen = HashSet::new();
    let mut seen_ir = HashSet::new();
    let mut total = Counts::default();
    let mut total_frames = 0usize;
    let mut details = String::new();
    for (i, entry) in entries.iter().enumerate() {
        let field = |key| {
            entry
                .get(key)
                .and_then(serde_json::Value::as_str)
                .filter(|s| !s.is_empty())
                .map(|s| root.join(s))
                .ok_or_else(|| fail(format!("manifest entry {} needs {key} path", i + 1)))
        };
        let annotation = field("annotation")?;
        let ir = field("ir")?;
        let duration = entry
            .get("duration")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| fail(format!("manifest entry {} needs numeric duration", i + 1)))?;
        let identity = std::fs::canonicalize(&annotation)?;
        let ir_identity = std::fs::canonicalize(&ir)?;
        if !seen.insert(identity.clone()) || !seen_ir.insert(ir_identity.clone()) {
            return Err(fail("tab manifest must not repeat annotations or IR files"));
        }
        if report_path
            .as_ref()
            .is_some_and(|p| p == &identity || p == &ir_identity)
        {
            return Err(fail(
                "report must not overwrite annotation or prediction input",
            ));
        }
        let refs = parse_jams(&annotation)?;
        let doc = TabDocument::from_json(&std::fs::read_to_string(&ir)?)?;
        let s = score_document(&refs, &doc, duration)?;
        let c = s.counts;
        total_frames += s.frames;
        total.pitch_tp += c.pitch_tp;
        total.pitch_ref += c.pitch_ref;
        total.pitch_pred += c.pitch_pred;
        total.tab_tp += c.tab_tp;
        total.tab_ref += c.tab_ref;
        total.tab_pred += c.tab_pred;
        total.pitch_string_matches += c.pitch_string_matches;
        total.wrong_string += c.wrong_string;
        total.note_ref += c.note_ref;
        total.note_pred += c.note_pred;
        total.onset_tp += c.onset_tp;
        total.onset_offset_tp += c.onset_offset_tp;
        total.tab_onset_tp += c.tab_onset_tp;
        total.tab_onset_offset_tp += c.tab_onset_offset_tp;
        details.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            i + 1,
            s.frames,
            s.reference_notes,
            s.predicted_notes,
            percent(2 * c.pitch_tp, c.pitch_ref + c.pitch_pred),
            percent(2 * c.tab_tp, c.tab_ref + c.tab_pred)
        ));
    }
    let output = format!("# GuitarSet framewise IR manifest evaluation\n\n{} manifest entries were all scored; inputs are user-supplied precomputed IR, not inference run by this command. Relative paths resolve against the manifest directory. Each duration is user-provided.\n\n- Total frames: {total_frames}\n\n| Micro-aggregated metric | TP | FP | FN | Precision | Recall | F1 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n{}\n{}\n{}\n{}\n{}\n{}\n\nNote-event rows use same MIDI pitch and onset within 50 ms; the offset row also requires offset within max(50 ms, 20% of reference duration). Tab note rows additionally require the same string (and therefore the same fret in standard tuning). Each uses independent maximum-cardinality matching. Notes crossing the window end are clipped.\n\nTDR-style ratio (correct tab positions / correct unique pitches): {}. Simultaneous unisons can produce a ratio above 100%; it is not the complement of wrong-string rate.\n\nWrong-string among pitch-matched string instances: {} / {} = {}.\n\n| Manifest entry | Frames | Reference notes | Predicted notes | Multi-pitch F1 | Tab-position F1 |\n| ---: | ---: | ---: | ---: | ---: | ---: |\n{details}\n10 ms frame centers; standard tuning and no capo; half-open note intervals; fixed MIDI note pitches rather than F0 contours. Multi-pitch unisons collapse to pitch sets; tab-position duplicates collapse to string/pitch sets. Wrong-string pairs maximize same-string matches among equal-pitch instances. Multiple IR tracks are combined.\n\nTDR-style ratio follows Wiggins and Kim (2019), Eq. 8, but the note-derived reference and frame grid differ from their evaluation protocol. Not a verified held-out split or tab-labeler-v3 comparison. Manifest selection and model/decoding provenance must be recorded separately. GuitarSet 1.1.0, Xi et al., CC BY 4.0; annotations may contain errors.\n", entries.len(), row("Multi-pitch frames",total.pitch_tp,total.pitch_ref,total.pitch_pred), row("Tab position frames",total.tab_tp,total.tab_ref,total.tab_pred), row("Pitch + onset notes", total.onset_tp,total.note_ref,total.note_pred), row("Pitch + onset + offset notes",total.onset_offset_tp,total.note_ref,total.note_pred), row("Tab + onset notes",total.tab_onset_tp,total.note_ref,total.note_pred), row("Tab + onset + offset notes",total.tab_onset_offset_tp,total.note_ref,total.note_pred), tdr(&total), total.wrong_string,total.pitch_string_matches,percent(total.wrong_string,total.pitch_string_matches));
    if let Some(path) = report {
        std::fs::write(path, output)?;
    } else {
        print!("{output}");
    }
    Ok(())
}

/// Stream a hash without retaining audio or weights in memory.
fn sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut digest = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Debug)]
struct AudioEntry {
    audio: std::path::PathBuf,
    reference: Vec<Reference>,
    duration: f64,
    hashes_verified: bool,
}

fn load_audio_manifest(
    manifest: &Path,
    weights: &Path,
    report: Option<&Path>,
) -> Result<Vec<AudioEntry>> {
    use std::collections::HashSet;
    let data: serde_json::Value = serde_json::from_reader(std::fs::File::open(manifest)?)?;
    let entries = data
        .as_array()
        .ok_or_else(|| fail("audio manifest must be a JSON array"))?;
    if entries.is_empty() || entries.len() > 10000 {
        return Err(fail("audio manifest needs 1–10000 entries"));
    }
    let root = manifest.parent().unwrap_or_else(|| Path::new("."));
    let destination = report.and_then(|p| std::fs::canonicalize(p).ok());
    let protected = [
        std::fs::canonicalize(manifest)?,
        std::fs::canonicalize(weights)?,
    ];
    if destination.as_ref().is_some_and(|d| protected.contains(d)) {
        return Err(fail("report must not overwrite manifest or weights"));
    }
    let mut annotations = HashSet::new();
    let mut audios = HashSet::new();
    let mut loaded = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        let path = |field| {
            entry
                .get(field)
                .and_then(serde_json::Value::as_str)
                .filter(|s| !s.is_empty())
                .map(|s| root.join(s))
                .ok_or_else(|| fail(format!("audio manifest entry {} needs {field} path", i + 1)))
        };
        let audio = path("audio")?;
        let annotation = path("annotation")?;
        let duration = entry
            .get("duration")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| {
                fail(format!(
                    "audio manifest entry {} needs numeric duration",
                    i + 1
                ))
            })?;
        if !duration.is_finite() || duration < 1.0 / 16000.0 || duration > MAX_DURATION {
            return Err(fail(
                "manifest duration must be between one sample and 28800 seconds",
            ));
        }
        let audio_id = std::fs::canonicalize(&audio)?;
        let annotation_id = std::fs::canonicalize(&annotation)?;
        if !audios.insert(audio_id.clone()) || !annotations.insert(annotation_id.clone()) {
            return Err(fail(
                "audio manifest must not repeat audio or annotation files",
            ));
        }
        if destination
            .as_ref()
            .is_some_and(|d| *d == audio_id || *d == annotation_id)
        {
            return Err(fail("report must not overwrite audio or annotation input"));
        }
        let expected_audio = entry.get("audio_sha256");
        let expected_annotation = entry.get("annotation_sha256");
        let hashes_verified = match (expected_audio, expected_annotation) {
            (None, None) => false,
            (Some(audio_hash), Some(annotation_hash)) => {
                for (path, hash) in [(&audio, audio_hash), (&annotation, annotation_hash)] {
                    let expected = hash
                        .as_str()
                        .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
                        .ok_or_else(|| {
                            fail(format!(
                                "audio manifest entry {} has invalid SHA-256",
                                i + 1
                            ))
                        })?;
                    if !sha256(path)?.eq_ignore_ascii_case(expected) {
                        return Err(fail(format!(
                            "SHA-256 mismatch for manifest entry {}: {}",
                            i + 1,
                            path.display()
                        )));
                    }
                }
                true
            }
            _ => {
                return Err(fail(format!(
                    "audio manifest entry {} must supply both hashes or neither",
                    i + 1
                )))
            }
        };
        let reference = parse_jams(&annotation)?;
        if reference.len() > MAX_NOTES {
            return Err(fail("reference exceeds 4096 notes"));
        }
        loaded.push(AudioEntry {
            audio,
            reference,
            duration,
            hashes_verified,
        });
    }
    Ok(loaded)
}

fn empty_document() -> TabDocument {
    TabDocument {
        meta: mtab_guitar::TrackMeta {
            title: None,
            tempo_map: mtab_guitar::TempoMap::constant(120.0),
            time_signatures: vec![(0, 4, 4)],
            frets: 22,
            source_hash: None,
        },
        tracks: vec![],
        chord_symbols: vec![],
    }
}

fn transcribe_for_evaluation(
    model: &mtab_core::lm::LmModel,
    entry: &AudioEntry,
    max_tokens: usize,
) -> Result<TabDocument> {
    let mut audio = mtab_audio::load_model_input(&entry.audio)?;
    // Do not pad missing audio or claim a larger evaluation window.
    let samples = (entry.duration * f64::from(mtab_audio::MODEL_SAMPLE_RATE)).floor() as usize;
    if samples > audio.samples.len() {
        return Err(fail("audio is shorter than the manifest duration"));
    }
    audio.samples.truncate(samples);
    let opts = mtab_core::lm::GenOptions {
        max_gen_len: max_tokens,
        seed: 0,
        ..Default::default()
    };
    let generated = mtab_audio::transcribe::transcribe(
        model,
        &audio,
        &["guitar".to_owned()],
        &opts,
        true,
        |_| {},
    )?;
    let mut groups = std::collections::BTreeMap::<u16, Vec<mtab_core::NoteEvent>>::new();
    for note in generated.notes {
        groups.entry(note.instrument).or_default().push(note);
    }
    let mut doc = empty_document();
    let tuning = Tuning::standard();
    for notes in groups.values() {
        let assigned = mtab_guitar::assign_notes(
            notes,
            &tuning,
            mtab_guitar::Capo::NONE,
            mtab_guitar::FingeringOptions::default(),
        )?;
        doc.tracks.push(mtab_guitar::GuitarTrack {
            tuning: tuning.clone(),
            capo: mtab_guitar::Capo::NONE,
            notes: assigned,
            instrument: None,
        });
    }
    Ok(doc)
}

fn run_audio_entries<F>(
    entries: &[AudioEntry],
    model_label: &str,
    max_tokens: usize,
    mut infer: F,
) -> Result<(String, usize)>
where
    F: FnMut(&AudioEntry) -> Result<TabDocument>,
{
    let mut total = Counts::default();
    let hashes_verified = entries.iter().filter(|e| e.hashes_verified).count();
    let mut frames = 0usize;
    let mut failures = 0usize;
    let mut details = String::new();
    for (i, entry) in entries.iter().enumerate() {
        eprintln!(
            "GuitarSet end-to-end recording {}/{}: {}",
            i + 1,
            entries.len(),
            entry.audio.display()
        );
        let effective = (entry.duration * 16000.0).floor() / 16000.0;
        let result = infer(entry).and_then(|doc| score_document(&entry.reference, &doc, effective));
        let (scored, status) = match result {
            Ok(score) => (score, "ok".to_owned()),
            Err(error) => {
                failures += 1;
                // Failed files count as an empty transcription, never dropped.
                let reason = error.to_string().replace(['|', '\n', '\r'], " ");
                (
                    score_document(&entry.reference, &empty_document(), effective)?,
                    format!("failed: {reason}"),
                )
            }
        };
        let c = scored.counts;
        frames += scored.frames;
        total.pitch_tp += c.pitch_tp;
        total.pitch_ref += c.pitch_ref;
        total.pitch_pred += c.pitch_pred;
        total.tab_tp += c.tab_tp;
        total.tab_ref += c.tab_ref;
        total.tab_pred += c.tab_pred;
        total.pitch_string_matches += c.pitch_string_matches;
        total.wrong_string += c.wrong_string;
        total.note_ref += c.note_ref;
        total.note_pred += c.note_pred;
        total.onset_tp += c.onset_tp;
        total.onset_offset_tp += c.onset_offset_tp;
        total.tab_onset_tp += c.tab_onset_tp;
        total.tab_onset_offset_tp += c.tab_onset_offset_tp;
        details.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            i + 1,
            scored.frames,
            scored.reference_notes,
            scored.predicted_notes,
            percent(2 * c.pitch_tp, c.pitch_ref + c.pitch_pred),
            percent(2 * c.tab_tp, c.tab_ref + c.tab_pred),
            status
        ));
    }
    Ok((format!("# GuitarSet end-to-end audio evaluation\n\nEntries: {}; failed entries: {failures}; total frames: {frames}; pairs with both supplied SHA-256 hashes verified: {hashes_verified}. Model architecture: {model_label}; greedy decoding, seed 0, guitar-family mask, strict EOS, token limit {max_tokens}, standard tuning, capo 0, 22 frets, default bounded fingering (no wide-stretch retry). The model is loaded once. Each manifest duration is rounded down to 16 kHz sample boundaries; only that prefix is transcribed and scored. No source separation, technique inference, or dynamics estimation.\n\n| Micro-aggregated metric | TP | FP | FN | Precision | Recall | F1 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n{}\n{}\n{}\n{}\n{}\n{}\n\nNote-event rows use same MIDI pitch and onset within 50 ms; the offset row also requires offset within max(50 ms, 20% of reference duration). Tab note rows additionally require the same string (and therefore the same fret in standard tuning). Each uses independent maximum-cardinality matching. Notes crossing the window end are clipped.\n\nTDR-style ratio (correct tab positions / correct unique pitches): {}. Simultaneous unisons can produce a ratio above 100%; it is not the complement of wrong-string rate.\n\nWrong-string among pitch-matched string instances: {} / {} = {}.\n\n| Manifest entry | Frames | Reference notes | Predicted notes | Multi-pitch F1 | Tab-position F1 | Status |\n| ---: | ---: | ---: | ---: | ---: | ---: | --- |\n{details}\nFailures (including strict-EOS, decoding, assignment or short audio) count as empty predictions, never excluded; thus failures contribute reference false negatives but cannot estimate their hypothetical false positives. Pitch and tab metrics are 10 ms note-derived fixed-pitch frame sets, not continuous F0. The TDR-style ratio follows Wiggins and Kim (2019), Eq. 8, but the reference and frame grid differ from their protocol. No held-out split, tab-labeler-v3 baseline, or independent ground-truth pairing is established by the manifest. Keep the manifest and document data selection separately; supplied hashes verify identity against the manifest, not provenance or license. GuitarSet 1.1.0, Xi et al., CC BY 4.0; annotations can contain errors.\n", entries.len(), row("Multi-pitch frames", total.pitch_tp,total.pitch_ref,total.pitch_pred), row("Tab position frames", total.tab_tp,total.tab_ref,total.tab_pred), row("Pitch + onset notes", total.onset_tp,total.note_ref,total.note_pred), row("Pitch + onset + offset notes",total.onset_offset_tp,total.note_ref,total.note_pred), row("Tab + onset notes",total.tab_onset_tp,total.note_ref,total.note_pred), row("Tab + onset + offset notes",total.tab_onset_offset_tp,total.note_ref,total.note_pred), tdr(&total), total.wrong_string,total.pitch_string_matches,percent(total.wrong_string,total.pitch_string_matches)), failures))
}

pub fn evaluate_audio_manifest(
    manifest: &Path,
    weights: &Path,
    size: mtab_core::ModelSize,
    max_tokens: usize,
    report: Option<&Path>,
) -> Result<()> {
    if max_tokens == 0 || max_tokens > 2000 {
        return Err(fail("--max-gen-len must be in 1–2000"));
    }
    let entries = load_audio_manifest(manifest, weights, report)?;
    let manifest_hash = sha256(manifest)?;
    let weights_hash = sha256(weights)?;
    let model = mtab_core::lm::LmModel::load(mtab_core::lm::LmConfig::from_size(size), weights)?;
    let (mut result, failures) =
        run_audio_entries(&entries, &size.to_string(), max_tokens, |entry| {
            transcribe_for_evaluation(&model, entry, max_tokens)
        })?;
    result.push_str(&format!("\n## Reproducibility identifiers\n\n- Audio manifest SHA-256: `{manifest_hash}`\n- Model weights SHA-256: `{weights_hash}`\n\nHashes identify bytes, not authorization, license, dataset completeness, or split independence.\n"));
    if let Some(path) = report {
        std::fs::write(path, result)?;
    } else {
        print!("{result}");
    }
    if failures > 0 {
        return Err(fail(format!(
            "end-to-end evaluation had {failures} failed recordings; see report"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn end_to_end_failures_are_counted_as_false_negatives() {
        let reference = vec![Reference {
            note: mtab_core::NoteEvent {
                start: 0.0,
                end: 1.0,
                midi: 45,
                velocity: 100,
                instrument: 24,
            },
            string: 1,
        }];
        let entries = vec![
            AudioEntry {
                audio: "one.wav".into(),
                reference: reference.clone(),
                duration: 1.0,
                hashes_verified: false,
            },
            AudioEntry {
                audio: "two.wav".into(),
                reference,
                duration: 1.0,
                hashes_verified: false,
            },
        ];
        let mut document = empty_document();
        document.tracks.push(mtab_guitar::GuitarTrack {
            tuning: Tuning::standard(),
            capo: mtab_guitar::Capo::NONE,
            instrument: None,
            notes: vec![mtab_guitar::TabNote {
                start: 0.0,
                end: 1.0,
                midi: 45,
                string: 1,
                fret: 0,
                velocity: 100,
                bend: None,
                technique: vec![],
                tie_from_prev: false,
            }],
        });
        let mut n = 0;
        let (report, failures) = run_audio_entries(&entries, "fixture", 512, |_| {
            n += 1;
            if n == 1 {
                Ok(document.clone())
            } else {
                Err(fail("synthetic | failure"))
            }
        })
        .unwrap();
        assert_eq!(failures, 1);
        assert!(report.contains("Entries: 2; failed entries: 1"), "{report}");
        assert!(
            report.contains("| Multi-pitch frames | 100 | 0 | 100 | 100.00% | 50.00% | 66.67% |"),
            "{report}"
        );
        assert!(
            report.contains("| Tab position frames | 100 | 0 | 100 | 100.00% | 50.00% | 66.67% |"),
            "{report}"
        );
        assert!(
            report.contains("| Pitch + onset notes | 1 | 0 | 1 | 100.00% | 50.00% | 66.67% |"),
            "{report}"
        );
        assert!(
            report.contains("failed: ") && report.contains("synthetic   failure"),
            "{report}"
        );
    }
    #[test]
    fn checked_in_audio_manifest_hashes_verify_before_loading_model() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let manifest = root.join("tests/fixtures/guitarset/guitarset_audio_manifest.json");
        let weights = root.join("../mtab-core/tests/fixtures/model/tiny.fixture");
        let entries = load_audio_manifest(&manifest, &weights, None).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].hashes_verified);
        assert_eq!(
            sha256(&entries[0].audio).unwrap(),
            "2308bb2dff9e2e332a9e9eb53e6a2e9915519a0388ef7e42a9b1102259f3f60a"
        );
    }
    #[test]
    fn actual_tiny_model_executes_audio_path_and_short_audio_fails() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let audio = root.join("tests/fixtures/guitarset/guitarset_solo_5s.wav");
        let annotation = root.join("tests/fixtures/guitarset/guitarset_solo_5s.jams");
        let weights = root.join("../mtab-core/tests/fixtures/model/tiny.fixture");
        let model = mtab_core::lm::LmModel::load(mtab_core::lm::LmConfig::fixture_tiny(), &weights)
            .unwrap();
        let mut entry = AudioEntry {
            audio,
            reference: parse_jams(&annotation).unwrap(),
            duration: 0.5,
            hashes_verified: false,
        };
        let first = transcribe_for_evaluation(&model, &entry, 2);
        let second = transcribe_for_evaluation(&model, &entry, 2);
        match (first, second) {
            (Ok(a), Ok(b)) => assert_eq!(a, b),
            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
            _ => panic!("synthetic-model inference must be deterministic"),
        }
        let (report, failures) =
            run_audio_entries(std::slice::from_ref(&entry), "tiny-test-only", 2, |e| {
                transcribe_for_evaluation(&model, e, 2)
            })
            .unwrap();
        assert!(report.contains("Entries: 1;") && report.contains("| Pitch + onset notes |"));
        if failures > 0 {
            assert!(report.contains("failed:"));
        }
        entry.duration = 6.0;
        assert!(transcribe_for_evaluation(&model, &entry, 2)
            .unwrap_err()
            .to_string()
            .contains("shorter"));
    }
    #[test]
    fn tdr_is_unclamped_and_undefined_without_pitch_matches() {
        let mut events = Vec::new();
        for reference in [true, false] {
            for string in [0, 1] {
                add_event(&mut events, reference, string, 50, 0.0, 0.02, 2);
            }
        }
        let counts = score(events, 2);
        assert_eq!(tdr(&counts), "4 / 2 = 200.00%");
        assert_eq!(tdr(&Counts::default()), "0 / 0 = n/a");
    }
    #[test]
    fn half_open_boundaries_and_unison_wrong_string() {
        assert_eq!(boundary(0.005, 100), 0);
        assert_eq!(boundary(0.015, 100), 1);
        assert_eq!(boundary(0.015001, 100), 2);
        let mut events = Vec::new();
        // A unison on two strings; prediction gets one string right and one wrong.
        for (r, s) in [(true, 0), (true, 1), (false, 1), (false, 2)] {
            add_event(&mut events, r, s, 60, 0.0, 0.02, 2);
        }
        let c = score(events, 2);
        assert_eq!(c.pitch_tp, 2);
        assert_eq!((c.tab_tp, c.tab_ref, c.tab_pred), (2, 4, 4));
        assert_eq!((c.wrong_string, c.pitch_string_matches), (2, 4));
    }
    #[test]
    fn sweep_matches_direct_frame_sampling_for_random_overlaps() {
        #[derive(Clone, Copy)]
        struct N {
            reference: bool,
            string: u8,
            pitch: u8,
            start: f64,
            end: f64,
        }
        let mut seed = 19u64;
        for _ in 0..200 {
            let mut notes = Vec::new();
            for i in 0..40 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let a = (seed >> 32) % 190;
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let b = a + 1 + (seed >> 32) % 50;
                notes.push(N {
                    reference: i % 2 == 0,
                    string: ((seed >> 20) % 6) as u8,
                    pitch: 40 + ((seed >> 12) % 12) as u8,
                    start: a as f64 / 1000.0,
                    end: b as f64 / 1000.0,
                });
            }
            let frames = 20;
            let mut events = Vec::new();
            for n in &notes {
                add_event(
                    &mut events,
                    n.reference,
                    n.string,
                    n.pitch,
                    n.start,
                    n.end,
                    frames,
                );
            }
            let actual = score(events, frames);
            let mut expected = Counts::default();
            for frame in 0..frames {
                let time = (frame as f64 + 0.5) * HOP;
                for pitch in 40..52 {
                    let mut r = 0u64;
                    let mut p = 0u64;
                    let mut eq = 0u64;
                    for string in 0..6 {
                        let present = |reference| {
                            notes.iter().any(|n| {
                                n.reference == reference
                                    && n.string == string
                                    && n.pitch == pitch
                                    && n.start <= time
                                    && time < n.end
                            })
                        };
                        let a = present(true);
                        let b = present(false);
                        r += u64::from(a);
                        p += u64::from(b);
                        eq += u64::from(a && b);
                    }
                    expected.pitch_ref += u64::from(r > 0);
                    expected.pitch_pred += u64::from(p > 0);
                    expected.pitch_tp += u64::from(r > 0 && p > 0);
                    expected.tab_ref += r;
                    expected.tab_pred += p;
                    expected.tab_tp += eq;
                    expected.pitch_string_matches += r.min(p);
                    expected.wrong_string += r.min(p) - eq;
                }
            }
            assert_eq!(actual, expected);
        }
    }
}
