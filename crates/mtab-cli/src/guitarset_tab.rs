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
    Ok(Scored {
        counts: score(events, frames),
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
    let output = format!("# GuitarSet framewise IR evaluation\n\nA user-supplied TabDocument and JAMS file must describe the same recording. Both use standard tuning and no capo. The evaluator uses fixed MIDI note pitches, not continuous F0 annotations; pitch bends, technique, velocity and instrument labels are ignored. Multiple predicted tracks are combined.\n\n- Window: first {duration:.6} seconds; frame centers at 5 ms + 10 ms × k, strictly before the window end\n- Frame count: {frames}\n- Reference notes: {}\n- Predicted notes: {}\n\n| Metric | TP | FP | FN | Precision | Recall | F1 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n{}\n{}\n\nWrong-string among pitch-matched string instances: {} / {} = {}. Per frame, identical pitch/string duplicates are collapsed; matching same-pitch instances maximizes same-string matches first. The multi-pitch row collapses unisons to pitch sets. The tablature row uses (string, MIDI pitch), equivalent to (string, fret) in standard tuning.\n\nThis is a single-example fixed-grid note-frame metric, not the published TabCNN TDR, a held-out corpus result, or a tab-labeler-v3 comparison. GuitarSet 1.1.0, Xi et al., CC BY 4.0; known annotation errors may affect scores.\n", scored.reference_notes, scored.predicted_notes, row("Multi-pitch frames",c.pitch_tp,c.pitch_ref,c.pitch_pred), row("Tab position frames",c.tab_tp,c.tab_ref,c.tab_pred), c.wrong_string,c.pitch_string_matches,percent(c.wrong_string,c.pitch_string_matches));
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
    let output = format!("# GuitarSet framewise IR manifest evaluation\n\n{} manifest entries were all scored; inputs are user-supplied precomputed IR, not inference run by this command. Relative paths resolve against the manifest directory. Each duration is user-provided.\n\n- Total frames: {total_frames}\n\n| Micro-aggregated metric | TP | FP | FN | Precision | Recall | F1 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n{}\n{}\n\nWrong-string among pitch-matched string instances: {} / {} = {}.\n\n| Manifest entry | Frames | Reference notes | Predicted notes | Multi-pitch F1 | Tab-position F1 |\n| ---: | ---: | ---: | ---: | ---: | ---: |\n{details}\n10 ms frame centers; standard tuning and no capo; half-open note intervals; fixed MIDI note pitches rather than F0 contours. Multi-pitch unisons collapse to pitch sets; tab-position duplicates collapse to string/pitch sets. Wrong-string pairs maximize same-string matches among equal-pitch instances. Multiple IR tracks are combined.\n\nNot published TabCNN TDR, a verified held-out split, or a tab-labeler-v3 comparison. Manifest selection and model/decoding provenance must be recorded separately. GuitarSet 1.1.0, Xi et al., CC BY 4.0; annotations may contain errors.\n", entries.len(), row("Multi-pitch frames",total.pitch_tp,total.pitch_ref,total.pitch_pred), row("Tab position frames",total.tab_tp,total.tab_ref,total.tab_pred), total.wrong_string,total.pitch_string_matches,percent(total.wrong_string,total.pitch_string_matches));
    if let Some(path) = report {
        std::fs::write(path, output)?;
    } else {
        print!("{output}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
