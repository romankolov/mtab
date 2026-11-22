//! Small, real-audio F0 regression against GuitarSet per-string annotations.
//! Reference note boundaries and strings are provided: not end-to-end accuracy.
use mtab_audio::pitch_track::PitchTracker;
use mtab_core::NoteEvent;
use serde_json::Value;
use std::path::PathBuf;

#[test]
fn isolated_excerpt_pitch_frames_against_guitarset_contours() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/guitarset");
    let audio = mtab_audio::load_model_input(&dir.join("guitarset_solo_5s.wav")).unwrap();
    let jams: Value =
        serde_json::from_slice(&std::fs::read(dir.join("guitarset_solo_5s.jams")).unwrap())
            .unwrap();
    let contours: Value =
        serde_json::from_slice(&std::fs::read(dir.join("guitarset_solo_5s_pitch.json")).unwrap())
            .unwrap();
    let mut notes = Vec::new();
    for ann in jams["annotations"].as_array().unwrap() {
        let string: usize = ann["annotation_metadata"]["data_source"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        for obs in ann["data"].as_array().unwrap() {
            let start = obs["time"].as_f64().unwrap();
            let end = start + obs["duration"].as_f64().unwrap();
            notes.push((
                string,
                NoteEvent {
                    start,
                    end,
                    midi: obs["value"].as_f64().unwrap().round() as u8,
                    velocity: 100,
                    instrument: 24,
                },
            ));
        }
    }
    assert_eq!(notes.len(), 20);
    let raw_notes: Vec<_> = notes.iter().map(|(_, note)| *note).collect();
    let suggestions = mtab_audio::bend_analysis::suggest_bends(&audio, &raw_notes).unwrap();
    let candidates = suggestions
        .iter()
        .filter(|s| matches!(s, mtab_audio::bend_analysis::BendAssessment::Candidate(_)))
        .count();
    eprintln!(
        "bend suggestions from reference notes: {candidates} of {}",
        raw_notes.len()
    );
    let mut tracker = PitchTracker::new(&audio).unwrap();
    let mut assessed_notes = 0;
    let mut eligible = 0;
    let mut voiced = 0;
    let mut errors = Vec::new();
    for (index, (string, note)) in notes.iter().enumerate() {
        if notes
            .iter()
            .enumerate()
            .any(|(other, (_, n))| other != index && n.start < note.end && note.start < n.end)
        {
            continue;
        }
        let contour = contours["strings"][*string].as_array().unwrap();
        let frames = tracker.track(note.start, note.end).unwrap();
        if frames.is_empty() {
            continue;
        }
        assessed_notes += 1;
        eprintln!(
            "isolated reference string={string} onset={:.3}s offset={:.3}s midi={}",
            note.start, note.end, note.midi
        );
        for frame in frames {
            let closest = contour
                .iter()
                .map(|pair| (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap()))
                .min_by(|a, b| {
                    (a.0 - frame.time)
                        .abs()
                        .total_cmp(&(b.0 - frame.time).abs())
                });
            let Some((time, reference_hz)) = closest else {
                continue;
            };
            if (time - frame.time).abs() > 0.01 {
                continue;
            }
            eligible += 1;
            if let Some(hz) = frame.frequency_hz {
                voiced += 1;
                errors.push((1200.0 * (hz / reference_hz).log2()).abs());
            }
        }
    }
    errors.sort_by(f64::total_cmp);
    let accurate = errors.iter().filter(|&&e| e < 50.0).count();
    eprintln!("GuitarSet 5s isolated reference notes={assessed_notes}, reference frames={eligible}, voiced={voiced}, within-50c={accurate}, median_abs_cents={:.2}", errors[errors.len()/2]);
    assert!(assessed_notes > 0 && eligible >= 10 && voiced >= 10);
    // This threshold is a fixture regression guard, not a dataset benchmark.
    assert!(accurate * 2 >= eligible, "too few frames within 50 cents");
}
