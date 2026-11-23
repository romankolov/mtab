//! Development-only F0 sanity check on the separately rendered MIDI fixture.
//! This does not test real guitar audio, model inference, or bend accuracy.
use mtab_audio::{bend_analysis::suggest_bends, load_model_input, pitch_track::pitch_track};
use mtab_core::NoteEvent;
use std::path::Path;

fn main() -> mtab_core::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("rendered WAV path");
    let mode = args.next().expect("bend or flat");
    assert!(mode == "bend" || mode == "flat", "expected bend or flat");
    assert!(args.next().is_none(), "unexpected argument");
    let audio = load_model_input(Path::new(&path))?;
    let note = NoteEvent {
        start: 0.0,
        end: 1.0,
        midi: 60,
        velocity: 92,
        instrument: 24,
    };
    let frames = pitch_track(&audio, note.start, note.end)?;
    let base = 440.0 * 2.0_f64.powf(-9.0 / 12.0);
    for time in [0.1_f64, 0.3, 0.5, 0.7, 0.9] {
        let frame = frames
            .iter()
            .min_by(|a, b| (a.time - time).abs().total_cmp(&(b.time - time).abs()))
            .expect("no pitch frames");
        assert!((frame.time - time).abs() <= 0.009, "missing nearby frame");
        let hz = frame.frequency_hz.expect("unvoiced reference frame");
        let cents = 1200.0 * (hz / base).log2();
        // Account for the actual frame center rather than the requested time.
        let expected = if mode == "bend" {
            200.0 * frame.time.min(1.0 - frame.time)
        } else {
            0.0
        };
        println!(
            "{:.3}s: {cents:.1} cents, expected {expected:.1}",
            frame.time
        );
        assert!((cents - expected).abs() <= 20.0, "pitch mismatch");
    }
    let assessments = suggest_bends(&audio, &[note])?;
    println!("experimental bend suggestion: {:?}", assessments[0]);
    if mode == "bend" {
        match assessments[0] {
            mtab_audio::bend_analysis::BendAssessment::Candidate(candidate) => {
                assert_eq!(
                    candidate.shape,
                    mtab_audio::bend_analysis::BendShape::BendRelease
                );
                assert!((80.0..=120.0).contains(&candidate.depth_cents));
                assert!((0.35..=0.6).contains(&candidate.peak_time));
            }
            other => panic!("rendered bend did not yield a bend/release suggestion: {other:?}"),
        }
    } else {
        assert!(
            !matches!(
                assessments[0],
                mtab_audio::bend_analysis::BendAssessment::Candidate(_)
            ),
            "flat control produced a bend candidate"
        );
    }
    Ok(())
}
