//! Development-only dynamics sanity check on a synthesized MIDI fixture.
//! This is not real guitar, model inference, or velocity calibration.
use mtab_audio::{dynamics::estimate_velocities, load_model_input};
use mtab_core::NoteEvent;
use std::path::Path;

fn main() -> mtab_core::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("rendered WAV path");
    let mode = args.next().expect("rising or flat");
    assert!(
        mode == "rising" || mode == "flat",
        "expected rising or flat"
    );
    assert!(args.next().is_none(), "unexpected argument");
    let audio = load_model_input(Path::new(&path))?;
    let mut notes: Vec<_> = [(0.2, 0.5), (0.8, 1.1), (1.4, 1.7)]
        .into_iter()
        .map(|(start, end)| NoteEvent {
            start,
            end,
            midi: 45,
            velocity: 100,
            instrument: 24,
        })
        .collect();
    let report = estimate_velocities(&audio, &mut notes)?;
    let velocities: Vec<_> = notes.iter().map(|n| n.velocity).collect();
    println!("{mode} fixture estimated velocities: {velocities:?}; {report:?}");
    assert_eq!(report.estimated, 3);
    assert_eq!(report.skipped, 0);
    let rising = velocities[0] < velocities[1] && velocities[1] < velocities[2];
    assert_eq!(rising, mode == "rising", "unexpected velocity ordering");
    Ok(())
}
