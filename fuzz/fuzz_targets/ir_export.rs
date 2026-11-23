#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 512 * 1024 {
        return;
    }
    // A tablature IR is user-supplied input for the exporters (`mtab eval`
    // reads it from disk), so hostile or merely broken JSON must be rejected
    // without panicking or hanging. All three notation writers render the same
    // plan, and MIDI does not use it, so exercise all four.
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(doc) = mtab_guitar::TabDocument::from_json(text) else {
        return;
    };
    fn check_plan(plan: &mtab_export::gp5_plan::ScorePlan) {
        // The plan bounds beats, tracks and notes, so the output is bounded
        // too; a valid plan never reports a negative or NaN displacement.
        assert!(plan.max_boundary_shift_seconds.is_finite());
        assert!(plan.max_boundary_shift_seconds >= 0.0);
        assert!((4..=255).contains(&plan.bpm));
    }
    if let Ok((bytes, plan)) = mtab_export::gp5::gp5_bytes(&doc) {
        assert!(!bytes.is_empty());
        check_plan(&plan);
    }
    if let Ok((text, plan)) = mtab_export::ascii::ascii_tab(&doc) {
        assert!(!text.is_empty());
        check_plan(&plan);
    }
    if let Ok((text, plan)) = mtab_export::musicxml::musicxml(&doc) {
        assert!(!text.is_empty());
        check_plan(&plan);
    }
    // MIDI is bounded by the exporter's own note limits; it must not panic.
    let _ = mtab_export::midi::midi_bytes(&doc);
});
