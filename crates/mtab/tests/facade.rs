//! The §6.11 facade, exercised end to end.
//!
//! These tests load the committed tiny checkpoint (`fixture_tiny`: 2 layers,
//! 32 dim) and run the real pipeline — chunking, MT3 tracking, string/fret
//! assignment and every exporter — over a silent WAV. The tiny weights are
//! synthetic, so the output is one empty track; what is under test is the
//! wiring and the contract, not transcription quality.

use std::path::PathBuf;

use muscriptor_tab::{
    Capo, CapoChoice, DecodeConfig, Grid, GuitarConfig, ModelSize, Progress, Stage, Transcriber,
    Tuning, TuningChoice,
};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../mtab-core/tests/fixtures/model/tiny.fixture")
}

/// A 1024-sample silent mono WAV at 16 kHz, written to a unique temp path.
fn silent_wav(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "mtab-facade-{}-{tag}-{}.wav",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let n = 1024u32;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + n * 2).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&16_000u32.to_le_bytes());
    bytes.extend_from_slice(&32_000u32.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(n * 2).to_le_bytes());
    bytes.resize(44 + (n * 2) as usize, 0);
    std::fs::write(&path, bytes).unwrap();
    path
}

/// The committed tiny checkpoint, loaded with its own config. It is not one of
/// the three shipped sizes, so it goes in as a model rather than a file path.
fn tiny_model() -> muscriptor_tab::LmModel {
    muscriptor_tab::LmModel::load(mtab_core::lm::LmConfig::fixture_tiny(), &fixture()).unwrap()
}

fn builder() -> muscriptor_tab::TranscriberBuilder {
    Transcriber::builder()
        .size(ModelSize::Small)
        .model(tiny_model())
        .decode(DecodeConfig {
            max_gen_len: 8,
            ..DecodeConfig::default()
        })
}

#[test]
fn builder_requires_weights() {
    let error = Transcriber::builder()
        .size(ModelSize::Small)
        .build()
        .map(|_| ())
        .unwrap_err();
    // No default weights: the NC licence makes silently picking some a bug.
    assert!(
        error.to_string().contains("weights not found"),
        "unexpected error: {error}"
    );
}

#[test]
fn builder_rejects_an_unsupported_grid_instead_of_ignoring_it() {
    let error = builder()
        .grid(Grid::Divisions {
            per_quarter: 8,
            triplet: false,
        })
        .build()
        .unwrap_err();
    assert!(
        error.to_string().contains("not yet implemented"),
        "unexpected error: {error}"
    );
}

#[test]
fn builder_rejects_a_nonsense_fretboard() {
    let error = builder()
        .guitar(GuitarConfig {
            frets: 0,
            ..GuitarConfig::default()
        })
        .build()
        .unwrap_err();
    assert!(
        error.to_string().contains("frets must be in 1..=36"),
        "unexpected error: {error}"
    );
}

#[test]
fn transcribes_a_file_through_the_public_api() {
    let wav = silent_wav("transcribe");
    let transcriber = builder()
        .guitar(GuitarConfig::fixed_standard(22))
        .build()
        .unwrap();
    let tab = transcriber.transcribe_file(&wav).unwrap();
    let _ = std::fs::remove_file(&wav);

    // Silent audio gives no notes; the document still has one usable track so
    // every exporter has something legal to write.
    assert_eq!(tab.document.tracks.len(), 1);
    assert!(tab.document.tracks[0].notes.is_empty());
    assert_eq!(tab.document.tracks[0].tuning, Tuning::standard());
    assert_eq!(tab.document.tracks[0].capo, Capo::NONE);
    assert_eq!(tab.document.meta.frets, 22);
    assert!(tab.omitted.is_empty());
}

#[test]
fn every_exporter_writes_a_parseable_file() {
    let wav = silent_wav("export");
    let transcriber = builder().build().unwrap();
    let tab = transcriber.transcribe_file(&wav).unwrap();
    let _ = std::fs::remove_file(&wav);

    let dir = std::env::temp_dir();
    let base = format!("mtab-facade-export-{}", std::process::id());

    let midi = dir.join(format!("{base}.mid"));
    tab.export_midi(&midi).unwrap();
    let bytes = std::fs::read(&midi).unwrap();
    assert_eq!(&bytes[..4], b"MThd", "MIDI must start with a header chunk");
    assert!(bytes.len() > 20);

    let gp5 = dir.join(format!("{base}.gp5"));
    tab.export_guitar_pro(&gp5).unwrap();
    let bytes = std::fs::read(&gp5).unwrap();
    assert!(
        bytes.starts_with(b"\x18FICHIER GUITAR PRO v5.00"),
        "GP5 must carry the v5.00 magic"
    );

    let xml = dir.join(format!("{base}.musicxml"));
    tab.export_musicxml(&xml).unwrap();
    let text = std::fs::read_to_string(&xml).unwrap();
    assert!(text.contains("<score-partwise"), "MusicXML root element");
    assert!(text.contains("<staff-details"), "tab staff is present");

    let txt = dir.join(format!("{base}.txt"));
    tab.export_ascii(&txt).unwrap();
    let ascii = std::fs::read_to_string(&txt).unwrap();
    assert_eq!(ascii, tab.ascii_tab().unwrap());
    // An empty document still prints a legal (if bare) tab block.
    assert!(!ascii.trim().is_empty(), "ASCII tab must not be blank");

    // The IR round-trips as JSON: it is the debugging contract.
    let ir = tab.ir_json().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&ir).unwrap();
    assert!(parsed["tracks"].is_array());

    for path in [midi, gp5, xml, txt] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn progress_reaches_the_caller_in_pipeline_order() {
    let wav = silent_wav("progress");
    let transcriber = builder().build().unwrap();
    let mut stages = Vec::new();
    transcriber
        .transcribe_file_with_progress(&wav, |p: Progress| {
            let last = stages.last();
            if last != Some(&p.stage) {
                stages.push(p.stage);
            }
        })
        .unwrap();
    let _ = std::fs::remove_file(&wav);

    // Audio → Decode → Fingering, never backwards, and never skipping the
    // expensive middle stage.
    assert!(stages.contains(&Stage::Audio));
    assert!(stages.contains(&Stage::Decode));
    assert!(stages.contains(&Stage::Fingering));
    let audio = stages.iter().position(|s| *s == Stage::Audio).unwrap();
    let decode = stages.iter().position(|s| *s == Stage::Decode).unwrap();
    assert!(audio < decode, "decode cannot precede audio: {stages:?}");
}

#[test]
fn a_fixed_tuning_skips_detection() {
    let wav = silent_wav("fixed");
    let transcriber = builder()
        .guitar(GuitarConfig {
            tuning: TuningChoice::Fixed(Tuning::from_name("drop-d").unwrap()),
            capo: CapoChoice::Fixed(Capo::new(2).unwrap()),
            ..GuitarConfig::default()
        })
        .build()
        .unwrap();
    let mut saw_tuning_stage = false;
    let tab = transcriber
        .transcribe_file_with_progress(&wav, |p| {
            if p.stage == Stage::Tuning {
                saw_tuning_stage = true;
            }
        })
        .unwrap();
    let _ = std::fs::remove_file(&wav);

    // Both halves are fixed, so no evidence is analysed at all.
    assert!(!saw_tuning_stage);
    assert_eq!(tab.document.tracks[0].tuning.name, "drop-d");
    assert_eq!(tab.document.tracks[0].capo.position(), 2);
}

#[test]
fn one_transcriber_serves_many_inputs() {
    // §6.11's point: the weights load once. Two calls on one instance must
    // both succeed and agree, which is also the determinism requirement.
    let first = silent_wav("many-a");
    let second = silent_wav("many-b");
    let transcriber = builder().build().unwrap();
    let a = transcriber.transcribe_file(&first).unwrap();
    let b = transcriber.transcribe_file(&second).unwrap();
    let again = transcriber.transcribe_file(&first).unwrap();
    for path in [first, second] {
        let _ = std::fs::remove_file(path);
    }
    assert_eq!(a.ir_json().unwrap(), b.ir_json().unwrap());
    assert_eq!(a.ir_json().unwrap(), again.ir_json().unwrap());
}
