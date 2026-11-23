#![no_main]

use libfuzzer_sys::fuzz_target;
use std::path::PathBuf;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > 1024 * 1024 {
        return;
    }
    // The first byte selects the hint; it is not part of the file.
    // Choose a format hint separately from payload; otherwise arbitrary data
    // can still be identified by Symphonia's content probe. Each process has
    // its own temporary file, including when libFuzzer runs worker processes.
    let extensions = ["wav", "mp3", "flac", "ogg", "m4a", "aiff", "caf"];
    let ext = extensions[usize::from(data[0]) % extensions.len()];
    let path: PathBuf = std::env::temp_dir()
        .join(format!("mtab-audio-fuzz-{}.{}", std::process::id(), ext));
    std::fs::write(&path, &data[1..]).expect("write fuzz input");
    let result = mtab_audio::decode::load(&path);
    std::fs::remove_file(path).expect("remove fuzz input");
    if let Ok(audio) = result {
        assert!(audio.sample_rate > 0);
        assert!(audio.channels > 0);
        assert_eq!(audio.samples.len() % usize::from(audio.channels), 0);
        assert!(!audio.samples.is_empty());
        assert!(audio.samples.len() <= mtab_audio::decode::MAX_DECODED_SAMPLES);
    }
});
