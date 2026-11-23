//! Model-free end-to-end checks of Hugging Face cache resolution and safety.
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn cached_sources_resolve_offline_and_cannot_overwrite_blob_alias() {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        Scratch(std::env::temp_dir().join(format!("mtab-hf-cli-{}-{n}", std::process::id())));
    let repo = root.0.join("hub/models--MuScriptor--muscriptor-small");
    let hash = "a".repeat(40);
    let snapshot = repo.join("snapshots").join(&hash);
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(repo.join("refs/main"), hash).unwrap();
    let weights = snapshot.join("model.safetensors");
    let content = b"not a model; deterministic safety fixture";
    std::fs::write(&weights, content).unwrap();
    let audio = root.0.join("input.wav");
    std::fs::write(&audio, b"not decoded: weights fail first").unwrap();
    let alias = root.0.join("output.mid");
    std::fs::hard_link(&weights, &alias).unwrap();
    for source in ["small", "hf://MuScriptor/muscriptor-small"] {
        let output = Command::new(env!("CARGO_BIN_EXE_mtab"))
            .env("HF_HUB_CACHE", root.0.join("hub"))
            .args([
                "transcribe",
                audio.to_str().unwrap(),
                "--weights",
                source,
                "--size",
                "small",
                "-o",
                alias.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success() && error.contains("resolved weights"),
            "{source}: {error}"
        );
        assert_eq!(std::fs::read(&weights).unwrap(), content);
    }
    std::fs::remove_file(&alias).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_mtab"))
        .env("HF_HUB_CACHE", root.0.join("hub"))
        .args([
            "transcribe",
            audio.to_str().unwrap(),
            "--weights",
            "small",
            "--size",
            "small",
            "-o",
            alias.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success() && !error.contains("weights not found"),
        "{error}"
    );
    assert!(!alias.exists());
    assert_eq!(std::fs::read(&weights).unwrap(), content);
    let output = Command::new(env!("CARGO_BIN_EXE_mtab"))
        .env_remove("HF_HUB_CACHE")
        .env("HF_HOME", &root.0)
        .args([
            "transcribe",
            audio.to_str().unwrap(),
            "--weights",
            "hf://MuScriptor/muscriptor-small",
            "--size",
            "small",
            "-o",
            alias.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success() && !error.contains("weights not found"),
        "HF_HOME: {error}"
    );
    let output = Command::new(env!("CARGO_BIN_EXE_mtab"))
        .env("HF_HUB_CACHE", root.0.join("hub"))
        .args([
            "transcribe",
            audio.to_str().unwrap(),
            "--weights",
            "small",
            "--size",
            "medium",
            "-o",
            alias.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success() && error.contains("does not match --size"),
        "{error}"
    );
}
