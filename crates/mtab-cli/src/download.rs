//! `mtab download` — the explicit, user-invoked weights fetch (§6.12, §10.1).
//!
//! Rules this module exists to honour:
//!
//! - **Explicit only.** Nothing here runs unless the user typed `mtab download`.
//!   `mtab transcribe` never calls it, and never fetches weights on its own.
//! - **Authenticated, not anonymous.** The weights are gated: the token is read
//!   from the environment variable the user names (`--token-env`, default
//!   `HF_TOKEN`) and is never written to disk, to a log, or into an error
//!   message.
//! - **Atomic.** The file is written to a `.part` sibling and renamed only
//!   after the download completes, so an interrupted run cannot leave a
//!   truncated file that later looks like real weights.
//! - **Resumable.** A `.part` from a previous attempt is continued with a
//!   `Range` request; a server that ignores the range restarts cleanly.
//! - **Verified.** The size is checked against `/api/models/<repo>` before the
//!   bytes are trusted, and the finished file is opened with the same
//!   safetensors header parser `inspect-weights` uses.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use mtab_core::model::ModelSize;
use mtab_core::safetensors::WeightsInfo;
use mtab_core::{MtabError, Result};

/// Hugging Face rejects requests without a client that identifies itself.
const USER_AGENT: &str = concat!("mtab/", env!("CARGO_PKG_VERSION"));
const HUB: &str = "https://huggingface.co";

/// What the helper is being asked to do.
#[derive(Debug, Clone)]
pub struct DownloadRequest {
    /// Which variant to fetch; its HF repo is derived from this.
    pub size: ModelSize,
    /// Where to write the file.
    pub destination: PathBuf,
    /// Environment variable holding the token.
    pub token_env: String,
}

/// Run a download, reporting progress to stderr.
pub fn download(request: &DownloadRequest) -> Result<PathBuf> {
    let token = std::env::var(&request.token_env).map_err(|_| MtabError::LicenseGateRequired {
        repo: request.size.hf_repo().to_owned(),
    })?;
    let token = token.trim().to_owned();
    if token.is_empty() {
        return Err(MtabError::LicenseGateRequired {
            repo: request.size.hf_repo().to_owned(),
        });
    }

    let repo = request.size.hf_repo();
    if let Some(parent) = request.destination.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let expected = expected(repo, &token)?;
    let part = part_path(&request.destination);

    eprintln!(
        "downloading {repo} ({} bytes, CC BY-NC 4.0) to {}",
        expected.size,
        request.destination.display()
    );
    // A `.part` already at the expected size needs verifying, not fetching: a
    // Range request past the end would be a 416.
    let already = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if already > expected.size {
        eprintln!("discarding an oversized .part ({already} bytes); starting over");
        std::fs::remove_file(&part)?;
        fetch_to(repo, &token, &part, expected.size)?;
    } else if already == expected.size {
        eprintln!("a complete .part is already present; verifying it");
    } else {
        fetch_to(repo, &token, &part, expected.size)?;
    }

    let actual = std::fs::metadata(&part)?.len();
    if actual != expected.size {
        let _ = std::fs::remove_file(&part);
        return Err(MtabError::DecodeFailure {
            chunk_index: 0,
            reason: format!(
                "downloaded {actual} bytes but the Hub reports {}; the file was discarded",
                expected.size
            ),
        });
    }
    if let Some(want) = &expected.sha256 {
        let got = sha256_file(&part)?;
        if !got.eq_ignore_ascii_case(want) {
            let _ = std::fs::remove_file(&part);
            return Err(MtabError::DecodeFailure {
                chunk_index: 0,
                reason: format!(
                    "checksum mismatch: the Hub lists sha256 {want}, the download hashes to {got}; the file was discarded"
                ),
            });
        }
        eprintln!("sha256 verified");
    }

    // The bytes only become the destination name once they parse as
    // safetensors, so a bad download never masquerades as weights.
    let info = WeightsInfo::from_file(&part)?;
    match info.likely_size() {
        Some(seen) if seen == request.size => {}
        Some(seen) => {
            let _ = std::fs::remove_file(&part);
            return Err(MtabError::ShapeMismatch {
                expected: format!("{} weights", request.size),
                found: format!("{} weights", seen),
            });
        }
        None => {
            let _ = std::fs::remove_file(&part);
            return Err(MtabError::ShapeMismatch {
                expected: format!("{} weights", request.size),
                found: "a safetensors file matching no official size".to_owned(),
            });
        }
    }
    std::fs::rename(&part, &request.destination)?;
    eprintln!(
        "wrote {} ({} bytes, {} tensors)",
        request.destination.display(),
        expected.size,
        info.tensor_count()
    );
    Ok(request.destination.clone())
}

/// SHA-256 of a file, streamed so a 5 GB checkpoint does not have to be held
/// in memory.
fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)?;
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// `<destination>.part` — where an interrupted download lives.
fn part_path(destination: &Path) -> PathBuf {
    let mut name = destination.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

/// What the Hub says the file should be.
struct Expected {
    size: u64,
    /// SHA-256 the Hub records for the LFS object, when it publishes one.
    sha256: Option<String>,
}

/// Ask the Hub how big the file is, and what it hashes to. Doubles as a
/// license-acceptance check: a 401/403 here means the account has not accepted
/// the model's terms.
fn expected(repo: &str, token: &str) -> Result<Expected> {
    // `blobs=true` is what makes the Hub include each file's size and LFS hash;
    // without it the siblings carry only names.
    let url = format!("{HUB}/api/models/{repo}?blobs=true");
    let body = get_text(&url, token, None)?;
    let json: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| MtabError::DecodeFailure {
            chunk_index: 0,
            reason: format!("the Hub returned unparseable metadata: {e}"),
        })?;
    let entry = json["siblings"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|s| s["rfilename"] == "model.safetensors")
        .ok_or_else(|| MtabError::DecodeFailure {
            chunk_index: 0,
            reason: format!(
                "{repo} has no model.safetensors in its file listing; check --size and that the token can see the repo"
            ),
        })?;
    let size = entry["size"]
        .as_u64()
        .ok_or_else(|| MtabError::DecodeFailure {
            chunk_index: 0,
            reason: format!("{repo} does not publish a size for model.safetensors"),
        })?;
    Ok(Expected {
        size,
        sha256: entry["lfs"]["sha256"].as_str().map(str::to_owned),
    })
}

fn get_text(url: &str, token: &str, range_from: Option<u64>) -> Result<String> {
    let response = request(url, token, range_from)?;
    let mut text = String::new();
    let mut reader = response.into_body().into_reader();
    reader.read_to_string(&mut text)?;
    Ok(text)
}

fn request(
    url: &str,
    token: &str,
    range_from: Option<u64>,
) -> Result<ureq::http::Response<ureq::Body>> {
    if !url.starts_with("https://") {
        return Err(MtabError::InvalidModelSource(format!(
            "refusing a non-HTTPS download from {url}"
        )));
    }
    let mut call = ureq::get(url)
        .header("user-agent", USER_AGENT)
        .header("authorization", format!("Bearer {token}"));
    if let Some(from) = range_from {
        call = call.header("range", format!("bytes={from}-"));
    }
    match call.call() {
        Ok(response) => Ok(response),
        Err(ureq::Error::StatusCode(401 | 403)) => Err(MtabError::LicenseGateRequired {
            repo: url.to_owned(),
        }),
        // Asking for a range past the end means the local file is stale or
        // wrong; saying so beats a bare status code.
        Err(ureq::Error::StatusCode(416)) => Err(MtabError::DecodeFailure {
            chunk_index: 0,
            reason: "the server rejected the resume offset; delete the .part file and retry".into(),
        }),
        Err(ureq::Error::StatusCode(code)) => Err(MtabError::DecodeFailure {
            chunk_index: 0,
            reason: format!("the Hub answered {code} for {url}"),
        }),
        Err(error) => Err(MtabError::DecodeFailure {
            chunk_index: 0,
            reason: format!("network error for {url}: {error}"),
        }),
    }
}

fn fetch_to(repo: &str, token: &str, part: &Path, expected: u64) -> Result<()> {
    let url = format!("{HUB}/{repo}/resolve/main/model.safetensors");
    let already = std::fs::metadata(part).map(|m| m.len()).unwrap_or(0);
    debug_assert!(already < expected, "caller only fetches an incomplete file");
    let response = if already > 0 {
        eprintln!("resuming at byte {already}");
        request(&url, token, Some(already))?
    } else {
        request(&url, token, None)?
    };
    // 206 means the range was honoured; 200 means the server ignored it and we
    // must start over rather than appending to what we already have.
    let continued = response.status() == 206;
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(continued)
        .truncate(!continued)
        .write(true)
        .open(part)?;

    let mut reader = response.into_body().into_reader();
    let mut buffer = vec![0u8; 1 << 20];
    let mut written = if continued { already } else { 0 };
    let mut last_report = std::time::Instant::now();
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        out.write_all(&buffer[..read])?;
        written += read as u64;
        if last_report.elapsed().as_secs() >= 5 {
            eprintln!("  {written} bytes");
            last_report = std::time::Instant::now();
        }
    }
    out.flush()?;
    out.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_path_appends_a_suffix() {
        assert_eq!(
            part_path(Path::new("weights/m.safetensors")),
            Path::new("weights/m.safetensors.part")
        );
    }

    #[test]
    fn refuses_plain_http() {
        let error = request("http://example.invalid/x", "t", None)
            .map(|_| ())
            .unwrap_err();
        assert!(error.to_string().contains("non-HTTPS"), "{error}");
    }

    #[test]
    fn missing_token_names_the_license_gate() {
        let request = DownloadRequest {
            size: ModelSize::Small,
            destination: PathBuf::from("/tmp/never-written.safetensors"),
            token_env: "MTAB_TEST_TOKEN_THAT_IS_NOT_SET".to_owned(),
        };
        let error = download(&request).map(|_| ()).unwrap_err();
        assert!(error.to_string().contains("gated"), "{error}");
        assert!(!Path::new("/tmp/never-written.safetensors").exists());
    }
}
