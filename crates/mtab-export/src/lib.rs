//! `mtab-export` — writers (PROMPT.md §6.10).
//!
//! - MIDI (`.mid`, SMF type 1 with tempo map + bends) — DONE (M4)
//! - Guitar Pro 5 (`.gp5`) — M6
//! - MusicXML 3.1 + ASCII tab — M6
//!
//! Deterministic output: identical input ⇒ byte-identical export (§10.7).

pub mod gp5;
pub mod gp5_plan;
pub mod midi;

use std::path::Path;

use mtab_core::{MtabError, Result};
use mtab_guitar::TabDocument;

/// Supported export formats (PROMPT.md §1: MIDI + Guitar Pro are the
/// product; MusicXML + ASCII tab are included bonuses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Midi,
    GuitarPro5,
    MusicXml,
    AsciiTab,
}

impl ExportFormat {
    /// Infer from a file extension (`.mid`, `.gp5`, `.musicxml`, `.txt`/`.tab`).
    pub fn from_path(path: &Path) -> Option<Self> {
        match path
            .extension()
            .and_then(|e| e.to_str())?
            .to_ascii_lowercase()
            .as_str()
        {
            "mid" | "midi" => Some(Self::Midi),
            "gp5" => Some(Self::GuitarPro5),
            "musicxml" | "xml" => Some(Self::MusicXml),
            "txt" | "tab" => Some(Self::AsciiTab),
            _ => None,
        }
    }
}

/// Export a document to `path` in `format`.
///
/// Writers land per the milestone plan (M4/M6); until then this returns
/// [`MtabError::NotImplemented`] so the CLI can say exactly what is missing.
pub fn export(doc: &TabDocument, path: &Path, format: ExportFormat) -> Result<()> {
    match format {
        ExportFormat::Midi => midi::write_midi(doc, path),
        ExportFormat::GuitarPro5 => Err(MtabError::NotImplemented(
            "Guitar Pro 5 export (milestone M6)",
        )),
        ExportFormat::MusicXml => Err(MtabError::NotImplemented("MusicXML export (milestone M6)")),
        ExportFormat::AsciiTab => Err(MtabError::NotImplemented("ASCII tab export (milestone M6)")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_from_extension() {
        assert_eq!(
            ExportFormat::from_path(Path::new("song.mid")),
            Some(ExportFormat::Midi)
        );
        assert_eq!(
            ExportFormat::from_path(Path::new("song.GP5")),
            Some(ExportFormat::GuitarPro5)
        );
        assert_eq!(
            ExportFormat::from_path(Path::new("song.musicxml")),
            Some(ExportFormat::MusicXml)
        );
        assert_eq!(
            ExportFormat::from_path(Path::new("song.txt")),
            Some(ExportFormat::AsciiTab)
        );
        assert_eq!(ExportFormat::from_path(Path::new("song.wav")), None);
    }

    #[test]
    fn export_reports_not_implemented() {
        let doc = TabDocument {
            meta: mtab_guitar::TrackMeta {
                title: None,
                tempo_map: mtab_guitar::TempoMap::constant(120.0),
                time_signatures: vec![],
                frets: 22,
                source_hash: None,
            },
            tracks: vec![],
            chord_symbols: vec![],
        };
        let err = export(&doc, Path::new("x.gp5"), ExportFormat::GuitarPro5).unwrap_err();
        assert!(matches!(err, MtabError::NotImplemented(_)));
        // MIDI now works end-to-end through the dispatcher.
        let tmp = std::env::temp_dir().join(format!("mtab-export-test-{}.mid", std::process::id()));
        export(&doc, &tmp, ExportFormat::Midi).unwrap();
        assert!(std::fs::metadata(&tmp).unwrap().len() > 0);
        std::fs::remove_file(&tmp).ok();
    }
}
