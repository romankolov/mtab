//! `mtab-export` — writers (PROMPT.md §6.10).
//!
//! - MIDI (`.mid`, SMF type 1 with tempo map + bends) — DONE (M4)
//! - Guitar Pro 5 (`.gp5`) — experimental restricted subset (M6)
//! - ASCII tab — experimental beat-aligned subset (M6)
//! - MusicXML 3.1 — experimental restricted subset (M6)
//!
//! Deterministic output: identical input ⇒ byte-identical export (§10.7).

pub mod ascii;
pub mod gp5;
pub mod gp5_plan;
pub mod midi;
pub mod musicxml;

use std::path::Path;

use mtab_core::Result;
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
/// GP5 supports only the restricted subset documented by [`gp5::gp5_bytes`].
/// ASCII and MusicXML use the same restricted rhythmic plan; MusicXML rejects bends.
pub fn export(doc: &TabDocument, path: &Path, format: ExportFormat) -> Result<()> {
    match format {
        ExportFormat::Midi => midi::write_midi(doc, path),
        ExportFormat::GuitarPro5 => gp5::write_gp5(doc, path).map(|_| ()),
        ExportFormat::MusicXml => musicxml::write_musicxml(doc, path),
        ExportFormat::AsciiTab => ascii::write_ascii(doc, path),
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
    fn export_midi_empty_document() {
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
        // MIDI now works end-to-end through the dispatcher.
        let tmp = std::env::temp_dir().join(format!("mtab-export-test-{}.mid", std::process::id()));
        export(&doc, &tmp, ExportFormat::Midi).unwrap();
        assert!(std::fs::metadata(&tmp).unwrap().len() > 0);
        std::fs::remove_file(&tmp).ok();
    }
}
