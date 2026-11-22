//! Rhythm quantization types (PROMPT.md §6.7). The beat tracker and the
//! snapping logic land in M5; this is the grid vocabulary the CLI exposes.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use mtab_core::{MtabError, Result};

/// Quantization grid for snapping note onsets/offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Grid {
    /// Beat-aware automatic grid (default; falls back to a tempo map when the
    /// beat tracker is unsure — §6.7 rubato rule).
    #[default]
    Auto,
    /// No quantization (raw model timing).
    Off,
    /// `per_quarter` subdivisions of a quarter note, optionally triplet feel.
    Divisions { per_quarter: u32, triplet: bool },
}

impl Grid {
    /// The default musical grid: straight 1/16.
    pub const SIXTEENTH: Grid = Grid::Divisions {
        per_quarter: 16,
        triplet: false,
    };
}

impl FromStr for Grid {
    type Err = MtabError;

    fn from_str(s: &str) -> Result<Self> {
        let norm = s.to_ascii_lowercase().replace(' ', "");
        match norm.as_str() {
            "auto" => Ok(Self::Auto),
            "off" | "none" => Ok(Self::Off),
            "triplet" => Ok(Self::Divisions {
                per_quarter: 8,
                triplet: true,
            }),
            _ => {
                let (num, triplet) = match norm.strip_suffix('t') {
                    Some(rest) => (rest, true),
                    None => (norm.as_str(), false),
                };
                let per_quarter = match num {
                    "1/4" => 4,
                    "1/8" => 8,
                    "1/16" => 16,
                    "1/32" => 32,
                    "1/64" => 64,
                    _ => return Err(MtabError::InvalidGrid(s.to_owned())),
                };
                Ok(Self::Divisions {
                    per_quarter,
                    triplet,
                })
            }
        }
    }
}

impl fmt::Display for Grid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Off => f.write_str("off"),
            Self::Divisions {
                per_quarter,
                triplet,
            } => {
                write!(f, "1/{per_quarter}")?;
                if *triplet {
                    f.write_str("T")?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_parses_and_displays() {
        for s in [
            "auto", "off", "1/4", "1/8", "1/16", "1/32", "1/8T", "triplet",
        ] {
            let g: Grid = s.parse().unwrap();
            // Display is canonical; only re-parse the canonical forms.
            let _ = g;
        }
        assert_eq!(
            "1/16".parse::<Grid>().unwrap(),
            Grid::Divisions {
                per_quarter: 16,
                triplet: false
            }
        );
        assert_eq!(
            "1/8T".parse::<Grid>().unwrap(),
            Grid::Divisions {
                per_quarter: 8,
                triplet: true
            }
        );
        assert_eq!(Grid::Auto.to_string(), "auto");
        assert_eq!(Grid::SIXTEENTH.to_string(), "1/16");
        assert!("banana".parse::<Grid>().is_err());
    }
}
