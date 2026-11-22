//! Tunings and capo (PROMPT.md §6.5).
//!
//! Tunings are stored as open-string MIDI pitches, low string first.
//! The library below is the template set the preliminary detector scores
//! against and the user can select manually.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use mtab_core::{MtabError, Result};

/// A guitar tuning: open-string MIDI pitches, low string first (6–8 strings).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tuning {
    /// Open-string MIDI pitches, low string first. Strictly ascending.
    pub strings: Vec<u8>,
    /// Canonical name (e.g. "standard", "drop-d"); "custom" for hand-built.
    pub name: String,
}

impl Tuning {
    pub const MIN_STRINGS: usize = 6;
    pub const MAX_STRINGS: usize = 8;

    /// Build a tuning from raw pitches; validates ascending + string count.
    pub fn new(name: impl Into<String>, strings: Vec<u8>) -> Result<Self> {
        let n = strings.len();
        if !(Self::MIN_STRINGS..=Self::MAX_STRINGS).contains(&n) {
            return Err(MtabError::InvalidTuning(format!(
                "{}: expected {}..={} strings, got {n}",
                name.into(),
                Self::MIN_STRINGS,
                Self::MAX_STRINGS
            )));
        }
        if !strings.windows(2).all(|w| w[0] < w[1]) {
            return Err(MtabError::InvalidTuning(format!(
                "{}: open-string pitches must be strictly ascending",
                name.into()
            )));
        }
        Ok(Self {
            strings,
            name: name.into(),
        })
    }

    pub fn string_count(&self) -> usize {
        self.strings.len()
    }

    /// MIDI pitch of an open string (0-based index, low string = 0).
    pub fn open_pitch(&self, string: usize) -> Option<u8> {
        self.strings.get(string).copied()
    }

    /// MIDI pitch of a fretted note; `None` if out of range for that string.
    pub fn pitch_of(&self, string: usize, fret: u8) -> Option<u8> {
        self.open_pitch(string).map(|p| p.saturating_add(fret))
    }

    // ---- The template library (§6.5) ----

    /// EADGBE — E2 A2 D3 G3 B3 E4.
    pub fn standard() -> Self {
        Self::new("standard", vec![40, 45, 50, 55, 59, 64]).unwrap()
    }
    /// DADGBE — drop D.
    pub fn drop_d() -> Self {
        Self::new("drop-d", vec![38, 45, 50, 55, 59, 64]).unwrap()
    }
    /// CGCFAD — drop C.
    pub fn drop_c() -> Self {
        Self::new("drop-c", vec![36, 43, 48, 53, 57, 62]).unwrap()
    }
    /// DADGAD.
    pub fn dadgad() -> Self {
        Self::new("dadgad", vec![38, 45, 50, 55, 57, 62]).unwrap()
    }
    /// DGDGBD — open G.
    pub fn open_g() -> Self {
        Self::new("open-g", vec![38, 43, 50, 55, 59, 62]).unwrap()
    }
    /// DADFAD — open D.
    pub fn open_d() -> Self {
        Self::new("open-d", vec![38, 45, 50, 54, 57, 62]).unwrap()
    }
    /// EBEG#BE — open E.
    pub fn open_e() -> Self {
        Self::new("open-e", vec![40, 47, 52, 56, 59, 64]).unwrap()
    }
    /// Eb Ab Db Gb Bb eb — half step down.
    pub fn half_step_down() -> Self {
        Self::new("half-step-down", vec![39, 44, 49, 54, 58, 63]).unwrap()
    }
    /// BEADGBE — 7-string standard (B).
    pub fn seven_string() -> Self {
        Self::new("7-string", vec![35, 40, 45, 50, 55, 59, 64]).unwrap()
    }
    /// F#BEADGBE — 8-string standard (F#).
    pub fn eight_string() -> Self {
        Self::new("8-string", vec![30, 35, 40, 45, 50, 55, 59, 64]).unwrap()
    }

    /// All templates, in display order.
    pub fn library() -> Vec<Tuning> {
        vec![
            Self::standard(),
            Self::half_step_down(),
            Self::drop_d(),
            Self::drop_c(),
            Self::dadgad(),
            Self::open_g(),
            Self::open_d(),
            Self::open_e(),
            Self::seven_string(),
            Self::eight_string(),
        ]
    }

    /// Look up a template by name or alias (case-insensitive; `-`/`_` equal).
    pub fn from_name(name: &str) -> Result<Self> {
        let norm = name.to_ascii_lowercase().replace('_', "-");
        let match_name = |n: &str| norm == n;
        let t = if match_name("standard") || match_name("e-std") || match_name("eadgbe") {
            Self::standard()
        } else if match_name("half-step-down") || match_name("eb-std") || match_name("half-step") {
            Self::half_step_down()
        } else if match_name("drop-d") || match_name("dropd") {
            Self::drop_d()
        } else if match_name("drop-c") || match_name("dropc") {
            Self::drop_c()
        } else if match_name("dadgad") {
            Self::dadgad()
        } else if match_name("open-g") {
            Self::open_g()
        } else if match_name("open-d") {
            Self::open_d()
        } else if match_name("open-e") {
            Self::open_e()
        } else if match_name("7-string") || match_name("7-string-standard") {
            Self::seven_string()
        } else if match_name("8-string") || match_name("8-string-standard") {
            Self::eight_string()
        } else {
            return Err(MtabError::InvalidTuning(name.to_owned()));
        };
        Ok(t)
    }
}

impl fmt::Display for Tuning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [", self.name)?;
        for (i, p) in self.strings.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{p}")?;
        }
        write!(f, "]")
    }
}

impl FromStr for Tuning {
    type Err = MtabError;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_name(s)
    }
}

/// Capo position 0–12 (0 = none).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Capo(u8);

impl<'de> Deserialize<'de> for Capo {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let position = u8::deserialize(deserializer)?;
        Self::new(position).map_err(serde::de::Error::custom)
    }
}

impl Capo {
    pub const NONE: Capo = Capo(0);

    pub fn new(pos: u8) -> Result<Self> {
        if pos <= 12 {
            Ok(Self(pos))
        } else {
            Err(MtabError::InvalidCapo(pos))
        }
    }

    pub fn position(self) -> u8 {
        self.0
    }
}

impl Default for Capo {
    fn default() -> Self {
        Self::NONE
    }
}

impl TryFrom<u8> for Capo {
    type Error = MtabError;

    fn try_from(value: u8) -> Result<Self> {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_tuning_pitches() {
        let t = Tuning::standard();
        assert_eq!(t.strings, vec![40, 45, 50, 55, 59, 64]);
        assert_eq!(t.pitch_of(0, 2), Some(42)); // low E + 2 frets = F#
        assert_eq!(t.pitch_of(5, 0), Some(64)); // high E open
    }

    #[test]
    fn library_names_round_trip() {
        for t in Tuning::library() {
            let looked = Tuning::from_name(&t.name).unwrap();
            assert_eq!(looked, t, "round-trip failed for {}", t.name);
        }
    }

    #[test]
    fn aliases_and_case_insensitivity() {
        assert_eq!(Tuning::from_name("EADGBE").unwrap(), Tuning::standard());
        assert_eq!(Tuning::from_name("Drop_D").unwrap(), Tuning::drop_d());
        assert_eq!(Tuning::from_name("dadgad").unwrap(), Tuning::dadgad());
        assert!(Tuning::from_name("banjo").is_err());
    }

    #[test]
    fn tuning_validation() {
        assert!(Tuning::new("bad", vec![40, 45, 44]).is_err()); // not ascending
        assert!(Tuning::new("bad", vec![40, 45, 50]).is_err()); // too few strings
        assert!(Tuning::new("ok7", vec![35, 40, 45, 50, 55, 59, 64]).is_ok());
    }

    #[test]
    fn capo_range() {
        assert_eq!(Capo::new(0).unwrap().position(), 0);
        assert_eq!(Capo::new(12).unwrap().position(), 12);
        assert!(Capo::new(13).is_err());
        assert_eq!(Capo::default().position(), 0);
    }
}
