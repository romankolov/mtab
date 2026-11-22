//! `mtab-core` — core of **muscriptor-tab**: MuScriptor model configuration,
//! weights loading, inference, and MT3 decoding.
//!
//! Scaffold only — the full architecture lives in `PROMPT.md` (§3–§6).
//! Expand this crate per milestones M1–M8.

/// MuScriptor model variants.
///
/// Shapes match the official release table (PROMPT.md §2):
/// small = 103M / 14 layers / dim 768, medium = 307M / 24 / 1024,
/// large = 1.4B / 48 / 1536.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelSize {
    Small,
    Medium,
    Large,
}

impl ModelSize {
    /// `(approx_params, layers, dim)` from the official MuScriptor table.
    pub const fn shape(self) -> (u64, usize, usize) {
        match self {
            Self::Small => (103_000_000, 14, 768),
            Self::Medium => (307_000_000, 24, 1024),
            Self::Large => (1_400_000_000, 48, 1536),
        }
    }

    /// All variants, small → large.
    pub const ALL: [ModelSize; 3] = [Self::Small, Self::Medium, Self::Large];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_shapes_match_reference_table() {
        assert_eq!(ModelSize::Small.shape(), (103_000_000, 14, 768));
        assert_eq!(ModelSize::Medium.shape(), (307_000_000, 24, 1024));
        assert_eq!(ModelSize::Large.shape(), (1_400_000_000, 48, 1536));
    }
}
