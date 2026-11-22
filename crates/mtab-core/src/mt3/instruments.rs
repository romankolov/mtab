//! Instrument group vocabulary (upstream `tokenizer/mt3.py`,
//! `MT3_FULL_PLUS_GROUP_NAMES` + `get_group_program_map("MT3_FULL_PLUS")`).
//!
//! Group ids index the model's learned program-group conditioning and MUST
//! match the released checkpoints. The model emits the FIRST program of a
//! group (the representative), which decoding maps back to the group name.

/// Human-readable name → group id (verbatim upstream table).
pub const GROUP_NAMES: &[(&str, u16)] = &[
    ("acoustic_piano", 0),
    ("electric_piano", 1),
    ("chromatic_percussion", 2),
    ("organ", 3),
    ("acoustic_guitar", 4),
    ("clean_electric_guitar", 5),
    ("distorted_electric_guitar", 6),
    ("acoustic_bass", 7),
    ("electric_bass", 8),
    ("violin", 9),
    ("viola", 10),
    ("cello", 11),
    ("contrabass", 12),
    ("orchestral_harp", 13),
    ("timpani", 14),
    ("string_ensemble", 15),
    ("synth_strings", 16),
    ("voice", 17),
    ("orchestra_hit", 18),
    ("trumpet", 19),
    ("trombone", 20),
    ("tuba", 21),
    ("french_horn", 22),
    ("brass_section", 23),
    ("soprano_and_alto_sax", 24),
    ("tenor_sax", 25),
    ("baritone_sax", 26),
    ("oboe", 27),
    ("english_horn", 28),
    ("bassoon", 29),
    ("clarinet", 30),
    ("flutes", 31),
    ("synth_lead", 32),
    ("synth_pad", 33),
    ("drums", 36),
];

/// Group id → program list (upstream `get_group_program_map("MT3_FULL_PLUS")`).
/// The FIRST entry is the representative program the model emits.
pub const GROUP_PROGRAMS: &[(u16, &[u16])] = &[
    (0, &[0, 1, 3, 6, 7]),
    (1, &[2, 4, 5]),
    (2, &[8, 9, 10, 11, 12, 13, 14, 15]),
    (3, &[16, 17, 18, 19, 20, 21, 22, 23]),
    (4, &[24, 25]),
    (5, &[26, 27, 28]),
    (6, &[29, 30, 31]),
    (7, &[32, 35]),
    (8, &[33, 34, 36, 37, 38, 39]),
    (9, &[40]),
    (10, &[41]),
    (11, &[42]),
    (12, &[43]),
    (13, &[46]),
    (14, &[47]),
    (15, &[48, 49, 44, 45]),
    (16, &[50, 51]),
    (17, &[52, 53, 54]),
    (18, &[55]),
    (19, &[56, 59]),
    (20, &[57]),
    (21, &[58]),
    (22, &[60]),
    (23, &[61, 62, 63]),
    (24, &[64, 65]),
    (25, &[66]),
    (26, &[67]),
    (27, &[68]),
    (28, &[69]),
    (29, &[70]),
    (30, &[71]),
    (31, &[72, 73, 74, 75, 76, 77, 78, 79]),
    (32, &[80, 81, 82, 83, 84, 85, 86, 87]),
    (33, &[88, 89, 90, 91, 92, 93, 94, 95]),
    (34, &[100]),
    (35, &[101]),
];

/// The guitar family groups (product-level alias for `--instruments guitar`).
pub const GUITAR_FAMILY: &[&str] = &[
    "acoustic_guitar",
    "clean_electric_guitar",
    "distorted_electric_guitar",
];

/// Look up a group id by exact name.
pub fn group_id(name: &str) -> Option<u16> {
    GROUP_NAMES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, id)| *id)
}

/// Group name by id.
pub fn group_name(id: u16) -> Option<&'static str> {
    GROUP_NAMES
        .iter()
        .find(|(_, gid)| *gid == id)
        .map(|(n, _)| *n)
}

/// Representative (first) program of a group — what the model emits.
pub fn representative_program(group_id: u16) -> Option<u16> {
    GROUP_PROGRAMS
        .iter()
        .find(|(gid, _)| *gid == group_id)
        .and_then(|(_, progs)| progs.first().copied())
}

/// Program → group id (for decode-time instrument naming).
pub fn group_for_program(program: u16) -> Option<u16> {
    GROUP_PROGRAMS
        .iter()
        .find(|(_, progs)| progs.contains(&program))
        .map(|(gid, _)| *gid)
}

/// Instrument name for a decoded program (upstream
/// `_build_instrument_for_program`: representative programs map to group
/// names; anything else surfaces as `program_<n>`).
pub fn instrument_for_program(program: u16) -> String {
    if program == super::DRUM_PROGRAM {
        return "drums".to_owned();
    }
    match group_for_program(program) {
        Some(gid) if representative_program(gid) == Some(program) => {
            group_name(gid).unwrap_or("unknown").to_owned()
        }
        _ => format!("program_{program}"),
    }
}

/// Resolve loosely-typed instrument tokens to canonical group names
/// (upstream `resolve_instrument_names`, minus the difflib fuzzy tier):
/// case-insensitive exact match first, then unique substring ("timp" →
/// "timpani", "dist" → "distorted_electric_guitar"). The product alias
/// "guitar" expands to the three guitar groups.
pub fn resolve_instrument_names(tokens: &[String]) -> Result<Vec<String>, String> {
    let mut resolved = Vec::new();
    for raw in tokens {
        let t = raw.trim().to_ascii_lowercase();
        if t == "guitar" {
            resolved.extend(GUITAR_FAMILY.iter().map(|s| s.to_string()));
            continue;
        }
        if let Some((name, _)) = GROUP_NAMES.iter().find(|(n, _)| *n == t) {
            resolved.push(name.to_string());
            continue;
        }
        let hits: Vec<&str> = GROUP_NAMES
            .iter()
            .filter(|(n, _)| n.contains(&t))
            .map(|(n, _)| *n)
            .collect();
        match hits.len() {
            1 => resolved.push(hits[0].to_owned()),
            0 => {
                return Err(format!(
                    "unknown instrument name '{raw}'; valid names: {}",
                    GROUP_NAMES
                        .iter()
                        .map(|(n, _)| *n)
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            }
            _ => {
                return Err(format!(
                    "ambiguous instrument name '{raw}'; matches: {}",
                    hits.join(", ")
                ))
            }
        }
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guitar_groups_match_upstream() {
        assert_eq!(group_id("acoustic_guitar"), Some(4));
        assert_eq!(group_id("clean_electric_guitar"), Some(5));
        assert_eq!(group_id("distorted_electric_guitar"), Some(6));
        assert_eq!(representative_program(4), Some(24));
        assert_eq!(representative_program(5), Some(26));
        assert_eq!(representative_program(6), Some(29));
        assert_eq!(instrument_for_program(29), "distorted_electric_guitar");
        assert_eq!(instrument_for_program(30), "program_30");
        assert_eq!(instrument_for_program(128), "drums");
    }

    #[test]
    fn resolve_names() {
        assert_eq!(
            resolve_instrument_names(&["dist".into(), "Timp".into()]).unwrap(),
            vec!["distorted_electric_guitar", "timpani"]
        );
        assert_eq!(
            resolve_instrument_names(&["guitar".into()]).unwrap(),
            vec![
                "acoustic_guitar",
                "clean_electric_guitar",
                "distorted_electric_guitar"
            ]
        );
        assert!(resolve_instrument_names(&["banjo".into()]).is_err());
    }
}
