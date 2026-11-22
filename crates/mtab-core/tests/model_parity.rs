//! Golden parity tests for the LM port (PROMPT.md §6.3: hidden states and
//! logits within ≤ 1e-3 fp32 drift of upstream fixtures; token streams exact).
//!
//! Fixtures are generated from the VERBATIM upstream modules by
//! `tools/gen_model_fixtures.py` (see fixtures/model/manifest.json).

use std::path::{Path, PathBuf};

use mtab_core::lm::{GenOptions, LmConfig, LmModel, RowSpec, MT3_VOCAB};

fn fix(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/model")
        .join(name)
}

fn manifest() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fix("manifest.json")).unwrap()).unwrap()
}

/// Read a raw `<name>.f64` golden and return its elements.
fn load_f64(name: &str) -> Vec<f64> {
    let bytes = std::fs::read(fix(&format!("{name}.f64"))).unwrap();
    assert!(bytes.len() % 8 == 0, "{name}: truncated");
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn load_wav() -> Vec<f32> {
    let bytes = std::fs::read(fix("wav.f32")).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// |a−b| ≤ 1e-3·max(1, |b|) with matching -inf positions (§6.3 drift budget).
fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w.is_infinite() && w < 0.0 {
            assert!(
                g.is_infinite() && g < 0.0,
                "{what}[{i}]: want -inf, got {g}"
            );
            continue;
        }
        let tol = 1e-3 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g}, want {w} (tol {tol})"
        );
    }
}

fn seq_ids() -> Vec<u32> {
    manifest()["tensors"]["seq"]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

fn prompt_ids() -> Vec<u32> {
    manifest()["tensors"]["prompt"]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

fn forbidden() -> Vec<u32> {
    manifest()["forbidden"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

fn model() -> LmModel {
    LmModel::load(LmConfig::fixture_tiny(), &fix("tiny.fixture")).unwrap()
}

fn mel_frames(model: &LmModel) -> Vec<f32> {
    let (window, fb) = model
        .mel_bank()
        .expect("fixture checkpoint carries the mel bank");
    let fe =
        mtab_audio::mel::MelFrontend::with_bank(mtab_audio::mel::MelConfig::upstream(), window, fb)
            .unwrap();
    let wav = load_wav();
    let mel: Vec<f32> = fe.compute_mel(&wav).into_iter().flatten().collect();
    assert_eq!(mel.len(), 501 * 512, "mel element count");
    mel
}

/// CFG batch: row 0 = the real 5 s chunk conditioned on `4 5 6`; row 1 = the
/// nullified row (upstream `nullify_all_conditions`).
fn cfg_rows(mel: &[f32]) -> [RowSpec<'_>; 2] {
    [
        RowSpec {
            mel: Some(mel),
            n_mel_frames: 501,
            valid_frames: 500,
            instrument: Some(&[4u16, 5, 6]),
            dataset: None,
        },
        RowSpec {
            mel: None,
            n_mel_frames: 501,
            valid_frames: 0,
            instrument: None,
            dataset: None,
        },
    ]
}

#[test]
fn condition_rows_match_upstream() {
    let m = model();
    let mel = mel_frames(&m);
    let conds1 = m.encode_conditions(&[cfg_rows(&mel)[0]]).unwrap();
    let conds2 = m.encode_conditions(&cfg_rows(&mel)).unwrap();

    // instrument "4 5 6" → rows 6, 7, 8 (upstream double +1); null → row 1.
    assert_close(
        &conds1.inst,
        &load_f64("cond1_instrument_group"),
        "cond1_inst",
    );
    assert_close(&conds1.ds, &load_f64("cond1_dataset_name"), "cond1_ds");
    assert_close(&conds1.mel, &load_f64("cond1_self_wav"), "cond1_mel");
    assert_close(
        &conds2.inst,
        &load_f64("cond2_instrument_group"),
        "cond2_inst",
    );
    assert_close(&conds2.ds, &load_f64("cond2_dataset_name"), "cond2_ds");
    assert_close(&conds2.mel, &load_f64("cond2_self_wav"), "cond2_mel");
}

#[test]
fn full_forward_logits_match() {
    let m = model();
    let mel = mel_frames(&m);
    let conds1 = m.encode_conditions(&[cfg_rows(&mel)[0]]).unwrap();
    let seq = seq_ids();
    let mut state = m.init_state(1, conds1.prep_len() + seq.len());
    let logits = m.forward(&seq, 1, Some(&conds1), true, &mut state);
    assert_close(&logits, &load_f64("logits_full"), "logits_full");
}

#[test]
fn cfg_forward_mix_matches() {
    let m = model();
    let mel = mel_frames(&m);
    let conds2 = m.encode_conditions(&cfg_rows(&mel)).unwrap();
    let seq = seq_ids();
    let t = seq.len();

    // raw doubled forward
    let mut doubled = seq.clone();
    doubled.extend_from_slice(&seq);
    let mut state = m.init_state(2, conds2.prep_len() + t);
    let all = m.forward(&doubled, 2, Some(&conds2), true, &mut state);
    assert_close(&all, &load_f64("logits_cfg_all"), "logits_cfg_all");

    // compute_logits mixes uncond + (cond - uncond) * 1.5 at the last step
    let cfg_coef: f64 = manifest()["cfg_coef"].as_f64().unwrap();
    assert_eq!(cfg_coef, 1.5);
    let mut state = m.init_state(2, conds2.prep_len() + t);
    let mixed = m.compute_logits(&seq, 1, &conds2, true, &mut state, 1.5, &forbidden());
    let mut expected = load_f64("logits_cfg_mixed_last");
    for v in &mut expected[MT3_VOCAB..] {
        *v = f64::NEG_INFINITY;
    }
    for id in forbidden() {
        expected[id as usize] = f64::NEG_INFINITY;
    }
    assert_close(&mixed, &expected, "cfg_mixed");
}

#[test]
fn incremental_cache_matches_full_forward() {
    let m = model();
    let mel = mel_frames(&m);
    let conds1 = m.encode_conditions(&[cfg_rows(&mel)[0]]).unwrap();
    let seq = seq_ids();
    let t = seq.len();

    let mut full_state = m.init_state(1, conds1.prep_len() + t);
    let full = m.forward(&seq, 1, Some(&conds1), true, &mut full_state);

    let mut state = m.init_state(1, conds1.prep_len() + t);
    for i in 0..t {
        let first = i == 0;
        let input: Vec<u32> = if first {
            seq[..=i].to_vec()
        } else {
            vec![seq[i]]
        };
        let logits = m.forward(&input, 1, Some(&conds1), first, &mut state);
        state.advance(input.len() + if first { conds1.prep_len() } else { 0 });
        assert_close(
            &logits,
            &full[i * m.config().card..(i + 1) * m.config().card],
            &format!("step {i}"),
        );
    }
}

#[test]
fn greedy_stream_matches_upstream() {
    let m = model();
    let mel = mel_frames(&m);
    let conds1 = m.encode_conditions(&[cfg_rows(&mel)[0]]).unwrap();
    let opts = GenOptions {
        max_gen_len: 24,
        early_stop: Some(1),
        forbidden: forbidden(),
        ..GenOptions::default()
    };
    let out = m.generate(None, &conds1, &opts);

    let want: Vec<u32> = manifest()["tensors"]["inc_tokens"]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(out.tokens, want);
    for &f in &forbidden() {
        assert!(!out.tokens.contains(&f), "forbidden token {f} emitted");
    }

    let gold = load_f64("inc_step_logits");
    assert_eq!(gold.len(), out.step_logits.len() * 1395);
    for (step, row) in out.step_logits.iter().enumerate() {
        let want_row = &gold[step * 1395..(step + 1) * 1395];
        assert_close(row, want_row, &format!("inc step {step}"));
    }
}

#[test]
fn cfg_prompt_stream_matches_upstream() {
    let m = model();
    let mel = mel_frames(&m);
    let conds2 = m.encode_conditions(&cfg_rows(&mel)).unwrap();
    let cfg_coef = manifest()["cfg_coef"].as_f64().unwrap() as f32;
    let opts = GenOptions {
        max_gen_len: 24,
        early_stop: Some(1),
        cfg_coef,
        forbidden: forbidden(),
        ..GenOptions::default()
    };
    let out = m.generate(Some(&prompt_ids()), &conds2, &opts);

    let want: Vec<u32> = manifest()["tensors"]["inc_cfg_tokens"]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(out.tokens, want);

    let gold = load_f64("inc_cfg_step_logits");
    for (step, row) in out.step_logits.iter().enumerate() {
        assert_close(
            row,
            &gold[step * 1395..(step + 1) * 1395],
            &format!("cfg step {step}"),
        );
    }
}

#[test]
fn beam_search_matches_upstream() {
    let m = model();
    let mel = mel_frames(&m);
    let conds1 = m.encode_conditions(&[cfg_rows(&mel)[0]]).unwrap();
    let opts = GenOptions {
        max_gen_len: 24,
        early_stop: Some(1),
        beam_size: 2,
        forbidden: forbidden(),
        ..GenOptions::default()
    };
    let out = m.generate(None, &conds1, &opts);
    let want: Vec<u32> = manifest()["tensors"]["beam_tokens"]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(out.tokens, want);
}

#[test]
fn sampling_is_deterministic_per_seed() {
    let m = model();
    let mel = mel_frames(&m);
    let conds1 = m.encode_conditions(&[cfg_rows(&mel)[0]]).unwrap();
    let mk = |seed| GenOptions {
        max_gen_len: 24,
        early_stop: Some(1),
        use_sampling: true,
        temp: 0.9,
        top_k: 40,
        forbidden: forbidden(),
        seed,
        ..GenOptions::default()
    };
    let a = m.generate(None, &conds1, &mk(7));
    let b = m.generate(None, &conds1, &mk(7));
    assert_eq!(a.tokens, b.tokens);
    assert!(a.tokens.len() <= 25);
    for &f in &forbidden() {
        assert!(!a.tokens.contains(&f));
    }
}
