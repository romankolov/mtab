//! Opt-in parity check against an upstream reference generated from licensed
//! small-model weights. This test never downloads or embeds the checkpoint.

use mtab_core::lm::{GenOptions, LmConfig, LmModel, RowSpec};
use mtab_core::mt3::MT3Tokenizer;
use std::path::Path;

#[test]
#[ignore = "requires licensed small weights and separately generated upstream reference"]
fn real_small_model_matches_upstream() {
    let weights = std::env::var_os("MTAB_SMALL_WEIGHTS").expect("set MTAB_SMALL_WEIGHTS");
    let reference = std::env::var_os("MTAB_SMALL_REFERENCE").expect("set MTAB_SMALL_REFERENCE");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let model = LmModel::load(
        LmConfig::from_size(mtab_core::ModelSize::Small),
        Path::new(&weights),
    )
    .unwrap();
    let bytes = include_bytes!("fixtures/model/guitar_plucks.f32");
    let wav: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let (window, bank) = model
        .mel_bank()
        .expect("small checkpoint must contain mel buffers");
    let frontend = mtab_audio::mel::MelFrontend::with_bank(
        mtab_audio::mel::MelConfig::upstream(),
        window,
        bank,
    )
    .unwrap();
    let mel: Vec<f32> = frontend.compute_mel(&wav).into_iter().flatten().collect();
    let conds = model
        .encode_conditions(&[RowSpec {
            mel: Some(&mel),
            n_mel_frames: 501,
            valid_frames: 500,
            instrument: Some(&[4, 5, 6]),
            dataset: None,
        }])
        .unwrap();
    let names = [
        "acoustic_guitar".into(),
        "clean_electric_guitar".into(),
        "distorted_electric_guitar".into(),
    ];
    let forbidden: Vec<u32> = MT3Tokenizer::released()
        .forbidden_token_ids(&names)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    for name in ["greedy", "forced_prompt"] {
        let scenario = &golden["scenarios"][name];
        let ids = |key: &str| -> Vec<u32> {
            scenario[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect()
        };
        let prompt = ids("prompt");
        let expected = ids("tokens");
        let opts = GenOptions {
            max_gen_len: golden["max_gen_len"].as_u64().unwrap() as usize,
            forbidden: forbidden.clone(),
            ..GenOptions::default()
        };
        let output = model.generate(
            (!prompt.is_empty()).then_some(prompt.as_slice()),
            &conds,
            &opts,
        );
        assert_eq!(output.tokens, expected, "{name} token mismatch");
        let rows = scenario["logits"].as_array().unwrap();
        assert_eq!(output.step_logits.len(), rows.len());
        let mut max_abs = 0.0f64;
        let mut max_normalized = 0.0f64;
        for (step, (got, want)) in output.step_logits.iter().zip(rows).enumerate() {
            let want = want.as_array().unwrap();
            assert_eq!(got.len(), want.len());
            for (i, (&a, b)) in got.iter().zip(want).enumerate() {
                if b.is_null() {
                    assert_eq!(
                        a,
                        f64::NEG_INFINITY,
                        "{name} step {step} token {i} should be masked"
                    );
                    continue;
                }
                let b = b.as_f64().unwrap();
                let diff = (a - b).abs();
                max_abs = max_abs.max(diff);
                max_normalized = max_normalized.max(diff / b.abs().max(1.0));
                assert!(
                    diff <= 1e-3 * b.abs().max(1.0),
                    "{name} step {step} token {i}: got {a}, want {b}, abs diff {diff}"
                );
            }
        }
        eprintln!("{name}: {} tokens match, {} logit rows; max abs error {max_abs:.3e}, max scaled error {max_normalized:.3e}", output.tokens.len(), rows.len());
    }
}
