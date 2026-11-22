//! Model-backed transcription of already-ingested 16 kHz mono audio.

use mtab_core::chunk::{transcribe_chunks, ChunkResult};
use mtab_core::lm::{GenOptions, LmModel, RowSpec};
use mtab_core::mt3::{instruments, MT3Tokenizer};
use mtab_core::{MtabError, ProgressEvent, Result};

use crate::mel::{MelConfig, MelFrontend};
use crate::{AudioBuffer, MODEL_SAMPLE_RATE};

/// Run ordered chunk generation with upstream-compatible mel conditions and
/// forced tie preludes. `instrument_names` are canonical names, such as the
/// three names in `instruments::GUITAR_FAMILY`; an empty slice means no mask.
/// Progress is delivered synchronously. Missing EOS is returned in the result
/// unless `strict_eos` is enabled.
pub fn transcribe<P: FnMut(ProgressEvent)>(
    model: &LmModel,
    audio: &AudioBuffer,
    instrument_names: &[String],
    options: &GenOptions,
    strict_eos: bool,
    progress: P,
) -> Result<ChunkResult> {
    if audio.sample_rate != MODEL_SAMPLE_RATE || audio.channels != 1 {
        return Err(MtabError::UnsupportedAudio(
            "model input must be mono at 16000 Hz; use load_model_input".into(),
        ));
    }
    if options.max_gen_len == 0
        || options.beam_size == 0
        || options.early_stop != Some(1)
        || !options.cfg_coef.is_finite()
        || !options.temp.is_finite()
        || options.temp <= 0.0
        || (options.beam_size > 1 && options.use_sampling)
    {
        return Err(MtabError::DecodeFailure { chunk_index: 0, reason: "invalid generation settings (use positive length, beam width and temperature, EOS=1, and no sampling with beam search)".into() });
    }
    let names = instruments::resolve_instrument_names(instrument_names).map_err(|reason| {
        MtabError::DecodeFailure {
            chunk_index: 0,
            reason,
        }
    })?;
    let groups: Vec<u16> = names
        .iter()
        .map(|name| instruments::group_id(name).expect("resolved name"))
        .collect();
    let tokenizer = MT3Tokenizer::released();
    let mut forbidden = options.forbidden.clone();
    if !names.is_empty() {
        forbidden.extend(
            tokenizer
                .forbidden_token_ids(&names)
                .into_iter()
                .map(|id| id as u32),
        );
        forbidden.sort_unstable();
        forbidden.dedup();
    }
    let frontend = match model.mel_bank() {
        Some((window, bank)) => MelFrontend::with_bank(MelConfig::upstream(), window, bank)?,
        None => MelFrontend::upstream(),
    };
    transcribe_chunks(
        &audio.samples,
        strict_eos,
        |index, chunk, prompt| {
            if prompt.is_some_and(|p| p.len() >= options.max_gen_len) {
                return Err(MtabError::DecodeFailure {
                    chunk_index: index,
                    reason: "forced tie prologue exceeds generation limit".into(),
                });
            }
            let mel: Vec<f32> = frontend.compute_mel(chunk).into_iter().flatten().collect();
            let frames = mel.len() / 512;
            let conditioned = RowSpec {
                mel: Some(&mel),
                n_mel_frames: frames,
                valid_frames: 500,
                instrument: (!groups.is_empty()).then_some(groups.as_slice()),
                dataset: None,
            };
            let null = RowSpec {
                mel: None,
                n_mel_frames: frames,
                valid_frames: 0,
                instrument: None,
                dataset: None,
            };
            let rows = if options.cfg_coef == 1.0 {
                vec![conditioned]
            } else {
                vec![conditioned, null]
            };
            let conds = model.encode_conditions(&rows)?;
            let opts = GenOptions {
                max_gen_len: options.max_gen_len,
                use_sampling: options.use_sampling,
                temp: options.temp,
                top_k: options.top_k,
                top_p: options.top_p,
                cfg_coef: options.cfg_coef,
                early_stop: options.early_stop,
                beam_size: options.beam_size,
                forbidden: forbidden.clone(),
                seed: options.seed.wrapping_add(index as u64),
            };
            Ok(model.generate(prompt, &conds, &opts).tokens)
        },
        progress,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mtab_core::lm::LmConfig;
    use std::path::Path;

    #[test]
    fn synthetic_model_runs_audio_to_token_pipeline() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mtab-core/tests/fixtures/model/tiny.fixture");
        let model = LmModel::load(LmConfig::fixture_tiny(), &path).unwrap();
        let input = AudioBuffer {
            samples: vec![0.0; 1024],
            sample_rate: MODEL_SAMPLE_RATE,
            channels: 1,
        };
        let opts = GenOptions {
            max_gen_len: 2,
            ..GenOptions::default()
        };
        let mut progress = Vec::new();
        let result = transcribe(&model, &input, &["guitar".into()], &opts, false, |event| {
            progress.push(event)
        })
        .unwrap();
        assert_eq!(
            progress,
            vec![
                ProgressEvent {
                    completed: 0,
                    total: 1
                },
                ProgressEvent {
                    completed: 1,
                    total: 1
                }
            ]
        );
        assert!(result.missing_eos.len() <= 1);
    }
}
