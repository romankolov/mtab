//! LMModel port: decoder-only transformer + conditioners (PROMPT.md §6.3).
//!
//! Faithful port of upstream `lm.py` / `transformer.py` / `conditioners.py`
//! inference paths: ScaledEmbedding (row `card` = BOS, `-1` → zero), condition
//! prepending in sequence order `[self_wav, dataset_name, instrument_group]`
//! on the first step, pre-LN layers without biases, `F.gelu` exact-erf FFN,
//! fp32 sinusoidal positions added after the condition stream, bottom-right
//! causal attention over a preallocated KV cache, `out_norm` + untied head,
//! classifier-free guidance (`uncond + (cond - uncond)·coef`), the hardcoded
//! `logits[1393:] = -inf` reserved-token mask, and `generate()` with
//! prelude-prompt support, forbidden tokens, seeded sampling and beam search.
//!
//! Conditioner quirks reproduced for checkpoint parity: class ids land in
//! embedding row `id + 2` (tokenize `+1` then forward `+1`), null/pad entries
//! land in row 1, and the mel path is `log(mel + 1e-6)` → linear projection →
//! length mask.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{MtabError, Result};
use crate::model::ModelSize;
use crate::nn::{
    argsort_desc, gelu, layer_norm, linear, log_softmax, sin_embedding, softmax_temp, top_k_filter,
    top_p_filter, Rng,
};
use crate::safetensors::{load_weights, remap_single_codebook_keys, Tensor};

/// MT3 vocabulary size — upstream masks `logits[1393:]` unconditionally
/// (`lm.py::_compute_logits`), even on the 1395-card checkpoints.
pub const MT3_VOCAB: usize = 1393;

/// Transformer hyper-parameters for one checkpoint size.
#[derive(Debug, Clone, Copy)]
pub struct LmConfig {
    pub card: usize,
    pub dim: usize,
    pub heads: usize,
    pub layers: usize,
    pub max_period: f32,
}

impl LmConfig {
    /// Upstream `_CONFIGS`: small 768/12h/14L card 1393, medium 1024/16h/24L
    /// card 1395, large 1536/24h/48L card 1395; FFN = 4·dim.
    pub fn from_size(size: ModelSize) -> Self {
        let (card, dim, heads, layers) = match size {
            ModelSize::Small => (1393, 768, 12, 14),
            ModelSize::Medium => (1395, 1024, 16, 24),
            ModelSize::Large => (1395, 1536, 24, 48),
        };
        Self {
            card,
            dim,
            heads,
            layers,
            max_period: 10_000.0,
        }
    }

    /// Tiny config used by the committed parity fixtures.
    pub fn fixture_tiny() -> Self {
        Self {
            card: 1395,
            dim: 32,
            heads: 4,
            layers: 2,
            max_period: 10_000.0,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }

    pub fn ffn(&self) -> usize {
        4 * self.dim
    }
}

struct LayerW {
    norm1_w: Vec<f32>,
    norm1_b: Vec<f32>,
    qkv_w: Vec<f32>,
    out_w: Vec<f32>,
    norm2_w: Vec<f32>,
    norm2_b: Vec<f32>,
    lin1_w: Vec<f32>,
    lin2_w: Vec<f32>,
}

/// Loaded model weights (f32) with checkpoint key names mapped to fields.
pub struct LmModel {
    cfg: LmConfig,
    emb: Vec<f32>,
    layers: Vec<LayerW>,
    out_norm_w: Vec<f32>,
    out_norm_b: Vec<f32>,
    head: Vec<f32>,
    mel_proj_w: Vec<f32>,
    mel_proj_b: Vec<f32>,
    inst_emb: Vec<f32>,
    ds_emb: Vec<f32>,
    /// Checkpoint `mel_spec_transform.spectrogram.window` `[n_fft]`.
    mel_window: Option<Vec<f32>>,
    /// Checkpoint `mel_spec_transform.mel_scale.fb` flat `[n_freqs × n_mels]`.
    mel_fb: Option<Vec<f32>>,
}

fn take(tensors: &mut BTreeMap<String, Tensor>, key: &str, want: &[usize]) -> Result<Vec<f32>> {
    let t = tensors
        .remove(key)
        .ok_or_else(|| MtabError::ShapeMismatch {
            expected: format!("tensor `{key}`"),
            found: "missing from checkpoint".into(),
        })?;
    if t.shape != want {
        return Err(MtabError::ShapeMismatch {
            expected: format!("`{key}` {want:?}"),
            found: format!("{:?}", t.shape),
        });
    }
    Ok(t.data)
}

impl LmModel {
    /// Load a checkpoint (safetensors) for `cfg`, applying upstream's
    /// legacy-key remap and validating every expected shape.
    pub fn load(cfg: LmConfig, path: &Path) -> Result<Self> {
        let mut tensors = remap_single_codebook_keys(load_weights(path)?)?;
        let d = cfg.dim;
        let ffn = cfg.ffn();

        let emb = take(&mut tensors, "emb.weight", &[cfg.card + 1, d])?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("transformer.layers.{i}");
            layers.push(LayerW {
                norm1_w: take(&mut tensors, &format!("{p}.norm1.weight"), &[d])?,
                norm1_b: take(&mut tensors, &format!("{p}.norm1.bias"), &[d])?,
                qkv_w: take(
                    &mut tensors,
                    &format!("{p}.self_attn.in_proj_weight"),
                    &[3 * d, d],
                )?,
                out_w: take(
                    &mut tensors,
                    &format!("{p}.self_attn.out_proj.weight"),
                    &[d, d],
                )?,
                norm2_w: take(&mut tensors, &format!("{p}.norm2.weight"), &[d])?,
                norm2_b: take(&mut tensors, &format!("{p}.norm2.bias"), &[d])?,
                lin1_w: take(&mut tensors, &format!("{p}.linear1.weight"), &[ffn, d])?,
                lin2_w: take(&mut tensors, &format!("{p}.linear2.weight"), &[d, ffn])?,
            });
        }
        let out_norm_w = take(&mut tensors, "out_norm.weight", &[d])?;
        let out_norm_b = take(&mut tensors, "out_norm.bias", &[d])?;
        let head = take(&mut tensors, "linear.weight", &[cfg.card, d])?;
        let cond = "condition_provider.conditioners";
        let mel_proj_w = take(
            &mut tensors,
            &format!("{cond}.self_wav.output_proj.weight"),
            &[d, 512],
        )?;
        let mel_proj_b = take(
            &mut tensors,
            &format!("{cond}.self_wav.output_proj.bias"),
            &[d],
        )?;
        let inst_emb = take(
            &mut tensors,
            &format!("{cond}.instrument_group.embed.weight"),
            &[1001, d],
        )?;
        let ds_emb = take(
            &mut tensors,
            &format!("{cond}.dataset_name.embed.weight"),
            &[5, d],
        )?;
        // Optional buffers: released checkpoints carry them (f32-constructed);
        // re-packed layouts may drop them.
        let mel_window = tensors
            .remove(&format!(
                "{cond}.self_wav.mel_spec_transform.spectrogram.window"
            ))
            .map(|t| t.data);
        let mel_fb = tensors
            .remove(&format!("{cond}.self_wav.mel_spec_transform.mel_scale.fb"))
            .map(|t| t.data);
        if let Some(w) = &mel_window {
            if w.len() != 2048 {
                return Err(MtabError::ShapeMismatch {
                    expected: "`spectrogram.window` [2048]".into(),
                    found: format!("[{}]", w.len()),
                });
            }
        }
        if let Some(f) = &mel_fb {
            if f.len() != 1025 * 512 {
                return Err(MtabError::ShapeMismatch {
                    expected: "`mel_scale.fb` [1025, 512]".into(),
                    found: format!("[{}]", f.len()),
                });
            }
        }

        Ok(Self {
            cfg,
            emb,
            layers,
            out_norm_w,
            out_norm_b,
            head,
            mel_proj_w,
            mel_proj_b,
            inst_emb,
            ds_emb,
            mel_window,
            mel_fb,
        })
    }

    /// The checkpoint's mel filterbank (`window [2048]`, `fb [1025×512]`) when
    /// present. Feed these to `MelFrontend::with_bank` so the mel path uses the
    /// exact stored values.
    pub fn mel_bank(&self) -> Option<(&[f32], &[f32])> {
        match (&self.mel_window, &self.mel_fb) {
            (Some(w), Some(f)) => Some((w, f)),
            _ => None,
        }
    }

    pub fn config(&self) -> &LmConfig {
        &self.cfg
    }

    /// Encode one conditioning batch (upstream `ConditioningProvider` output
    /// order: `self_wav`, `instrument_group`, `dataset_name`).
    pub fn encode_conditions(&self, rows: &[RowSpec]) -> Result<Conditions> {
        let batch = rows.len();
        let t_mel = rows.first().map(|r| r.n_mel_frames).unwrap_or(0);
        let t_inst = rows
            .iter()
            .map(|r| r.instrument.map_or(1, |ids| ids.len()))
            .max()
            .unwrap_or(1);
        let t_ds = 1; // dataset_name is always null in the transcription path

        let mut mel = vec![0.0f64; batch * t_mel * self.cfg.dim];
        let mut inst = vec![0.0f64; batch * t_inst * self.cfg.dim];
        let mut ds = vec![0.0f64; batch * t_ds * self.cfg.dim];

        for (b, row) in rows.iter().enumerate() {
            // --- self_wav: log(mel + 1e-6) @ proj_wᵀ + bias, length-masked ---
            if let Some(raw) = row.mel {
                if raw.len() != t_mel * 512 {
                    return Err(MtabError::ShapeMismatch {
                        expected: format!("mel [{t_mel}, 512]"),
                        found: format!("{} frames", raw.len() / 512),
                    });
                }
                let logmel: Vec<f64> = raw.iter().map(|v| (*v as f64 + 1e-6).ln()).collect();
                let emb = linear(
                    &logmel,
                    &self.mel_proj_w,
                    Some(&self.mel_proj_b),
                    t_mel,
                    512,
                    self.cfg.dim,
                );
                let valid = row.valid_frames.min(t_mel);
                let dst = &mut mel[b * t_mel * self.cfg.dim..(b + 1) * t_mel * self.cfg.dim];
                dst[..valid * self.cfg.dim].copy_from_slice(&emb[..valid * self.cfg.dim]);
            }
            // --- class conditions: row = embed[id + 2], null/pad → row 1 ---
            fill_class_rows(
                &mut inst[b * t_inst * self.cfg.dim..(b + 1) * t_inst * self.cfg.dim],
                &self.inst_emb,
                row.instrument,
                t_inst,
                self.cfg.dim,
            );
            fill_class_rows(
                &mut ds[b * t_ds * self.cfg.dim..(b + 1) * t_ds * self.cfg.dim],
                &self.ds_emb,
                row.dataset,
                t_ds,
                self.cfg.dim,
            );
        }

        Ok(Conditions {
            batch,
            t_mel,
            t_inst,
            t_ds,
            mel,
            inst,
            ds,
        })
    }

    /// Allocate the incremental-decode state (KV caches + position offset).
    pub fn init_state(&self, batch: usize, capacity: usize) -> RunState {
        RunState {
            batch,
            offset: 0,
            layers: (0..self.cfg.layers)
                .map(|_| LayerState {
                    k: vec![0.0; batch * capacity * self.cfg.dim],
                    v: vec![0.0; batch * capacity * self.cfg.dim],
                })
                .collect(),
        }
    }

    /// Full forward: `tokens` is `batch × T` row-major; with `first` and
    /// `conds` the condition rows are prepended (upstream `first_step`).
    /// Returns `[batch, T, card]` logits for the token positions (condition
    /// outputs trimmed, as upstream).
    pub fn forward(
        &self,
        tokens: &[u32],
        batch: usize,
        conds: Option<&Conditions>,
        first: bool,
        state: &mut RunState,
    ) -> Vec<f64> {
        let t = tokens.len() / batch;
        let prep = if first {
            conds.map(|c| c.prep_len()).unwrap_or(0)
        } else {
            0
        };
        let tin = t + prep;
        assert_eq!(state.batch, batch);
        let d = self.cfg.dim;

        // --- embed: condition rows first, then token rows ---
        let mut x = vec![0.0f64; batch * tin * d];
        for b in 0..batch {
            if prep > 0 {
                let c = conds.expect("first_step without conditions");
                c.prepend_rows(b, &mut x[b * tin * d..b * tin * d + prep * d]);
            }
            for s in 0..t {
                let id = tokens[b * t + s] as usize;
                let row = if id == usize::MAX {
                    0
                } else {
                    id.min(self.cfg.card)
                };
                for (o, w) in x[(b * tin + prep + s) * d..(b * tin + prep + s + 1) * d]
                    .iter_mut()
                    .zip(&self.emb[row * d..(row + 1) * d])
                {
                    *o = *w as f64;
                }
            }
        }

        // --- fp32 sinusoidal positions (upstream adds them after the conds) ---
        for b in 0..batch {
            for i in 0..tin {
                let pos = (state.offset + i) as f32;
                let mut pe = vec![0.0f32; d];
                sin_embedding(pos, d, &mut pe);
                let dst = &mut x[(b * tin + i) * d..(b * tin + i + 1) * d];
                for j in 0..d {
                    dst[j] += pe[j] as f64;
                }
            }
        }

        // --- transformer layers ---
        let heads = self.cfg.heads;
        let dh = self.cfg.head_dim();
        for (li, lw) in self.layers.iter().enumerate() {
            let h = layer_norm(&x, &lw.norm1_w, &lw.norm1_b, d);
            let qkv = linear(&h, &lw.qkv_w, None, batch * tin, d, 3 * d);
            // write k/v into the cache and attend over the full prefix
            let mut attn = vec![0.0f64; batch * tin * d];
            let lst = &mut state.layers[li];
            let cap = lst.k.len() / batch / d;
            for b in 0..batch {
                for i in 0..tin {
                    let row = &qkv[(b * tin + i) * 3 * d..(b * tin + i + 1) * 3 * d];
                    let slot = (b * cap + state.offset + i) * d;
                    lst.k[slot..slot + d].copy_from_slice(&row[d..2 * d]);
                    lst.v[slot..slot + d].copy_from_slice(&row[2 * d..3 * d]);
                }
                for i in 0..tin {
                    let q = &qkv[(b * tin + i) * 3 * d..(b * tin + i) * 3 * d + d];
                    // Each head has its own scores, softmax, and value reduction.
                    let upto = state.offset + i + 1;
                    for hh in 0..heads {
                        let mut scores = vec![0.0f64; upto];
                        for (j, sc) in scores.iter_mut().enumerate() {
                            let k = &lst.k
                                [(b * cap + j) * d + hh * dh..(b * cap + j) * d + (hh + 1) * dh];
                            let qs = &q[hh * dh..(hh + 1) * dh];
                            let mut dot = 0.0f64;
                            for u in 0..dh {
                                dot += qs[u] * k[u];
                            }
                            *sc = dot / (dh as f64).sqrt();
                        }
                        let probs = softmax_temp(&scores, 1.0);
                        for (j, &p) in probs.iter().enumerate() {
                            let v = &lst.v
                                [(b * cap + j) * d + hh * dh..(b * cap + j) * d + (hh + 1) * dh];
                            let dst = &mut attn
                                [(b * tin + i) * d + hh * dh..(b * tin + i) * d + (hh + 1) * dh];
                            for u in 0..dh {
                                dst[u] += p * v[u];
                            }
                        }
                    }
                }
            }
            let proj = linear(&attn, &lw.out_w, None, batch * tin, d, d);
            for (xv, pv) in x.iter_mut().zip(&proj) {
                *xv += *pv;
            }
            let h2 = layer_norm(&x, &lw.norm2_w, &lw.norm2_b, d);
            let mut ff = linear(&h2, &lw.lin1_w, None, batch * tin, d, self.cfg.ffn());
            gelu(&mut ff);
            let ff2 = linear(&ff, &lw.lin2_w, None, batch * tin, self.cfg.ffn(), d);
            for (xv, pv) in x.iter_mut().zip(&ff2) {
                *xv += *pv;
            }
        }

        // --- out_norm + head on the token positions only ---
        let mut logits = vec![0.0f64; batch * t * self.cfg.card];
        let trimmed: Vec<f64> = (0..batch)
            .flat_map(|b| x[(b * tin + prep) * d..(b * tin + tin) * d].to_vec())
            .collect();
        let normed = layer_norm(&trimmed, &self.out_norm_w, &self.out_norm_b, d);
        let head_out = linear(&normed, &self.head, None, batch * t, d, self.cfg.card);
        logits.copy_from_slice(&head_out);
        logits
    }

    /// Upstream `_compute_logits`: CFG mixing, reserved-token mask, forbidden
    /// tokens, last timestep. `tokens` carries `eff_batch` rows; `conds` must
    /// carry `eff_batch` (cfg 1.0) or `2 × eff_batch` rows.
    #[allow(clippy::too_many_arguments)] // Mirrors the upstream inference entry point.
    pub fn compute_logits(
        &self,
        tokens: &[u32],
        eff_batch: usize,
        conds: &Conditions,
        first: bool,
        state: &mut RunState,
        cfg_coef: f32,
        forbidden: &[u32],
    ) -> Vec<f64> {
        let t = tokens.len() / eff_batch;
        let raw = if cfg_coef == 1.0 {
            self.forward(tokens, eff_batch, Some(conds), first, state)
        } else {
            let mut doubled = Vec::with_capacity(2 * eff_batch * t);
            doubled.extend_from_slice(tokens);
            doubled.extend_from_slice(tokens);
            let all = self.forward(&doubled, 2 * eff_batch, Some(conds), first, state);
            // split: rows [0..eff) = cond, [eff..2eff) = uncond
            let stride = t * self.cfg.card;
            let mut mixed = Vec::with_capacity(eff_batch * self.cfg.card);
            let cfg_coef = cfg_coef as f64;
            for b in 0..eff_batch {
                for k in 0..self.cfg.card {
                    let c = all[b * stride + (t - 1) * self.cfg.card + k];
                    let u = all[(eff_batch + b) * stride + (t - 1) * self.cfg.card + k];
                    mixed.push(u + (c - u) * cfg_coef);
                }
            }
            return self.mask_logits(mixed, forbidden);
        };
        // last timestep of each row
        let stride = t * self.cfg.card;
        let mut last = Vec::with_capacity(eff_batch * self.cfg.card);
        for b in 0..eff_batch {
            last.extend_from_slice(&raw[b * stride + (t - 1) * self.cfg.card..(b + 1) * stride]);
        }
        self.mask_logits(last, forbidden)
    }

    fn mask_logits(&self, mut logits: Vec<f64>, forbidden: &[u32]) -> Vec<f64> {
        let card = self.cfg.card;
        for row in logits.chunks_exact_mut(card) {
            for v in row[MT3_VOCAB.min(card)..].iter_mut() {
                *v = f64::NEG_INFINITY;
            }
            for &f in forbidden {
                if (f as usize) < card {
                    row[f as usize] = f64::NEG_INFINITY;
                }
            }
        }
        logits
    }

    /// Upstream `LMModel.generate` (batch of one sample): returns the emitted
    /// token stream (prompt first, then generated tokens, EOS included when
    /// hit) plus per-step masked logits for the generated steps (empty under
    /// beam search).
    pub fn generate(
        &self,
        prompt: Option<&[u32]>,
        conds: &Conditions,
        opts: &GenOptions,
    ) -> GenOutput {
        assert!(opts.beam_size >= 1);
        if opts.beam_size > 1 {
            assert!(opts.early_stop.is_some(), "beam search requires early_stop");
        }
        let mut rng = Rng::new(opts.seed);
        let prep = conds.prep_len();
        let eff = opts.beam_size;
        let cache_batch = eff * if opts.cfg_coef == 1.0 { 1 } else { 2 };
        let mut state = self.init_state(cache_batch, prep + opts.max_gen_len);

        // gen row-major [eff][max_gen_len+1], -2 = ungenerated
        let ungenerated = u32::MAX - 1;
        let mut gen = vec![ungenerated; eff * (opts.max_gen_len + 1)];
        for b in 0..eff {
            gen[b * (opts.max_gen_len + 1)] = self.cfg.card as u32; // BOS
        }
        let mut start_offset = 0;
        if let Some(p) = prompt {
            for b in 0..eff {
                gen[b * (opts.max_gen_len + 1) + 1..b * (opts.max_gen_len + 1) + 1 + p.len()]
                    .copy_from_slice(p);
            }
            start_offset = p.len();
        }

        let conds_eff = if eff == 1 {
            conds.clone_rows()
        } else {
            conds.repeat_beams(eff)
        };

        if opts.beam_size == 1 {
            return self.generate_streaming(
                &mut gen,
                &conds_eff,
                opts,
                &mut state,
                &mut rng,
                start_offset,
                ungenerated,
                prep,
            );
        }
        self.generate_beam(
            &mut gen,
            &conds_eff,
            opts,
            &mut state,
            start_offset,
            ungenerated,
            prep,
        )
    }

    #[allow(clippy::too_many_arguments)] // Generation state is passed explicitly.
    fn generate_streaming(
        &self,
        gen: &mut [u32],
        conds: &Conditions,
        opts: &GenOptions,
        state: &mut RunState,
        rng: &mut Rng,
        start_offset: usize,
        ungenerated: u32,
        prep: usize,
    ) -> GenOutput {
        let row_len = opts.max_gen_len + 1;
        let mut out_tokens = Vec::new();
        let mut step_logits = Vec::new();
        for t in 0..start_offset {
            out_tokens.push(gen[t + 1]);
        }
        for offset in start_offset..opts.max_gen_len {
            if let Some(eos) = opts.early_stop {
                if gen[..row_len].contains(&eos) {
                    break;
                }
            }
            let first = offset == start_offset;
            let input: Vec<u32> = if first {
                gen[..=offset].to_vec()
            } else {
                gen[offset..=offset].to_vec()
            };
            let logits = self.compute_logits(
                &input,
                1,
                conds,
                first,
                state,
                opts.cfg_coef,
                &opts.forbidden,
            );
            step_logits.push(logits.clone());
            let next = if opts.use_sampling && opts.temp > 0.0 {
                let probs = softmax_temp(&logits, opts.temp as f64);
                let filtered = if opts.top_p > 0.0 {
                    top_p_filter(&probs, opts.top_p as f64)
                } else if opts.top_k > 0 {
                    top_k_filter(&probs, opts.top_k)
                } else {
                    probs
                };
                rng.sample_categorical(&filtered) as u32
            } else {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap() as u32
            };
            let increment = input.len() + if first { prep } else { 0 };
            state.advance(increment);
            let this_step = gen[offset + 1];
            gen[offset + 1] = if this_step == ungenerated {
                next
            } else {
                this_step
            };
            out_tokens.push(gen[offset + 1]);
        }
        GenOutput {
            tokens: out_tokens,
            step_logits,
        }
    }

    #[allow(clippy::too_many_arguments)] // Generation state is passed explicitly.
    fn generate_beam(
        &self,
        gen: &mut [u32],
        conds: &Conditions,
        opts: &GenOptions,
        state: &mut RunState,
        start_offset: usize,
        ungenerated: u32,
        prep: usize,
    ) -> GenOutput {
        let beam = opts.beam_size;
        let row_len = opts.max_gen_len + 1;
        let eos = opts.early_stop.expect("checked");
        let alpha = 0.75f64;
        let mut beam_scores = vec![0.0f64; beam];
        let mut last_offset = start_offset.wrapping_sub(1);

        for offset in start_offset..opts.max_gen_len {
            last_offset = offset;
            let first = offset == start_offset;
            let t = if first { offset + 1 } else { 1 };
            let mut input = vec![0u32; beam * t];
            for b in 0..beam {
                input[b * t..(b + 1) * t]
                    .copy_from_slice(&gen[b * row_len + if first { 0 } else { offset }..][..t]);
            }
            let logits = self.compute_logits(
                &input,
                beam,
                conds,
                first,
                state,
                opts.cfg_coef,
                &opts.forbidden,
            );
            let increment = t + if first { prep } else { 0 };
            state.advance(increment);
            let card = self.cfg.card;
            let mut log_probs = Vec::with_capacity(beam * card);
            for b in 0..beam {
                log_probs.extend_from_slice(&log_softmax(&logits[b * card..(b + 1) * card]));
            }

            // top-beam candidates per current beam
            let mut topk_scores = vec![0.0f64; beam * beam];
            let mut topk_tokens = vec![0u32; beam * beam];
            for b in 0..beam {
                let order = argsort_desc(&log_probs[b * card..(b + 1) * card]);
                for (r, &ti) in order.iter().take(beam).enumerate() {
                    topk_scores[b * beam + r] = log_probs[b * card + ti];
                    topk_tokens[b * beam + r] = ti as u32;
                }
            }

            let mut beam_has_ended = vec![false; beam];
            let mut beam_lengths = vec![0f64; beam];
            for b in 0..beam {
                let row = &gen[b * row_len..(b + 1) * row_len];
                if let Some(pos) = row.iter().position(|&x| x == eos) {
                    beam_has_ended[b] = true;
                    beam_lengths[b] = (pos.max(1)) as f64;
                } else {
                    beam_lengths[b] = (offset + 1) as f64;
                }
            }
            for b in 0..beam {
                if beam_has_ended[b] {
                    for r in 0..beam {
                        topk_scores[b * beam + r] = 0.0;
                    }
                }
            }
            let lp: Vec<f64> = beam_lengths.iter().map(|l| 1.0 / l.powf(alpha)).collect();

            // candidates (beam × beam²), flattened as cand[b * beam + r]
            let mut cand = vec![0.0f64; beam * beam];
            for b in 0..beam {
                for r in 0..beam {
                    cand[b * beam + r] = (beam_scores[b] + topk_scores[b * beam + r]) * lp[b];
                }
            }
            let mut new_scores = vec![0.0f64; beam];
            let mut prev_local = vec![0usize; beam];
            let mut tok_rank = vec![0usize; beam];
            if offset == start_offset {
                // all beams identical: take the first beam's top-beam tokens
                for r in 0..beam {
                    new_scores[r] = cand[r];
                    prev_local[r] = 0;
                    tok_rank[r] = r;
                }
            } else {
                // global top-beam over beam × beam candidates
                let order = argsort_desc(&cand);
                for (r, &flat) in order.iter().take(beam).enumerate() {
                    new_scores[r] = cand[flat];
                    prev_local[r] = flat / beam;
                    tok_rank[r] = flat % beam;
                }
            }
            let next_tokens: Vec<u32> = (0..beam)
                .map(|r| topk_tokens[prev_local[r] * beam + tok_rank[r]])
                .collect();
            for b in 0..beam {
                beam_scores[b] = new_scores[b] / lp[prev_local[b]];
            }

            // reorder generation buffer and KV caches by prev_local
            let old_gen = gen.to_vec();
            for b in 0..beam {
                gen[b * row_len..(b + 1) * row_len].copy_from_slice(
                    &old_gen[prev_local[b] * row_len..(prev_local[b] + 1) * row_len],
                );
            }
            if prev_local.iter().enumerate().any(|(b, &p)| p != b) {
                for ls in &mut state.layers {
                    let cap_stride = ls.k.len() / state.batch;
                    let old_k = ls.k.clone();
                    let old_v = ls.v.clone();
                    let reorder: Vec<usize> = if state.batch == 2 * beam {
                        let mut r = prev_local.clone();
                        r.extend(prev_local.iter().map(|&p| p + beam));
                        r
                    } else {
                        prev_local.clone()
                    };
                    for (nb, &ob) in reorder.iter().enumerate() {
                        ls.k[nb * cap_stride..(nb + 1) * cap_stride]
                            .copy_from_slice(&old_k[ob * cap_stride..(ob + 1) * cap_stride]);
                        ls.v[nb * cap_stride..(nb + 1) * cap_stride]
                            .copy_from_slice(&old_v[ob * cap_stride..(ob + 1) * cap_stride]);
                    }
                }
            }

            for b in 0..beam {
                let this_step = gen[b * row_len + offset + 1];
                gen[b * row_len + offset + 1] = if this_step == ungenerated {
                    next_tokens[b]
                } else {
                    this_step
                };
            }
            if (0..beam).all(|b| gen[b * row_len..(b + 1) * row_len].contains(&eos)) {
                break;
            }
        }

        // best beam by unnormalized score
        let best = beam_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();
        let mut tokens = Vec::new();
        for t in 1..=last_offset + 1 {
            tokens.push(gen[best * row_len + t]);
        }
        GenOutput {
            tokens,
            step_logits: Vec::new(),
        }
    }
}

/// One batch row of conditioning (upstream `ConditioningAttributes`).
#[derive(Clone, Copy)]
pub struct RowSpec<'a> {
    /// Raw mel magnitudes `[n_mel_frames × 512]` (pre-log, `compute_mel`
    /// layout); `None` = nullified row (CFG unconditional side).
    pub mel: Option<&'a [f32]>,
    pub n_mel_frames: usize,
    /// Frames surviving the length mask (`length_samples / hop` upstream).
    pub valid_frames: usize,
    /// Instrument group ids (0-based, MT3_FULL_PLUS); `None` = null.
    pub instrument: Option<&'a [u16]>,
    /// Dataset class ids; `None` = null (always, in this pipeline).
    pub dataset: Option<&'a [u16]>,
}

/// Condition embeddings for a batch, in provider order.
#[derive(Clone)]
pub struct Conditions {
    pub batch: usize,
    pub t_mel: usize,
    pub t_inst: usize,
    pub t_ds: usize,
    pub mel: Vec<f64>,
    pub inst: Vec<f64>,
    pub ds: Vec<f64>,
}

impl Conditions {
    pub fn prep_len(&self) -> usize {
        self.t_mel + self.t_inst + self.t_ds
    }

    fn prepend_rows(&self, b: usize, out: &mut [f64]) {
        let d = self.mel.len() / self.batch / self.t_mel.max(1);
        let mut at = 0;
        // Upstream prepends via repeated `cat([cond, input_])` while iterating
        // the conditions, resulting in [self_wav, dataset_name, instrument_group].
        for (src, len) in [
            (&self.mel, self.t_mel * d),
            (&self.ds, self.t_ds * d),
            (&self.inst, self.t_inst * d),
        ] {
            out[at..at + len].copy_from_slice(&src[b * len..(b + 1) * len]);
            at += len;
        }
    }

    fn clone_rows(&self) -> Conditions {
        self.clone()
    }

    /// `torch.repeat_interleave(cond, beam, dim=0)`: each row's block repeats
    /// consecutively (`[r0×beam, r1×beam, …]`).
    fn repeat_beams(&self, beam: usize) -> Conditions {
        let mut out = Conditions {
            batch: self.batch * beam,
            t_mel: self.t_mel,
            t_inst: self.t_inst,
            t_ds: self.t_ds,
            mel: Vec::new(),
            inst: Vec::new(),
            ds: Vec::new(),
        };
        for (src, dst) in [
            (&self.mel, &mut out.mel),
            (&self.inst, &mut out.inst),
            (&self.ds, &mut out.ds),
        ] {
            let row = src.len() / self.batch;
            dst.reserve(src.len() * beam);
            for b in 0..self.batch {
                for _ in 0..beam {
                    dst.extend_from_slice(&src[b * row..(b + 1) * row]);
                }
            }
        }
        out
    }
}

fn fill_class_rows(dst: &mut [f64], table: &[f32], ids: Option<&[u16]>, t: usize, d: usize) {
    let rows = table.len() / d;
    for i in 0..t {
        let id = match ids {
            Some(v) if i < v.len() => v[i] as usize + 2, // upstream double +1
            _ => 1,                                      // null / pad
        };
        assert!(id < rows, "class id maps to row {id} ≥ {rows}");
        for (o, w) in dst[i * d..(i + 1) * d]
            .iter_mut()
            .zip(&table[id * d..(id + 1) * d])
        {
            *o = *w as f64;
        }
    }
}

/// Per-layer KV cache + global position offset.
pub struct LayerState {
    k: Vec<f64>,
    v: Vec<f64>,
}

pub struct RunState {
    batch: usize,
    offset: usize,
    layers: Vec<LayerState>,
}

impl RunState {
    /// `increment_steps` — advance the position offset after consuming rows.
    pub fn advance(&mut self, n: usize) {
        self.offset += n;
    }
}

/// Generation options (PROMPT.md §6.3: greedy default, `--temperature`,
/// `--beam-size`, `--seed`).
pub struct GenOptions {
    pub max_gen_len: usize,
    pub use_sampling: bool,
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub cfg_coef: f32,
    pub early_stop: Option<u32>,
    pub beam_size: usize,
    pub forbidden: Vec<u32>,
    pub seed: u64,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            max_gen_len: 2000,
            use_sampling: false,
            temp: 1.0,
            top_k: 0,
            top_p: 0.0,
            cfg_coef: 1.0,
            early_stop: Some(1),
            beam_size: 1,
            forbidden: Vec::new(),
            seed: 0,
        }
    }
}

/// Result of [`LmModel::generate`].
pub struct GenOutput {
    /// Prompt tokens first (when given), then generated tokens (EOS included).
    pub tokens: Vec<u32>,
    /// Masked last-step logits per generated step (beam search: empty).
    pub step_logits: Vec<Vec<f64>>,
}
