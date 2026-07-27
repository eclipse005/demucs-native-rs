//! CPU-resident inference engine for HTDemucs v4.
//!
//! Pure Rust + gemm + rayon. Stores weights as f32; computes in f32.
//! Pipeline: STFT → CaC → freq+time encoders → cross-domain transformer
//! → freq+time decoders → iSTFT + time add → stems.

use std::borrow::Cow;

use crate::dsp::resample::resample_channel;
use crate::dsp::stft::Stft;
use crate::metadata::ModelInfo;
use crate::model::HTDemucs;
use crate::weights::WeightStore;
use crate::{LoadOptions, Stem, StemSelection, TRAINING_LENGTH};

/// One HTDemucs model (one signature). For htdemucs/htdemucs_6s this is the
/// single model; for htdemucs_ft there are 4 (one per stem).
pub struct CpuModel {
    pub sig: String,
    pub model: HTDemucs,
}

/// CPU engine: holds loaded model(s) + selected stems.
pub struct CpuEngine {
    info: &'static ModelInfo,
    models: Vec<CpuModel>,
    selected_stems: Vec<crate::metadata::StemId>,
    n_sources: usize,
    bottom_channels: usize,
}

impl CpuEngine {
    pub(crate) fn load(
        store: WeightStore,
        info: &'static ModelInfo,
        opts: &LoadOptions,
    ) -> anyhow::Result<Self> {
        let selected_stems: Vec<crate::metadata::StemId> = match &opts.stems {
            StemSelection::All => info.stems.to_vec(),
            StemSelection::Some(s) => s.clone(),
        };
        let n_sources = info.stems.len();
        let bottom_channels = if n_sources == 6 { 384 } else { 512 };

        let sigs_to_load: Vec<String> = if info.signatures.len() == 1 {
            vec![info.signatures[0].to_string()]
        } else {
            info.stems
                .iter()
                .enumerate()
                .filter(|(_, &s)| selected_stems.contains(&s))
                .map(|(i, _)| info.signatures[i].to_string())
                .collect()
        };

        let mut models = Vec::with_capacity(sigs_to_load.len());
        for sig in &sigs_to_load {
            anyhow::ensure!(
                store.signature(sig).is_some(),
                "signature {} not found in weight store",
                sig
            );
            let model = HTDemucs::from_store(&store, sig, n_sources, bottom_channels)?;
            models.push(CpuModel {
                sig: sig.clone(),
                model,
            });
        }
        log::info!(
            "CpuEngine: loaded {} model(s) for {}",
            models.len(),
            info.id
        );

        Ok(Self {
            info,
            models,
            selected_stems,
            n_sources,
            bottom_channels,
        })
    }

    /// Run source separation on stereo audio.
    pub fn separate(
        &self,
        left: &[f32],
        right: &[f32],
        sample_rate: u32,
    ) -> anyhow::Result<Vec<Stem>> {
        // ─── 0. Resample to 44100 Hz if needed ──────────────────────────
        let needs_resample = sample_rate != crate::SAMPLE_RATE as u32;
        let (left_in, right_in): (Cow<[f32]>, Cow<[f32]>) = if needs_resample {
            let l = resample_channel(left, sample_rate, crate::SAMPLE_RATE as u32)
                .map_err(|e| anyhow::anyhow!("resample: {e}"))?;
            let r = resample_channel(right, sample_rate, crate::SAMPLE_RATE as u32)
                .map_err(|e| anyhow::anyhow!("resample: {e}"))?;
            (Cow::Owned(l), Cow::Owned(r))
        } else {
            (Cow::Borrowed(left), Cow::Borrowed(right))
        };
        let left = &*left_in;
        let right = &*right_in;
        let n_samples = left.len();

        // ─── 1. Short audio fast path (≤ TRAINING_LENGTH) ───────────────
        let stems = if n_samples <= TRAINING_LENGTH {
            self.separate_single_segment(left, right, n_samples)?
        } else {
            // ─── 2. Chunked inference for long audio ─────────────────────
            let segment = TRAINING_LENGTH;
            let stride = segment * 3 / 4;
            let num_chunks = n_samples.saturating_sub(segment).div_ceil(stride) + 1;
            let n_stems = self.info.stems.len();

            let mut out_left = vec![vec![0.0f32; n_samples]; n_stems];
            let mut out_right = vec![vec![0.0f32; n_samples]; n_stems];
            let mut sum_weight = vec![0.0f32; n_samples];

            for chunk_idx in 0..num_chunks {
                let start = chunk_idx * stride;
                let end = (start + segment).min(n_samples);
                let chunk_len = end - start;

                let left_chunk = &left[start..end];
                let right_chunk = &right[start..end];

                let chunk_stems =
                    self.separate_single_segment(left_chunk, right_chunk, chunk_len)?;

                let window = triangular_window(chunk_len);
                for stem in chunk_stems.iter() {
                    let s = self
                        .info
                        .stems
                        .iter()
                        .position(|&id| id == stem.id)
                        .unwrap();
                    for i in 0..chunk_len {
                        let w = window[i];
                        out_left[s][start + i] += w * stem.left[i];
                        out_right[s][start + i] += w * stem.right[i];
                    }
                }
                for i in 0..chunk_len {
                    sum_weight[start + i] += window[i];
                }
            }

            // Normalize by accumulated weight.
            let mut stems = Vec::with_capacity(n_stems);
            for (s, &stem_id) in self.info.stems.iter().enumerate() {
                for i in 0..n_samples {
                    let w = sum_weight[i];
                    if w > 0.0 {
                        out_left[s][i] /= w;
                        out_right[s][i] /= w;
                    }
                }
                stems.push(Stem {
                    id: stem_id,
                    left: std::mem::take(&mut out_left[s]),
                    right: std::mem::take(&mut out_right[s]),
                });
            }
            stems
        };

        // Filter to requested stems (mirrors burn `-s`; htdemucs_ft emits all
        // n_sources stems per model, keep only selected).
        let stems: Vec<Stem> = stems
            .into_iter()
            .filter(|s| self.selected_stems.contains(&s.id))
            .collect();

        // ─── 3. Resample outputs back to original rate if needed ─────────
        if needs_resample {
            let mut resampled = Vec::with_capacity(stems.len());
            for mut stem in stems {
                stem.left = resample_channel(&stem.left, crate::SAMPLE_RATE as u32, sample_rate)
                    .map_err(|e| anyhow::anyhow!("resample: {e}"))?;
                stem.right = resample_channel(&stem.right, crate::SAMPLE_RATE as u32, sample_rate)
                    .map_err(|e| anyhow::anyhow!("resample: {e}"))?;
                resampled.push(stem);
            }
            Ok(resampled)
        } else {
            Ok(stems)
        }
    }

    /// Process a single segment (≤ TRAINING_LENGTH) through the full pipeline.
    fn separate_single_segment(
        &self,
        left: &[f32],
        right: &[f32],
        n_samples: usize,
    ) -> anyhow::Result<Vec<Stem>> {
        let padded_len = TRAINING_LENGTH;
        let mut left_padded = vec![0.0f32; padded_len];
        let mut right_padded = vec![0.0f32; padded_len];
        left_padded[..n_samples].copy_from_slice(left);
        right_padded[..n_samples].copy_from_slice(right);

        let mut stft = Stft::new(crate::N_FFT, crate::HOP_LENGTH);
        let left_spec = stft.forward(&left_padded)?;
        let right_spec = stft.forward(&right_padded)?;
        let n_bins = crate::N_FFT / 2;
        let n_frames = left_spec.len() / n_bins;

        // CaC: convert each [n_frames, n_bins] complex to [2, n_bins, n_frames] f32.
        let left_cac = crate::dsp::cac::stft_to_cac(&left_spec, crate::N_FFT);
        let right_cac = crate::dsp::cac::stft_to_cac(&right_spec, crate::N_FFT);
        // Stack left/right into a single [1, 4, n_bins, n_frames] freq tensor.
        let freq: Vec<f32> = build_freq_tensor(&left_cac, &right_cac, n_bins, n_frames);
        let time = build_time_tensor(&left_padded, &right_padded, padded_len);

        // Determine which models to run (htdemucs_ft has 4 separate per-stem models).
        let stems = if self.models.len() == 1 {
            // Single model: 4-stem htdemucs or 6-stem htdemucs_6s.
            let (freq_out, freq_shape, time_out, time_shape) = crate::ops_cpu::htdemucs_forward(
                &freq,
                [1, 4, n_bins, n_frames],
                &time,
                [1, 2, padded_len],
                &self.models[0].model,
            );
            // Trim freq to actual n_frames (n_frames = padded_len.div_ceil(HOP_LENGTH)).
            // n_frames from STFT may be < forward's T if input was padded with zeros that
            // round down. In practice n_frames matches.
            crate::ops_cpu::extract_stems(
                &freq_out,
                freq_shape,
                &time_out,
                time_shape,
                n_frames,
                padded_len,
                n_samples,
                &mut stft,
            )
        } else {
            // htdemucs_ft: one model per stem signature.
            let mut out_stems: Vec<Stem> = Vec::new();
            for (i, model) in self.models.iter().enumerate() {
                let stem_id = self.info.stems[i];
                let (freq_out, freq_shape, time_out, time_shape) = crate::ops_cpu::htdemucs_forward(
                    &freq,
                    [1, 4, n_bins, n_frames],
                    &time,
                    [1, 2, padded_len],
                    &model.model,
                );
                let stems = crate::ops_cpu::extract_stems(
                    &freq_out,
                    freq_shape,
                    &time_out,
                    time_shape,
                    n_frames,
                    padded_len,
                    n_samples,
                    &mut stft,
                );
                for s in stems {
                    if s.id == stem_id {
                        out_stems.push(s);
                        break;
                    }
                }
            }
            out_stems
        };

        Ok(stems)
    }
}

/// Build the freq tensor [1, 4, n_bins, n_frames] from left and right CaC
/// buffers. Layout: [left.re | left.im | right.re | right.im] stacked along
/// channel dim.
pub(crate) fn build_freq_tensor(left_cac: &[f32], right_cac: &[f32], n_bins: usize, n_frames: usize) -> Vec<f32> {
    // Source CaC layout: [2, n_bins, n_frames] = [re, im, re, im, ...]
    // We want [4, n_bins, n_frames] = [left.re, left.im, right.re, right.im]
    // i.e. interleave: for each (bin, frame), output 4 channels.
    // Output index: ((4 * bin) + 2) * n_frames + frame? No — row-major [4, n_bins, n_frames]
    // = out[c * n_bins * n_frames + bin * n_frames + frame].
    let mut out = vec![0.0f32; 4 * n_bins * n_frames];
    let bin_frames = n_bins * n_frames;
    for bin in 0..n_bins {
        for frame in 0..n_frames {
            let l_re = left_cac[0 * bin_frames + bin * n_frames + frame];
            let l_im = left_cac[1 * bin_frames + bin * n_frames + frame];
            let r_re = right_cac[0 * bin_frames + bin * n_frames + frame];
            let r_im = right_cac[1 * bin_frames + bin * n_frames + frame];
            out[0 * bin_frames + bin * n_frames + frame] = l_re;
            out[1 * bin_frames + bin * n_frames + frame] = l_im;
            out[2 * bin_frames + bin * n_frames + frame] = r_re;
            out[3 * bin_frames + bin * n_frames + frame] = r_im;
        }
    }
    out
}

/// Build the time tensor [1, 2, padded_len] from stereo audio.
pub(crate) fn build_time_tensor(left: &[f32], right: &[f32], padded_len: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; 2 * padded_len];
    data[..left.len()].copy_from_slice(left);
    data[padded_len..padded_len + right.len()].copy_from_slice(right);
    data
}

/// Triangular (Bartlett) window of length `n`. Ramps from 0 at edges to 1 at
/// the center.
pub(crate) fn triangular_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let denom = (n - 1) as f32;
    (0..n).map(|i| 1.0 - (2.0 * i as f32 / denom - 1.0).abs()).collect()
}

unsafe impl Send for CpuEngine {}
unsafe impl Sync for CpuEngine {}
