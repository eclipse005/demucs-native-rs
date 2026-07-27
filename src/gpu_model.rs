//! GPU-side weight containers for HTDemucs v4.
//!
//! Mirrors `model.rs` (CPU-side, f32 storage) but holds `CudaSlice<f16>`
//! instead of `Vec<f32>`. The data is uploaded once at load time and stays
//! resident on the GPU for the steady-state forward pass.
//!
//! Each `GpuXxx` type has a `from_cpu(xxx, state)` constructor that copies
//! the corresponding CPU weight's data to the GPU (converting f32 → f16).
//!
//! Pipeline: at load time, `CpuEngine::load` builds the CPU weight tree;
//! `CudaEngine::load` mirrors it into a GPU tree via `from_cpu` calls.

use anyhow::Result;
use cudarc::driver::{safe::PushKernelArg, CudaSlice, DevicePtr, LaunchConfig};
use half::f16;
use std::sync::Arc;

use crate::cuda_engine::CudaState;
use crate::model::{
    Bias, Conv1dWeight, Conv2dWeight, CrossAttnLayer, CrossDomainTransformer, DConv,
    DConvLayer, FreqEmb, GroupNorm1, HDecLayer, HEncLayer, HTDemucs, LayerNorm1,
    LayerScale, MhaWeights, SelfAttnLayer, TDecLayer, TEncLayer, Weight2D,
};

// ═══════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════

/// Upload a f32 host slice to the device as f16.
fn upload_f32_as_f16(state: &CudaState, data: &[f32]) -> Result<CudaSlice<f16>> {
    let f16_data: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
    state
        .stream
        .clone_htod(&f16_data)
        .map_err(|e| anyhow::anyhow!("upload f32→f16: {e:?}"))
}

// ═══════════════════════════════════════════════════════════════════════
//  Weight containers
// ═══════════════════════════════════════════════════════════════════════

/// 2D weight matrix on the GPU, pre-transposed to `[in, out]` row-major
/// f16. `from_cpu` reorders the source `Weight2D::data` (which is
/// `[out, in]` PyTorch layout) into `[in, out]`. After upload, passing
/// the buffer to `gemm_f16(A[m, in], B[in, out], m, out, in)` directly
/// computes `A @ W`.
#[derive(Clone)]
pub struct GpuWeight2D {
    pub data: CudaSlice<f16>,
    pub rows: usize, // semantic = in_features (after transpose)
    pub cols: usize, // semantic = out_features
}

impl GpuWeight2D {
    pub fn from_cpu(state: &Arc<CudaState>, w: &Weight2D) -> Result<Self> {
        // Transpose on the host (small one-time cost at load).
        // Result is [in, out] row-major.
        let in_dim = w.cols;
        let out_dim = w.rows;
        let mut transposed = vec![0.0f32; in_dim * out_dim];
        for j in 0..out_dim {
            for i in 0..in_dim {
                transposed[i * out_dim + j] = w.data[j * in_dim + i];
            }
        }
        let data = upload_f32_as_f16(state, &transposed)?;
        Ok(Self {
            data,
            rows: in_dim,  // = in_dim (rows in transposed layout)
            cols: out_dim, // = out_dim (cols in transposed layout)
        })
    }
}

/// 4D conv weight on GPU: stored as f16 in original PyTorch layout
/// `[out_ch, in_ch, kH, kW]` (semantically `[out_ch, patch=in*kH*kW]`).
/// `gemm_f16` accepts this directly: it expects `B[k, n]` row-major where
/// `k = patch` and `n = out_ch`.
#[derive(Clone)]
pub struct GpuConv2dWeight {
    pub data: CudaSlice<f16>,
    pub out_ch: usize,
    pub in_ch: usize,
    pub kh: usize,
    pub kw: usize,
}

impl GpuConv2dWeight {
    pub fn from_cpu(state: &Arc<CudaState>, w: &Conv2dWeight) -> Result<Self> {
        // Host-side transpose from PyTorch [out_ch, in_ch*kH*kW] to
        // [in_ch*kH*kW, out_ch] row-major. The `gemm_f16` arg-swap trick
        // expects B in [k, n] row-major where k = patch, n = out_ch.
        // Without this transpose, GEMM would compute A @ W^T (col-major
        // interpretation) instead of A @ W^T (row-major), giving wrong
        // answers for any c_out != patch.
        let out_ch = w.out_ch;
        let patch = w.in_ch * w.kh * w.kw;
        let mut transposed = vec![0.0f32; patch * out_ch];
        for oc in 0..out_ch {
            for p in 0..patch {
                transposed[p * out_ch + oc] = w.data[oc * patch + p];
            }
        }
        let data = upload_f32_as_f16(state, &transposed)?;
        Ok(Self {
            data,
            out_ch: w.out_ch,
            in_ch: w.in_ch,
            kh: w.kh,
            kw: w.kw,
        })
    }
}

/// 1D conv weight on GPU: [out_ch, in_ch, k] row-major f16.
#[derive(Clone)]
pub struct GpuConv1dWeight {
    pub data: CudaSlice<f16>,
    pub out_ch: usize,
    pub in_ch: usize,
    pub k: usize,
}

impl GpuConv1dWeight {
    pub fn from_cpu(state: &Arc<CudaState>, w: &Conv1dWeight) -> Result<Self> {
        // Host-side transpose from PyTorch [out_ch, in_ch*k] to
        // [in_ch*k, out_ch] row-major. Same reason as Conv2dWeight.
        let out_ch = w.out_ch;
        let patch = w.in_ch * w.k;
        let mut transposed = vec![0.0f32; patch * out_ch];
        for oc in 0..out_ch {
            for p in 0..patch {
                transposed[p * out_ch + oc] = w.data[oc * patch + p];
            }
        }
        let data = upload_f32_as_f16(state, &transposed)?;
        Ok(Self {
            data,
            out_ch: w.out_ch,
            in_ch: w.in_ch,
            k: w.k,
        })
    }
}

/// ConvTranspose2d weight on GPU. The CPU-side `take_conv_transpose2d`
/// already reorders PyTorch [c_in, c_out, kH, kW] to `[patch, c_out]`
/// row-major (where patch = c_in*kH*kW), so we just upload as-is.
#[derive(Clone)]
pub struct GpuConvTranspose2dWeight {
    pub data: CudaSlice<f16>,
    pub out_ch: usize,
    pub in_ch: usize,
    pub kh: usize,
    pub kw: usize,
}

impl GpuConvTranspose2dWeight {
    pub fn from_cpu(state: &Arc<CudaState>, w: &Conv2dWeight) -> Result<Self> {
        let data = upload_f32_as_f16(state, &w.data)?;
        Ok(Self {
            data,
            out_ch: w.out_ch,
            in_ch: w.in_ch,
            kh: w.kh,
            kw: w.kw,
        })
    }
}

/// ConvTranspose1d weight on GPU (same load-time reorder as 2d).
#[derive(Clone)]
pub struct GpuConvTranspose1dWeight {
    pub data: CudaSlice<f16>,
    pub out_ch: usize,
    pub in_ch: usize,
    pub k: usize,
}

impl GpuConvTranspose1dWeight {
    pub fn from_cpu(state: &Arc<CudaState>, w: &Conv1dWeight) -> Result<Self> {
        let data = upload_f32_as_f16(state, &w.data)?;
        Ok(Self {
            data,
            out_ch: w.out_ch,
            in_ch: w.in_ch,
            k: w.k,
        })
    }
}

/// Bias vector on GPU.
#[derive(Clone)]
pub struct GpuBias {
    pub data: CudaSlice<f16>,
    pub len: usize,
}

impl GpuBias {
    pub fn from_cpu(state: &Arc<CudaState>, b: &Bias) -> Result<Self> {
        let data = upload_f32_as_f16(state, &b.data)?;
        Ok(Self {
            data,
            len: b.len,
        })
    }
}

/// LayerNorm1: gamma + beta on GPU.
#[derive(Clone)]
pub struct GpuLayerNorm1 {
    pub gamma: CudaSlice<f16>,
    pub beta: CudaSlice<f16>,
    pub dim: usize,
}

impl GpuLayerNorm1 {
    pub fn from_cpu(state: &Arc<CudaState>, ln: &LayerNorm1) -> Result<Self> {
        Ok(Self {
            gamma: upload_f32_as_f16(state, &ln.gamma)?,
            beta: upload_f32_as_f16(state, &ln.beta)?,
            dim: ln.dim,
        })
    }
}

/// GroupNorm1 (1 group, per-batch along C×L).
#[derive(Clone)]
pub struct GpuGroupNorm1 {
    pub gamma: CudaSlice<f16>,
    pub beta: CudaSlice<f16>,
    pub num_channels: usize,
}

impl GpuGroupNorm1 {
    pub fn from_cpu(state: &Arc<CudaState>, gn: &GroupNorm1) -> Result<Self> {
        Ok(Self {
            gamma: upload_f32_as_f16(state, &gn.gamma)?,
            beta: upload_f32_as_f16(state, &gn.beta)?,
            num_channels: gn.num_channels,
        })
    }
}

/// LayerScale on GPU.
#[derive(Clone)]
pub struct GpuLayerScale {
    pub scale: CudaSlice<f16>,
}

impl GpuLayerScale {
    pub fn from_cpu(state: &Arc<CudaState>, s: &LayerScale) -> Result<Self> {
        Ok(Self {
            scale: upload_f32_as_f16(state, &s.scale)?,
        })
    }
}

/// Multi-head attention weights on GPU. The packed in_proj_weight
/// `[3*d, d]` is split into three `[d, d]` Q/K/V weights, each
/// transposed to `[in=d, out=d]` for use with `linear_with_bias` /
/// `gemm_f16`. out_proj_weight is similarly transposed to `[d, d]`.
#[derive(Clone)]
pub struct GpuMhaWeights {
    pub q_w: GpuWeight2D, // [d, d] (in, out)
    pub k_w: GpuWeight2D,
    pub v_w: GpuWeight2D,
    pub q_b: GpuBias, // [d]
    pub k_b: GpuBias,
    pub v_b: GpuBias,
    pub out_proj_w: GpuWeight2D, // [d, d]
    pub out_proj_b: GpuBias, // [d]
    pub d_model: usize,
    pub n_heads: usize,
}

impl GpuMhaWeights {
    pub fn from_cpu(state: &Arc<CudaState>, m: &MhaWeights) -> Result<Self> {
        let d = m.d_model;
        // in_proj_weight is [3d, d] row-major. Slice into Q/K/V [d, d] each.
        let wq = &m.in_proj_weight[0..d * d];
        let wk = &m.in_proj_weight[d * d..2 * d * d];
        let wv = &m.in_proj_weight[2 * d * d..3 * d * d];
        // Build Weight2D in PyTorch [out, in] layout (rows=out=d, cols=in=d).
        // GpuWeight2D::from_cpu does the single transpose to [in, out] for us.
        let mk = |w: &[f32]| -> Weight2D {
            Weight2D {
                data: w.to_vec(),
                rows: d,
                cols: d,
            }
        };
        let q_w = GpuWeight2D::from_cpu(state, &mk(wq))?;
        let k_w = GpuWeight2D::from_cpu(state, &mk(wk))?;
        let v_w = GpuWeight2D::from_cpu(state, &mk(wv))?;
        // Biases: [d] each.
        let mk_b = |b: &[f32]| -> Bias {
            Bias {
                data: b.to_vec(),
                len: b.len(),
            }
        };
        let q_b = GpuBias::from_cpu(state, &mk_b(&m.in_proj_bias[0..d]))?;
        let k_b = GpuBias::from_cpu(state, &mk_b(&m.in_proj_bias[d..2 * d]))?;
        let v_b = GpuBias::from_cpu(state, &mk_b(&m.in_proj_bias[2 * d..3 * d]))?;
        // out_proj: [d, d] → [d, d] (in, out).
        let out_proj_w = GpuWeight2D::from_cpu(
            state,
            &mk(&m.out_proj_weight),
        )?;
        let out_proj_b = GpuBias::from_cpu(
            state,
            &Bias {
                data: m.out_proj_bias.clone(),
                len: m.out_proj_bias.len(),
            },
        )?;
        Ok(Self {
            q_w,
            k_w,
            v_w,
            q_b,
            k_b,
            v_b,
            out_proj_w,
            out_proj_b,
            d_model: m.d_model,
            n_heads: m.n_heads,
        })
    }
}

/// Self-attention layer weights on GPU.
#[derive(Clone)]
pub struct GpuSelfAttnLayer {
    pub norm1: GpuLayerNorm1,
    pub attn: GpuMhaWeights,
    pub gamma_1: GpuLayerScale,
    pub norm2: GpuLayerNorm1,
    pub linear1: GpuWeight2D,         // [ffn_dim, d_model]
    pub linear1_bias: GpuBias,       // [ffn_dim]
    pub linear2: GpuWeight2D,         // [d_model, ffn_dim]
    pub linear2_bias: GpuBias,       // [d_model]
    pub gamma_2: GpuLayerScale,
    pub norm_out: GpuGroupNorm1,
}

impl GpuSelfAttnLayer {
    pub fn from_cpu(state: &Arc<CudaState>, s: &SelfAttnLayer) -> Result<Self> {
        Ok(Self {
            norm1: GpuLayerNorm1::from_cpu(state, &s.norm1)?,
            attn: GpuMhaWeights::from_cpu(state, &s.attn)?,
            gamma_1: GpuLayerScale::from_cpu(state, &s.gamma_1)?,
            norm2: GpuLayerNorm1::from_cpu(state, &s.norm2)?,
            linear1: GpuWeight2D::from_cpu(state, &s.linear1)?,
            linear1_bias: GpuBias::from_cpu(state, &Bias {
                data: s.linear1_bias.clone(),
                len: s.linear1_bias.len(),
            })?,
            linear2: GpuWeight2D::from_cpu(state, &s.linear2)?,
            linear2_bias: GpuBias::from_cpu(state, &Bias {
                data: s.linear2_bias.clone(),
                len: s.linear2_bias.len(),
            })?,
            gamma_2: GpuLayerScale::from_cpu(state, &s.gamma_2)?,
            norm_out: GpuGroupNorm1::from_cpu(state, &s.norm_out)?,
        })
    }
}

/// Cross-attention layer weights on GPU.
#[derive(Clone)]
pub struct GpuCrossAttnLayer {
    pub norm1: GpuLayerNorm1,
    pub norm2: GpuLayerNorm1,
    pub attn: GpuMhaWeights,
    pub gamma_1: GpuLayerScale,
    pub norm3: GpuLayerNorm1,
    pub linear1: GpuWeight2D,
    pub linear1_bias: GpuBias,
    pub linear2: GpuWeight2D,
    pub linear2_bias: GpuBias,
    pub gamma_2: GpuLayerScale,
    pub norm_out: GpuGroupNorm1,
}

impl GpuCrossAttnLayer {
    pub fn from_cpu(state: &Arc<CudaState>, c: &CrossAttnLayer) -> Result<Self> {
        Ok(Self {
            norm1: GpuLayerNorm1::from_cpu(state, &c.norm1)?,
            norm2: GpuLayerNorm1::from_cpu(state, &c.norm2)?,
            attn: GpuMhaWeights::from_cpu(state, &c.attn)?,
            gamma_1: GpuLayerScale::from_cpu(state, &c.gamma_1)?,
            norm3: GpuLayerNorm1::from_cpu(state, &c.norm3)?,
            linear1: GpuWeight2D::from_cpu(state, &c.linear1)?,
            linear1_bias: GpuBias::from_cpu(state, &Bias {
                data: c.linear1_bias.clone(),
                len: c.linear1_bias.len(),
            })?,
            linear2: GpuWeight2D::from_cpu(state, &c.linear2)?,
            linear2_bias: GpuBias::from_cpu(state, &Bias {
                data: c.linear2_bias.clone(),
                len: c.linear2_bias.len(),
            })?,
            gamma_2: GpuLayerScale::from_cpu(state, &c.gamma_2)?,
            norm_out: GpuGroupNorm1::from_cpu(state, &c.norm_out)?,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  DConv + HEnc/TEnc/HDec/TDec layers
// ═══════════════════════════════════════════════════════════════════════

/// One DConv layer on GPU: conv1 → GN → GELU → conv2 → GN → GLU → LS, +residual.
#[derive(Clone)]
pub struct GpuDConvLayer {
    pub conv1: GpuConv1dWeight,
    pub conv1_bias: GpuBias,
    pub norm1: GpuGroupNorm1,
    pub conv2: GpuConv1dWeight,
    pub conv2_bias: GpuBias,
    pub norm2: GpuGroupNorm1,
    pub scale: GpuLayerScale,
}

impl GpuDConvLayer {
    pub fn from_cpu(state: &Arc<CudaState>, l: &DConvLayer) -> Result<Self> {
        Ok(Self {
            conv1: GpuConv1dWeight::from_cpu(state, &l.conv1)?,
            conv1_bias: GpuBias::from_cpu(state, &l.conv1_bias)?,
            norm1: GpuGroupNorm1::from_cpu(state, &l.norm1)?,
            conv2: GpuConv1dWeight::from_cpu(state, &l.conv2)?,
            conv2_bias: GpuBias::from_cpu(state, &l.conv2_bias)?,
            norm2: GpuGroupNorm1::from_cpu(state, &l.norm2)?,
            scale: GpuLayerScale::from_cpu(state, &l.scale)?,
        })
    }
}

/// 2-layer DConv on GPU.
#[derive(Clone)]
pub struct GpuDConv {
    pub layers: Vec<GpuDConvLayer>,
}

impl GpuDConv {
    pub fn from_cpu(state: &Arc<CudaState>, d: &DConv) -> Result<Self> {
        let mut layers = Vec::with_capacity(d.layers.len());
        for l in &d.layers {
            layers.push(GpuDConvLayer::from_cpu(state, l)?);
        }
        Ok(Self { layers })
    }
}

/// Frequency encoder layer on GPU.
#[derive(Clone)]
pub struct GpuHEncLayer {
    pub conv: GpuConv2dWeight,
    pub conv_bias: GpuBias,
    pub dconv: GpuDConv,
    pub rewrite: GpuConv2dWeight,
    pub rewrite_bias: GpuBias,
}

impl GpuHEncLayer {
    pub fn from_cpu(state: &Arc<CudaState>, l: &HEncLayer) -> Result<Self> {
        Ok(Self {
            conv: GpuConv2dWeight::from_cpu(state, &l.conv)?,
            conv_bias: GpuBias::from_cpu(state, &l.conv_bias)?,
            dconv: GpuDConv::from_cpu(state, &l.dconv)?,
            rewrite: GpuConv2dWeight::from_cpu(state, &l.rewrite)?,
            rewrite_bias: GpuBias::from_cpu(state, &l.rewrite_bias)?,
        })
    }
}

/// Time encoder layer on GPU.
#[derive(Clone)]
pub struct GpuTEncLayer {
    pub conv: GpuConv1dWeight,
    pub conv_bias: GpuBias,
    pub dconv: GpuDConv,
    pub rewrite: GpuConv1dWeight,
    pub rewrite_bias: GpuBias,
}

impl GpuTEncLayer {
    pub fn from_cpu(state: &Arc<CudaState>, l: &TEncLayer) -> Result<Self> {
        Ok(Self {
            conv: GpuConv1dWeight::from_cpu(state, &l.conv)?,
            conv_bias: GpuBias::from_cpu(state, &l.conv_bias)?,
            dconv: GpuDConv::from_cpu(state, &l.dconv)?,
            rewrite: GpuConv1dWeight::from_cpu(state, &l.rewrite)?,
            rewrite_bias: GpuBias::from_cpu(state, &l.rewrite_bias)?,
        })
    }
}

/// Frequency decoder layer on GPU.
#[derive(Clone)]
pub struct GpuHDecLayer {
    pub rewrite: GpuConv2dWeight,
    pub rewrite_bias: GpuBias,
    pub dconv: GpuDConv,
    pub conv_tr: GpuConvTranspose2dWeight,
    pub conv_tr_bias: GpuBias,
    pub last: bool,
}

impl GpuHDecLayer {
    pub fn from_cpu(state: &Arc<CudaState>, l: &HDecLayer) -> Result<Self> {
        Ok(Self {
            rewrite: GpuConv2dWeight::from_cpu(state, &l.rewrite)?,
            rewrite_bias: GpuBias::from_cpu(state, &l.rewrite_bias)?,
            dconv: GpuDConv::from_cpu(state, &l.dconv)?,
            conv_tr: GpuConvTranspose2dWeight::from_cpu(state, &l.conv_tr)?,
            conv_tr_bias: GpuBias::from_cpu(state, &l.conv_tr_bias)?,
            last: l.last,
        })
    }
}

/// Time decoder layer on GPU.
#[derive(Clone)]
pub struct GpuTDecLayer {
    pub rewrite: GpuConv1dWeight,
    pub rewrite_bias: GpuBias,
    pub dconv: GpuDConv,
    pub conv_tr: GpuConvTranspose1dWeight,
    pub conv_tr_bias: GpuBias,
    pub last: bool,
}

impl GpuTDecLayer {
    pub fn from_cpu(state: &Arc<CudaState>, l: &TDecLayer) -> Result<Self> {
        Ok(Self {
            rewrite: GpuConv1dWeight::from_cpu(state, &l.rewrite)?,
            rewrite_bias: GpuBias::from_cpu(state, &l.rewrite_bias)?,
            dconv: GpuDConv::from_cpu(state, &l.dconv)?,
            conv_tr: GpuConvTranspose1dWeight::from_cpu(state, &l.conv_tr)?,
            conv_tr_bias: GpuBias::from_cpu(state, &l.conv_tr_bias)?,
            last: l.last,
        })
    }
}

/// Freq embedding on GPU.
#[derive(Clone)]
pub struct GpuFreqEmb {
    pub data: CudaSlice<f16>,
    pub n_bins: usize,
    pub dim: usize,
}

impl GpuFreqEmb {
    pub fn from_cpu(state: &Arc<CudaState>, e: &FreqEmb) -> Result<Self> {
        Ok(Self {
            data: upload_f32_as_f16(state, &e.data)?,
            n_bins: e.n_bins,
            dim: e.dim,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Cross-domain transformer + HTDemucs top-level
// ═══════════════════════════════════════════════════════════════════════

/// One transformer layer on GPU (either self-attn or cross-attn).
#[derive(Clone)]
pub enum GpuTransformerLayerWeights {
    SelfAttn(GpuSelfAttnLayer),
    CrossAttn(GpuCrossAttnLayer),
}

#[derive(Clone)]
pub struct GpuCrossDomainTransformer {
    pub norm_in: GpuLayerNorm1,
    pub norm_in_t: GpuLayerNorm1,
    pub channel_upsampler: Option<GpuConv1dWeight>,
    pub channel_upsampler_bias: Option<GpuBias>,
    pub channel_downsampler: Option<GpuConv1dWeight>,
    pub channel_downsampler_bias: Option<GpuBias>,
    pub channel_upsampler_t: Option<GpuConv1dWeight>,
    pub channel_upsampler_t_bias: Option<GpuBias>,
    pub channel_downsampler_t: Option<GpuConv1dWeight>,
    pub channel_downsampler_t_bias: Option<GpuBias>,
    pub layers: Vec<GpuTransformerLayerWeights>,
    pub layers_t: Vec<GpuTransformerLayerWeights>,
}

impl GpuCrossDomainTransformer {
    pub fn from_cpu(state: &Arc<CudaState>, c: &CrossDomainTransformer) -> Result<Self> {
        let mut layers = Vec::with_capacity(c.layers.len());
        for l in &c.layers {
            layers.push(match l {
                crate::model::TransformerLayerWeights {
                    self_attn: Some(s),
                    ..
                } => GpuTransformerLayerWeights::SelfAttn(GpuSelfAttnLayer::from_cpu(
                    state, s,
                )?),
                crate::model::TransformerLayerWeights {
                    cross_attn: Some(c2),
                    ..
                } => GpuTransformerLayerWeights::CrossAttn(GpuCrossAttnLayer::from_cpu(
                    state, c2,
                )?),
                _ => anyhow::bail!("transformer layer has neither self_attn nor cross_attn"),
            });
        }
        let mut layers_t = Vec::with_capacity(c.layers_t.len());
        for l in &c.layers_t {
            layers_t.push(match l {
                crate::model::TransformerLayerWeights {
                    self_attn: Some(s),
                    ..
                } => GpuTransformerLayerWeights::SelfAttn(GpuSelfAttnLayer::from_cpu(
                    state, s,
                )?),
                crate::model::TransformerLayerWeights {
                    cross_attn: Some(c2),
                    ..
                } => GpuTransformerLayerWeights::CrossAttn(GpuCrossAttnLayer::from_cpu(
                    state, c2,
                )?),
                _ => anyhow::bail!("transformer layer has neither self_attn nor cross_attn"),
            });
        }
        let cu = c
            .channel_upsampler
            .as_ref()
            .map(|w| GpuConv1dWeight::from_cpu(state, w))
            .transpose()?;
        let cu_b = c
            .channel_upsampler_bias
            .as_ref()
            .map(|b| {
                GpuBias::from_cpu(
                    state,
                    &Bias {
                        data: b.clone(),
                        len: b.len(),
                    },
                )
            })
            .transpose()?;
        let cd = c
            .channel_downsampler
            .as_ref()
            .map(|w| GpuConv1dWeight::from_cpu(state, w))
            .transpose()?;
        let cd_b = c
            .channel_downsampler_bias
            .as_ref()
            .map(|b| {
                GpuBias::from_cpu(
                    state,
                    &Bias {
                        data: b.clone(),
                        len: b.len(),
                    },
                )
            })
            .transpose()?;
        let cu_t = c
            .channel_upsampler_t
            .as_ref()
            .map(|w| GpuConv1dWeight::from_cpu(state, w))
            .transpose()?;
        let cu_t_b = c
            .channel_upsampler_t_bias
            .as_ref()
            .map(|b| {
                GpuBias::from_cpu(
                    state,
                    &Bias {
                        data: b.clone(),
                        len: b.len(),
                    },
                )
            })
            .transpose()?;
        let cd_t = c
            .channel_downsampler_t
            .as_ref()
            .map(|w| GpuConv1dWeight::from_cpu(state, w))
            .transpose()?;
        let cd_t_b = c
            .channel_downsampler_t_bias
            .as_ref()
            .map(|b| {
                GpuBias::from_cpu(
                    state,
                    &Bias {
                        data: b.clone(),
                        len: b.len(),
                    },
                )
            })
            .transpose()?;
        Ok(Self {
            norm_in: GpuLayerNorm1::from_cpu(state, &c.norm_in)?,
            norm_in_t: GpuLayerNorm1::from_cpu(state, &c.norm_in_t)?,
            channel_upsampler: cu,
            channel_upsampler_bias: cu_b,
            channel_downsampler: cd,
            channel_downsampler_bias: cd_b,
            channel_upsampler_t: cu_t,
            channel_upsampler_t_bias: cu_t_b,
            channel_downsampler_t: cd_t,
            channel_downsampler_t_bias: cd_t_b,
            layers,
            layers_t,
        })
    }
}

/// Full HTDemucs model on GPU.
#[derive(Clone)]
pub struct GpuHTDemucs {
    pub encoders: Vec<GpuHEncLayer>,
    pub tencoders: Vec<GpuTEncLayer>,
    pub crosstransformer: GpuCrossDomainTransformer,
    pub decoders: Vec<GpuHDecLayer>,
    pub tdecoders: Vec<GpuTDecLayer>,
    pub freq_emb: GpuFreqEmb,
    pub n_sources: usize,
    pub bottom_channels: usize,
}

impl GpuHTDemucs {
    /// Mirror a CPU HTDemucs onto the GPU.
    pub fn from_cpu(state: &Arc<CudaState>, m: &HTDemucs) -> Result<Self> {
        let mut encoders = Vec::with_capacity(m.encoders.len());
        for e in &m.encoders {
            encoders.push(GpuHEncLayer::from_cpu(state, e)?);
        }
        let mut tencoders = Vec::with_capacity(m.tencoders.len());
        for t in &m.tencoders {
            tencoders.push(GpuTEncLayer::from_cpu(state, t)?);
        }
        let mut decoders = Vec::with_capacity(m.decoders.len());
        for d in &m.decoders {
            decoders.push(GpuHDecLayer::from_cpu(state, d)?);
        }
        let mut tdecoders = Vec::with_capacity(m.tdecoders.len());
        for t in &m.tdecoders {
            tdecoders.push(GpuTDecLayer::from_cpu(state, t)?);
        }
        Ok(Self {
            encoders,
            tencoders,
            crosstransformer: GpuCrossDomainTransformer::from_cpu(state, &m.crosstransformer)?,
            decoders,
            tdecoders,
            freq_emb: GpuFreqEmb::from_cpu(state, &m.freq_emb)?,
            n_sources: m.n_sources,
            bottom_channels: m.bottom_channels,
        })
    }
}
