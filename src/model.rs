//! HTDemucs model layers — hand-written, no burn.
//!
//! Mirrors the structure of the burn-based `demucs-core::model::conv` but uses
//! plain `Vec<f32>` / `Vec<f16>` for weights and the `gemm` crate for matmuls.
//! Each layer's `forward` operates on flat f32 buffers with explicit shape
//! tracking, matching the burn version's numerics exactly for layer-by-layer
//! comparison.

#[allow(unused_imports)]
use half::f16;

use crate::weights::WeightStore;

// ─── Weight containers ───────────────────────────────────────────────────────

/// 2D weight matrix: [out_features, in_features] row-major (PyTorch layout).
pub struct Weight2D {
    pub data: Vec<f32>,
    pub rows: usize, // out_features
    pub cols: usize, // in_features
}

/// 4D conv weight: [out_ch, in_ch, kH, kW] row-major (PyTorch layout).
pub struct Conv2dWeight {
    pub data: Vec<f32>,
    pub out_ch: usize,
    pub in_ch: usize,
    pub kh: usize,
    pub kw: usize,
}

/// 1D conv weight: [out_ch, in_ch, k] row-major (PyTorch layout).
pub struct Conv1dWeight {
    pub data: Vec<f32>,
    pub out_ch: usize,
    pub in_ch: usize,
    pub k: usize,
}

/// Bias vector (optional — some convs/norms have none).
pub struct Bias {
    pub data: Vec<f32>,
    pub len: usize,
}

// ─── Transformer parameter containers ────────────────────────────────────────

/// Layer normalization: gamma + beta over the last dim.
pub struct LayerNorm1 {
    pub gamma: Vec<f32>, // [dim]
    pub beta: Vec<f32>,  // [dim]
    pub dim: usize,
}

/// Multi-head attention weights in **packed** PyTorch layout.
///
/// `in_proj_weight` is `[3*d_model, d_model]` (Q, K, V concatenated along dim 0).
/// `out_proj.weight` is `[d_model, d_model]`.
///
/// All stored as PyTorch layout — no transpose at load time. GEMM uses stride tricks.
pub struct MhaWeights {
    pub in_proj_weight: Vec<f32>, // [3*d_model, d_model]
    pub in_proj_bias: Vec<f32>,   // [3*d_model]
    pub out_proj_weight: Vec<f32>, // [d_model, d_model]
    pub out_proj_bias: Vec<f32>,  // [d_model]
    pub d_model: usize,
    pub n_heads: usize,
}

// ─── Transformer layers ─────────────────────────────────────────────────────

/// Self-attention block. Forward:
///   x = x + γ₁ · mha_self(norm1(x))
///   x = x + γ₂ · linear2(gelu(linear1(norm2(x))) + linear1_bias) + linear2_bias)
pub struct SelfAttnLayer {
    pub norm1: LayerNorm1,
    pub attn: MhaWeights,
    pub gamma_1: LayerScale,
    pub norm2: LayerNorm1,
    pub linear1: Weight2D,         // [ffn_dim, d_model]
    pub linear1_bias: Vec<f32>,    // [ffn_dim]
    pub linear2: Weight2D,         // [d_model, ffn_dim]
    pub linear2_bias: Vec<f32>,    // [d_model]
    pub gamma_2: LayerScale,
    /// MyGroupNorm(1, d_model) applied after the second residual add.
    /// Normalizes globally over (S, D) per batch — NOT per-position LayerNorm.
    pub norm_out: GroupNorm1,
}

/// Cross-attention block. Q comes from a separate query stream; K, V from
/// a cross (key/value) stream. Forward:
///   x = x + γ₁ · mha(norm1(x_q), norm2(x_kv))
///   x = x + γ₂ · linear2(gelu(linear1(norm3(x))) + linear1_bias) + linear2_bias)
///   x = norm_out(x)        // MyGroupNorm(1, d_model)
pub struct CrossAttnLayer {
    pub norm1: LayerNorm1,      // for Q
    pub norm2: LayerNorm1,      // for KV
    pub attn: MhaWeights,
    pub gamma_1: LayerScale,
    pub norm3: LayerNorm1,      // pre-FFN
    pub linear1: Weight2D,      // [ffn_dim, d_model]
    pub linear1_bias: Vec<f32>, // [ffn_dim]
    pub linear2: Weight2D,      // [d_model, ffn_dim]
    pub linear2_bias: Vec<f32>, // [d_model]
    pub gamma_2: LayerScale,
    /// MyGroupNorm(1, d_model) applied after the second residual add.
    pub norm_out: GroupNorm1,
}

/// One transformer layer — either self-attn or cross-attn (5 layers:
/// self, cross, self, cross, self).
pub struct TransformerLayerWeights {
    pub self_attn: Option<SelfAttnLayer>,
    pub cross_attn: Option<CrossAttnLayer>,
}

// ─── Cross-domain transformer ───────────────────────────────────────────────

/// Full cross-domain transformer at the bottleneck of HTDemucs.
///
/// For 4-stem / ft: `bottleneck_ch = 384`, `bottom_channels = 512`, so
/// channel up/down samplers are present (384 ↔ 512, Conv1d k=1).
/// For 6-stem: `bottleneck_ch == bottom_channels == 384`, no samplers.
///
/// 5 transformer layers each for freq (`layers`) and time (`layers_t`), with
/// pattern [self, cross, self, cross, self].
pub struct CrossDomainTransformer {
    pub norm_in: LayerNorm1,
    pub norm_in_t: LayerNorm1,
    /// Optional because 6-stem has no channel resampling.
    pub channel_upsampler: Option<Conv1dWeight>,    // Conv1d [d_model, bottleneck_ch, 1]
    pub channel_upsampler_bias: Option<Vec<f32>>,  // [d_model]
    pub channel_downsampler: Option<Conv1dWeight>, // Conv1d [bottleneck_ch, d_model, 1]
    pub channel_downsampler_bias: Option<Vec<f32>>,
    pub channel_upsampler_t: Option<Conv1dWeight>,
    pub channel_upsampler_t_bias: Option<Vec<f32>>,
    pub channel_downsampler_t: Option<Conv1dWeight>,
    pub channel_downsampler_t_bias: Option<Vec<f32>>,
    pub layers: Vec<TransformerLayerWeights>,
    pub layers_t: Vec<TransformerLayerWeights>,
}

// ─── Frequency / time decoder layers ─────────────────────────────────────────

/// Frequency decoder layer.
///
/// Forward:
///   x = x + skip
///   x = rewrite(x)            // Conv2d(3,3), padding (1,1) — chin → 2*chin
///   x = glu(x)                // GLU on dim=1: 2*chin → chin
///   x = dconv(x)              // per-frequency: [B*Fr, C, T] → [B*Fr, C, T]
///   x = conv_tr(x)            // ConvTranspose2d([8,1]) — chin → chout, 4× upsample
///   if x.fr > freq_target: x.fr = freq_target  // trim extra bins
///   if !last: gelu(x)
pub struct HDecLayer {
    pub rewrite: Conv2dWeight,      // [2*chin, chin, 3, 3]
    pub rewrite_bias: Bias,         // [2*chin]
    pub dconv: DConv,
    pub conv_tr: Conv2dWeight,      // [chin, chout, 8, 1]  PyTorch ConvTranspose layout
    pub conv_tr_bias: Bias,         // [chout]
    pub last: bool,
}

impl HDecLayer {
    pub fn from_store(
        store: &WeightStore,
        sig: &str,
        prefix: &str,
        _chin: usize,
        _chout: usize,
        last: bool,
    ) -> anyhow::Result<Self> {
        let rewrite = take_conv2d(store, sig, &format!("{}.rewrite", prefix))?;
        let rewrite_bias = take_bias(store, sig, &format!("{}.rewrite", prefix))?;
        let dconv = take_dconv(store, sig, &format!("{}.dconv.layers", prefix), _chin)?;
        let conv_tr = take_conv_transpose2d(store, sig, &format!("{}.conv_tr", prefix))?;
        let conv_tr_bias = take_bias(store, sig, &format!("{}.conv_tr", prefix))?;
        Ok(Self {
            rewrite,
            rewrite_bias,
            dconv,
            conv_tr,
            conv_tr_bias,
            last,
        })
    }
}

/// Time decoder layer — analogous to HDecLayer but for 1D.
///
/// Forward:
///   trim skip to match x.time
///   x = x + skip
///   x = rewrite(x)            // Conv1d(3), padding 1 — chin → 2*chin
///   x = glu(x)                // 2*chin → chin
///   x = dconv(x)              // 1D dconv, no flatten
///   x = conv_tr(x)            // ConvTranspose1d — chin → chout, 4× upsample
///   if x.time > time_target: trim
///   if !last: gelu(x)
pub struct TDecLayer {
    pub rewrite: Conv1dWeight,      // [2*chin, chin, 3]
    pub rewrite_bias: Bias,         // [2*chin]
    pub dconv: DConv,
    pub conv_tr: Conv1dWeight,      // [chin, chout, 8]  PyTorch ConvTranspose layout
    pub conv_tr_bias: Bias,         // [chout]
    pub last: bool,
}

impl TDecLayer {
    pub fn from_store(
        store: &WeightStore,
        sig: &str,
        prefix: &str,
        _chin: usize,
        _chout: usize,
        last: bool,
    ) -> anyhow::Result<Self> {
        let rewrite = take_conv1d(store, sig, &format!("{}.rewrite", prefix))?;
        let rewrite_bias = take_bias(store, sig, &format!("{}.rewrite", prefix))?;
        let dconv = take_dconv(store, sig, &format!("{}.dconv.layers", prefix), _chin)?;
        let conv_tr = take_conv_transpose1d(store, sig, &format!("{}.conv_tr", prefix))?;
        let conv_tr_bias = take_bias(store, sig, &format!("{}.conv_tr", prefix))?;
        Ok(Self {
            rewrite,
            rewrite_bias,
            dconv,
            conv_tr,
            conv_tr_bias,
            last,
        })
    }
}

/// Time encoder layer.
///
/// Forward:
///   pad input right so length is divisible by STRIDE (=4)
///   x = conv(x)               // Conv1d(k=8, stride=4, padding=2) — chin → chout
///   x = gelu(x)
///   x = dconv(x)              // 1D DConv (no flatten)
///   x = rewrite(x)            // Conv1d(k=1, chout → 2*chout)
///   x = glu(x)                // 2*chout → chout
pub struct TEncLayer {
    pub conv: Conv1dWeight,      // [chout, chin, 8]
    pub conv_bias: Bias,         // [chout]
    pub dconv: DConv,
    pub rewrite: Conv1dWeight,   // [2*chout, chout, 1]
    pub rewrite_bias: Bias,      // [2*chout]
}

impl TEncLayer {
    pub fn from_store(
        store: &WeightStore,
        sig: &str,
        prefix: &str,
        _chin: usize,
        _chout: usize,
    ) -> anyhow::Result<Self> {
        let conv = take_conv1d(store, sig, &format!("{}.conv", prefix))?;
        let conv_bias = take_bias(store, sig, &format!("{}.conv", prefix))?;
        let dconv = take_dconv(store, sig, &format!("{}.dconv.layers", prefix), _chout)?;
        let rewrite = take_conv1d(store, sig, &format!("{}.rewrite", prefix))?;
        let rewrite_bias = take_bias(store, sig, &format!("{}.rewrite", prefix))?;
        Ok(Self {
            conv,
            conv_bias,
            dconv,
            rewrite,
            rewrite_bias,
        })
    }
}

// ─── GroupNorm parameters ────────────────────────────────────────────────────

/// GroupNorm with 1 group (= LayerNorm over all channels): gamma + beta.
pub struct GroupNorm1 {
    pub gamma: Vec<f32>, // [num_channels]
    pub beta: Vec<f32>,  // [num_channels]
    pub num_channels: usize,
}

// ─── LayerScale ──────────────────────────────────────────────────────────────

/// Per-channel learnable scale ( initialised to ones).
pub struct LayerScale {
    pub scale: Vec<f32>, // [ch]
}

// ─── DConvLayer ──────────────────────────────────────────────────────────────

/// One DConv layer: conv1(k=3, dilated) → GroupNorm → GELU → conv2(k=1)
/// → GroupNorm → GLU → LayerScale, with a residual connection.
pub struct DConvLayer {
    pub conv1: Conv1dWeight,   // [compress, ch, 3]
    pub conv1_bias: Bias,      // [compress]
    pub norm1: GroupNorm1,     // [compress]
    pub conv2: Conv1dWeight,   // [2*ch, compress, 1]
    pub conv2_bias: Bias,      // [2*ch]
    pub norm2: GroupNorm1,     // [2*ch]
    pub scale: LayerScale,     // [ch]
}

/// DConv = 2 DConvLayers stacked (dilation 1, 2).
pub struct DConv {
    pub layers: Vec<DConvLayer>,
}

// ─── HEncLayer (frequency encoder) ───────────────────────────────────────────

/// Frequency encoder layer.
///
/// Forward:
///   x [B, C_in, Fr, T]
///   → Conv2d(k=[8,1], s=[4,1], p=[2,0])  → [B, C_out, Fr/4, T]
///   → GELU
///   → reshape [B*Fr, C_out, T]
///   → DConv
///   → reshape back [B, C_out, Fr, T]
///   → Conv2d(k=[1,1])  → [B, 2*C_out, Fr, T]
///   → GLU(dim=1)  → [B, C_out, Fr, T]
pub struct HEncLayer {
    pub conv: Conv2dWeight,      // [chout, chin, 8, 1]
    pub conv_bias: Bias,         // [chout]
    pub dconv: DConv,
    pub rewrite: Conv2dWeight,   // [2*chout, chout, 1, 1]
    pub rewrite_bias: Bias,      // [2*chout]
}

impl HEncLayer {
    /// Load an HEncLayer's weights from the store under the given signature.
    /// `prefix` is e.g. `encoder.0`.
    pub fn from_store(store: &WeightStore, sig: &str, prefix: &str) -> anyhow::Result<Self> {
        let conv = take_conv2d(store, sig, &format!("{}.conv", prefix))?;
        let conv_bias = take_bias(store, sig, &format!("{}.conv", prefix))?;
        let chout = conv.out_ch;

        let dconv = take_dconv(store, sig, &format!("{}.dconv.layers", prefix), chout)?;
        let rewrite = take_conv2d(store, sig, &format!("{}.rewrite", prefix))?;
        let rewrite_bias = take_bias(store, sig, &format!("{}.rewrite", prefix))?;

        Ok(Self {
            conv,
            conv_bias,
            dconv,
            rewrite,
            rewrite_bias,
        })
    }
}

// ─── Weight loading helpers ──────────────────────────────────────────────────

fn take_conv2d(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
) -> anyhow::Result<Conv2dWeight> {
    let t = store.take(sig, &format!("{}.weight", prefix))?;
    let shape = &t.shape;
    anyhow::ensure!(shape.len() == 4, "expected 4D conv weight, got {:?}", shape);
    let out_ch = shape[0];
    let in_ch = shape[1];
    let kh = shape[2];
    let kw = shape[3];
    Ok(Conv2dWeight {
        data: t.to_f32_vec(),
        out_ch,
        in_ch,
        kh,
        kw,
    })
}

fn take_conv1d(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
) -> anyhow::Result<Conv1dWeight> {
    let t = store.take(sig, &format!("{}.weight", prefix))?;
    let shape = &t.shape;
    anyhow::ensure!(shape.len() == 3, "expected 3D conv weight, got {:?}", shape);
    let out_ch = shape[0];
    let in_ch = shape[1];
    let k = shape[2];
    Ok(Conv1dWeight {
        data: t.to_f32_vec(),
        out_ch,
        in_ch,
        k,
    })
}

/// ConvTranspose-specific loader: PyTorch stores ConvTranspose weights as
/// `[in_channels, out_channels, kH, kW]` — the in/out dimensions are swapped
/// compared to regular Conv2d. We transpose them on load so that downstream
/// operators (which assume `[out, in, kH, kW]`) can treat it as a normal conv
/// weight, with the convention that the stored `out_ch` is what ConvTranspose
/// will actually produce (i.e. the weight's "out_channels" in PyTorch).
fn take_conv_transpose2d(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
) -> anyhow::Result<Conv2dWeight> {
    let t = store.take(sig, &format!("{}.weight", prefix))?;
    let shape = &t.shape;
    anyhow::ensure!(shape.len() == 4, "expected 4D conv_transpose weight, got {:?}", shape);
    // PyTorch stores ConvTranspose2d weights as [in, out, kH, kW] (in/out are
    // swapped compared to regular Conv2d). We reorder to a [patch, c_out]
    // row-major layout, where patch = in*kH*kW and `reordered[i, oc] =
    // a[ic, oc, kh, kw]` with i = ic*kH*kW + kh*kW + kw. With this layout,
    // the GEMM can use the same row-major rhs strides as `conv2d` (rhs_cs =
    // patch, rhs_rs = 1).
    let in_ch = shape[0];
    let out_ch = shape[1];
    let kh = shape[2];
    let kw = shape[3];
    let data = t.to_f32_vec();
    let patch = in_ch * kh * kw;
    let mut reordered = vec![0.0f32; patch * out_ch];
    for ic in 0..in_ch {
        for oc in 0..out_ch {
            for khh in 0..kh {
                for kww in 0..kw {
                    let src = ((ic * out_ch + oc) * kh + khh) * kw + kww;
                    let i = ic * kh * kw + khh * kw + kww;
                    let dst = i * out_ch + oc;
                    reordered[dst] = data[src];
                }
            }
        }
    }
    Ok(Conv2dWeight {
        data: reordered,
        out_ch,
        in_ch,
        kh,
        kw,
    })
}

/// ConvTranspose1d loader — same swap as `take_conv_transpose2d` for the 1D case.
fn take_conv_transpose1d(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
) -> anyhow::Result<Conv1dWeight> {
    let t = store.take(sig, &format!("{}.weight", prefix))?;
    let shape = &t.shape;
    anyhow::ensure!(
        shape.len() == 3,
        "expected 3D conv_transpose weight, got {:?}",
        shape
    );
    let in_ch = shape[0];
    let out_ch = shape[1];
    let k = shape[2];
    let data = t.to_f32_vec();
    // Reorder from PyTorch [in, out, k] row-major to [patch=in*k, c_out]
    // row-major: `reordered[i, oc] = a[ic, oc, dk]` with i = ic*k + dk. This
    // matches the [patch, c_out] row-major that the GEMM in conv_transpose1d
    // expects (rhs_rs=c_out, rhs_cs=1).
    let patch = in_ch * k;
    let mut reordered = vec![0.0f32; patch * out_ch];
    for ic in 0..in_ch {
        for oc in 0..out_ch {
            for kk in 0..k {
                let src = (ic * out_ch + oc) * k + kk;
                let i = ic * k + kk;
                let dst = i * out_ch + oc;
                reordered[dst] = data[src];
            }
        }
    }
    Ok(Conv1dWeight {
        data: reordered,
        out_ch,
        in_ch,
        k,
    })
}

fn take_bias(store: &WeightStore, sig: &str, prefix: &str) -> anyhow::Result<Bias> {
    let t = store.take(sig, &format!("{}.bias", prefix))?;
    let data = t.to_f32_vec();
    let len = data.len();
    Ok(Bias { data, len })
}

fn take_groupnorm(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
) -> anyhow::Result<GroupNorm1> {
    let w = store.take(sig, &format!("{}.weight", prefix))?;
    let b = store.take(sig, &format!("{}.bias", prefix))?;
    let gamma = w.to_f32_vec();
    let beta = b.to_f32_vec();
    let num_channels = gamma.len();
    Ok(GroupNorm1 {
        gamma,
        beta,
        num_channels,
    })
}

/// Same as `take_groupnorm` but asserts the channel count matches `expected`
/// (used by the transformer `norm_out` which must equal the d_model dim).
fn take_group_norm1(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
    expected: usize,
) -> anyhow::Result<GroupNorm1> {
    let gn = take_groupnorm(store, sig, prefix)?;
    anyhow::ensure!(
        gn.num_channels == expected,
        "{}.num_channels={} but expected {}",
        prefix,
        gn.num_channels,
        expected
    );
    Ok(gn)
}

fn take_layer_scale(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
) -> anyhow::Result<LayerScale> {
    let t = store.take(sig, &format!("{}.scale", prefix))?;
    Ok(LayerScale {
        scale: t.to_f32_vec(),
    })
}

fn take_dconv_layer(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
) -> anyhow::Result<DConvLayer> {
    // PyTorch layout: {prefix}.0 → conv1, .1 → norm1, .3 → conv2, .4 → norm2, .6 → scale
    let conv1 = take_conv1d(store, sig, &format!("{}.0", prefix))?;
    let conv1_bias = take_bias(store, sig, &format!("{}.0", prefix))?;
    let norm1 = take_groupnorm(store, sig, &format!("{}.1", prefix))?;
    let conv2 = take_conv1d(store, sig, &format!("{}.3", prefix))?;
    let conv2_bias = take_bias(store, sig, &format!("{}.3", prefix))?;
    let norm2 = take_groupnorm(store, sig, &format!("{}.4", prefix))?;
    let scale = take_layer_scale(store, sig, &format!("{}.6", prefix))?;
    Ok(DConvLayer {
        conv1,
        conv1_bias,
        norm1,
        conv2,
        conv2_bias,
        norm2,
        scale,
    })
}

fn take_dconv(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
    _ch: usize,
) -> anyhow::Result<DConv> {
    let dconv_depth = crate::DCONV_DEPTH;
    let mut layers = Vec::with_capacity(dconv_depth);
    for j in 0..dconv_depth {
        layers.push(take_dconv_layer(store, sig, &format!("{}.{}", prefix, j))?);
    }
    Ok(DConv { layers })
}

// Note: `f16` is imported at the top via `use half::f16;` and will be used
// once we switch weight storage from f32 to f16. For now weights are f32.

// ─── Frequency encoder (4 HEncLayers + freq_emb) ─────────────────────────────

/// Frequency embedding: [N_FFT/2, first_chout] lookup table.
///
/// In the PyTorch model this is a `ScaledEmbedding` with `scale=10`. The burn
/// loader bakes the `* 10.0` into the weights at load time (see burn
/// weights/load.rs:580-583), so we do the same here. The additional `* 0.2`
/// (`freq_emb_scale`) is applied at forward time, not load time.
pub struct FreqEmb {
    /// [n_fft/2, first_chout] row-major, raw weights (no scaling baked in).
    /// The forward applies `* 0.2` once, matching burn's
    /// `freq = freq + emb * 0.2` line.
    pub data: Vec<f32>,
    pub n_bins: usize,    // N_FFT/2 = 2048
    pub dim: usize,       // first_chout = 48
}

impl FreqEmb {
    pub fn from_store(store: &WeightStore, sig: &str, first_chout: usize) -> anyhow::Result<Self> {
        let t = store.take(sig, "freq_emb.embedding.weight")?;
        let shape = &t.shape;
        anyhow::ensure!(shape.len() == 2, "freq_emb shape {:?}", shape);
        let n_bins = shape[0];
        let dim = shape[1];
        anyhow::ensure!(dim == first_chout, "freq_emb dim {} != first_chout {}", dim, first_chout);
        Ok(Self {
            data: t.to_f32_vec(),
            n_bins,
            dim,
        })
    }
}

/// Complete frequency encoder: 4 HEncLayers + freq_emb.
///
/// Channel progression: CaC(4) → 48 → 96 → 192 → 384
pub struct FreqEncoder {
    pub layers: Vec<HEncLayer>,
    pub freq_emb: FreqEmb,
}

impl FreqEncoder {
    pub fn from_store(store: &WeightStore, sig: &str) -> anyhow::Result<Self> {
        let mut layers = Vec::with_capacity(4);
        for i in 0..4 {
            layers.push(HEncLayer::from_store(store, sig, &format!("encoder.{}", i))?);
        }
        let freq_emb = FreqEmb::from_store(store, sig, 48)?; // first_chout = CHANNELS = 48
        Ok(Self { layers, freq_emb })
    }
}

// ─── Transformer weight loaders ──────────────────────────────────────────────

fn take_layernorm(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
) -> anyhow::Result<LayerNorm1> {
    let w = store.take(sig, &format!("{}.weight", prefix))?;
    let b = store.take(sig, &format!("{}.bias", prefix))?;
    let gamma = w.to_f32_vec();
    let beta = b.to_f32_vec();
    let dim = gamma.len();
    anyhow::ensure!(beta.len() == dim, "layernorm dim mismatch ({} vs {})", beta.len(), dim);
    Ok(LayerNorm1 { gamma, beta, dim })
}

fn take_weight2d(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
) -> anyhow::Result<Weight2D> {
    let t = store.take(sig, &format!("{}.weight", prefix))?;
    let shape = &t.shape;
    anyhow::ensure!(shape.len() == 2, "expected 2D weight, got {:?}", shape);
    Ok(Weight2D {
        data: t.to_f32_vec(),
        rows: shape[0],
        cols: shape[1],
    })
}

fn take_linear_with_bias(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
) -> anyhow::Result<(Weight2D, Vec<f32>)> {
    let w = take_weight2d(store, sig, prefix)?;
    let b = store.take(sig, &format!("{}.bias", prefix))?;
    Ok((w, b.to_f32_vec()))
}

fn take_mha(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
    d_model: usize,
    n_heads: usize,
) -> anyhow::Result<MhaWeights> {
    let in_proj_w = store.take(sig, &format!("{}.in_proj_weight", prefix))?;
    let in_proj_b = store.take(sig, &format!("{}.in_proj_bias", prefix))?;
    let out_proj = take_linear_with_bias(store, sig, &format!("{}.out_proj", prefix))?;
    let in_proj_shape = &in_proj_w.shape;
    anyhow::ensure!(
        in_proj_shape.len() == 2
            && in_proj_shape[0] == 3 * d_model
            && in_proj_shape[1] == d_model,
        "in_proj shape {:?} mismatch (expected [3*{}, {}])",
        in_proj_shape,
        d_model,
        d_model
    );
    Ok(MhaWeights {
        in_proj_weight: in_proj_w.to_f32_vec(),
        in_proj_bias: in_proj_b.to_f32_vec(),
        out_proj_weight: out_proj.0.data,
        out_proj_bias: out_proj.1,
        d_model,
        n_heads,
    })
}

fn take_self_attn_layer(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
    d_model: usize,
    n_heads: usize,
) -> anyhow::Result<SelfAttnLayer> {
    let norm1 = take_layernorm(store, sig, &format!("{}.norm1", prefix))?;
    let attn = take_mha(store, sig, &format!("{}.self_attn", prefix), d_model, n_heads)?;
    let gamma_1 = take_layer_scale(store, sig, &format!("{}.gamma_1", prefix))?;
    let norm2 = take_layernorm(store, sig, &format!("{}.norm2", prefix))?;
    let (linear1, linear1_bias) = take_linear_with_bias(store, sig, &format!("{}.linear1", prefix))?;
    let (linear2, linear2_bias) = take_linear_with_bias(store, sig, &format!("{}.linear2", prefix))?;
    let gamma_2 = take_layer_scale(store, sig, &format!("{}.gamma_2", prefix))?;
    let norm_out = take_group_norm1(store, sig, &format!("{}.norm_out", prefix), d_model)?;
    Ok(SelfAttnLayer {
        norm1,
        attn,
        gamma_1,
        norm2,
        linear1,
        linear1_bias,
        linear2,
        linear2_bias,
        gamma_2,
        norm_out,
    })
}

fn take_cross_attn_layer(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
    d_model: usize,
    n_heads: usize,
) -> anyhow::Result<CrossAttnLayer> {
    let norm1 = take_layernorm(store, sig, &format!("{}.norm1", prefix))?;
    let norm2 = take_layernorm(store, sig, &format!("{}.norm2", prefix))?;
    let attn = take_mha(store, sig, &format!("{}.cross_attn", prefix), d_model, n_heads)?;
    let gamma_1 = take_layer_scale(store, sig, &format!("{}.gamma_1", prefix))?;
    let norm3 = take_layernorm(store, sig, &format!("{}.norm3", prefix))?;
    let (linear1, linear1_bias) = take_linear_with_bias(store, sig, &format!("{}.linear1", prefix))?;
    let (linear2, linear2_bias) = take_linear_with_bias(store, sig, &format!("{}.linear2", prefix))?;
    let gamma_2 = take_layer_scale(store, sig, &format!("{}.gamma_2", prefix))?;
    let norm_out = take_group_norm1(store, sig, &format!("{}.norm_out", prefix), d_model)?;
    Ok(CrossAttnLayer {
        norm1,
        norm2,
        attn,
        gamma_1,
        norm3,
        linear1,
        linear1_bias,
        linear2,
        linear2_bias,
        gamma_2,
        norm_out,
    })
}

fn take_transformer_layer(
    store: &WeightStore,
    sig: &str,
    prefix: &str,
    d_model: usize,
    n_heads: usize,
) -> anyhow::Result<TransformerLayerWeights> {
    // Determine self vs cross by probing the key for the sub-attention.
    // Self layers (0, 2, 4) have `prefix.self_attn.in_proj_weight`.
    // Cross layers (1, 3) have `prefix.cross_attn.in_proj_weight`.
    let self_key = format!("{}.self_attn.in_proj_weight", prefix);
    let cross_key = format!("{}.cross_attn.in_proj_weight", prefix);
    if store.try_take(sig, &self_key).is_some() {
        let layer = take_self_attn_layer(store, sig, prefix, d_model, n_heads)?;
        Ok(TransformerLayerWeights {
            self_attn: Some(layer),
            cross_attn: None,
        })
    } else if store.try_take(sig, &cross_key).is_some() {
        let layer = take_cross_attn_layer(store, sig, prefix, d_model, n_heads)?;
        Ok(TransformerLayerWeights {
            self_attn: None,
            cross_attn: Some(layer),
        })
    } else {
        anyhow::bail!("transformer layer {}: neither self_attn nor cross_attn found", prefix)
    }
}

impl CrossDomainTransformer {
    /// Load the cross-domain transformer from a WeightStore.
    ///
    /// `bottleneck_ch` is the channel dim of the last encoder (384 for
    /// HTDemucs). `bottom_channels` is the transformer's internal dim
    /// (512 for 4-stem/ft, 384 for 6-stem — in the latter case channel
    /// upsamplers are absent).
    pub fn from_store(
        store: &WeightStore,
        sig: &str,
        bottleneck_ch: usize,
        bottom_channels: usize,
    ) -> anyhow::Result<Self> {
        let n_heads = crate::T_HEADS;
        let has_resample = bottleneck_ch != bottom_channels;

        // Channel resamplers (optional, 6-stem has none).
        let (channel_upsampler, channel_upsampler_bias) = if has_resample {
            let w = take_conv1d(store, sig, "channel_upsampler")?;
            let b = store.take(sig, "channel_upsampler.bias")?.to_f32_vec();
            (Some(w), Some(b))
        } else {
            (None, None)
        };
        let (channel_downsampler, channel_downsampler_bias) = if has_resample {
            let w = take_conv1d(store, sig, "channel_downsampler")?;
            let b = store.take(sig, "channel_downsampler.bias")?.to_f32_vec();
            (Some(w), Some(b))
        } else {
            (None, None)
        };
        let (channel_upsampler_t, channel_upsampler_t_bias) = if has_resample {
            let w = take_conv1d(store, sig, "channel_upsampler_t")?;
            let b = store.take(sig, "channel_upsampler_t.bias")?.to_f32_vec();
            (Some(w), Some(b))
        } else {
            (None, None)
        };
        let (channel_downsampler_t, channel_downsampler_t_bias) = if has_resample {
            let w = take_conv1d(store, sig, "channel_downsampler_t")?;
            let b = store.take(sig, "channel_downsampler_t.bias")?.to_f32_vec();
            (Some(w), Some(b))
        } else {
            (None, None)
        };

        // Input norms.
        let norm_in = take_layernorm(store, sig, "crosstransformer.norm_in")?;
        let norm_in_t = take_layernorm(store, sig, "crosstransformer.norm_in_t")?;

        // 5 transformer layers each for freq and time.
        let mut layers = Vec::with_capacity(crate::T_LAYERS);
        for i in 0..crate::T_LAYERS {
            layers.push(take_transformer_layer(
                store,
                sig,
                &format!("crosstransformer.layers.{}", i),
                bottom_channels,
                n_heads,
            )?);
        }
        let mut layers_t = Vec::with_capacity(crate::T_LAYERS);
        for i in 0..crate::T_LAYERS {
            layers_t.push(take_transformer_layer(
                store,
                sig,
                &format!("crosstransformer.layers_t.{}", i),
                bottom_channels,
                n_heads,
            )?);
        }

        Ok(Self {
            norm_in,
            norm_in_t,
            channel_upsampler,
            channel_upsampler_bias,
            channel_downsampler,
            channel_downsampler_bias,
            channel_upsampler_t,
            channel_upsampler_t_bias,
            channel_downsampler_t,
            channel_downsampler_t_bias,
            layers,
            layers_t,
        })
    }
}

// ─── Top-level HTDemucs model ────────────────────────────────────────────────

/// One full HTDemucs model: 4 freq encoders + 4 time encoders + cross-domain
/// transformer + 4 freq decoders + 4 time decoders + freq embedding.
pub struct HTDemucs {
    pub encoders: Vec<HEncLayer>,
    pub tencoders: Vec<TEncLayer>,
    pub crosstransformer: CrossDomainTransformer,
    pub decoders: Vec<HDecLayer>,
    pub tdecoders: Vec<TDecLayer>,
    pub freq_emb: FreqEmb,
    pub n_sources: usize,
    pub bottom_channels: usize,
}

impl HTDemucs {
    /// Load a complete HTDemucs model from the store under one signature.
    ///
    /// `n_sources` is the number of stems (4 for htdemucs/htdemucs_ft, 6 for
    /// htdemucs_6s). `bottom_channels` is the transformer's d_model
    /// (512 for 4-stem/ft, 384 for 6-stem).
    pub fn from_store(
        store: &WeightStore,
        sig: &str,
        n_sources: usize,
        bottom_channels: usize,
    ) -> anyhow::Result<Self> {
        let depth = crate::DEPTH as usize;
        let channels = crate::CHANNELS; // 48
        let bottleneck_ch = channels * 2_usize.pow(depth as u32 - 1); // 384
        let freq_out_channels = n_sources * 4; // 4-stem: 16, 6-stem: 24
        let time_out_channels = n_sources * 2; // 4-stem: 8, 6-stem: 12

        // ─── Encoders ─────────────────────────────────────────────────────
        // PyTorch ordering: encoder.{i} and tencoder.{i} are 0=shallow..3=deep.
        // TEncLayer layer 0: chin=2 (stereo), chout=48; layer i: chin=48·2^(i-1), chout=48·2^i.
        let mut encoders = Vec::with_capacity(depth);
        let mut tencoders = Vec::with_capacity(depth);
        for i in 0..depth {
            let chout = channels * 2_usize.pow(i as u32);
            let t_chin = if i == 0 { 2 } else { channels * 2_usize.pow(i as u32 - 1) };
            encoders.push(HEncLayer::from_store(store, sig, &format!("encoder.{}", i))?);
            tencoders.push(TEncLayer::from_store(
                store,
                sig,
                &format!("tencoder.{}", i),
                t_chin,
                chout,
            )?);
        }

        // ─── Cross-domain transformer ─────────────────────────────────────
        let crosstransformer = CrossDomainTransformer::from_store(
            store, sig, bottleneck_ch, bottom_channels,
        )?;

        // ─── Decoders ──────────────────────────────────────────────────────
        // PyTorch decoder.{i} is the deepest-first ordering: decoder.0 = i=3.
        // We store them in the same order as PyTorch (decoder.0..3) so the
        // forward loop pops them in reverse and matches the encoder skips.
        let mut decoders = Vec::with_capacity(depth);
        let mut tdecoders = Vec::with_capacity(depth);
        for i in 0..depth {
            // PyTorch depth-reversed layer: i=0 is the deepest (i=DEPTH-1 in
            // build terms). chin = bottleneck_ch / 2^(DEPTH-1-i).
            //   i=0: chin = 384,  i=1: chin=192,  i=2: chin=96,  i=3: chin=48
            let chin = bottleneck_ch / 2_usize.pow(i as u32);
            let chout_freq = if i + 1 < depth {
                bottleneck_ch / 2_usize.pow((i + 1) as u32)
            } else {
                freq_out_channels
            };
            let chout_time = if i + 1 < depth {
                bottleneck_ch / 2_usize.pow((i + 1) as u32)
            } else {
                time_out_channels
            };
            let last = i + 1 == depth;
            decoders.push(HDecLayer::from_store(
                store, sig, &format!("decoder.{}", i), chin, chout_freq, last,
            )?);
            tdecoders.push(TDecLayer::from_store(
                store, sig, &format!("tdecoder.{}", i), chin, chout_time, last,
            )?);
        }

        // ─── Freq embedding (ScaledEmbedding, scale=10 baked in by FreqEmb) ─
        let freq_emb = FreqEmb::from_store(store, sig, channels)?;

        Ok(Self {
            encoders,
            tencoders,
            crosstransformer,
            decoders,
            tdecoders,
            freq_emb,
            n_sources,
            bottom_channels,
        })
    }
}

