//! CPU forward operators for HTDemucs — pure Rust + gemm crate + rayon.
//!
//! All operators work on plain `Vec<f32>` with explicit shape tracking.
//! Numerics match the burn reference exactly (same padding/stride/activation).
//!
//! Layout conventions:
//! - 4D tensors: [B, C, H, W] row-major (batch, channel, height, width)
//! - 3D tensors: [B, C, L] row-major (batch, channel, length)
//! - Weights: PyTorch layout ([out, in, kH, kW] for conv2d, [out, in, k] for conv1d)

use gemm::{gemm, Parallelism};
use rayon::prelude::*;

use crate::model::{
    Bias, Conv1dWeight, Conv2dWeight, DConv, DConvLayer, FreqEncoder, GroupNorm1, HDecLayer,
    HEncLayer, HTDemucs, LayerNorm1, LayerScale, MhaWeights, TDecLayer, TEncLayer, Weight2D,
};

// ═══════════════════════════════════════════════════════════════════════
//  Conv2d (stride, no dilation) via im2col + GEMM
// ═══════════════════════════════════════════════════════════════════════

/// Conv2d forward via im2col.
///
/// Input:  x   [B, C_in, H, W] row-major
/// Weight: w   [C_out, C_in, kH, kW] row-major (PyTorch layout)
/// Bias:   bias [C_out] (added to each output element)
///
/// Padding: (pad_h, pad_w) applied symmetrically.
/// Stride:  (stride_h, stride_w).
/// No dilation (dilation = 1).
///
/// Output: [B, C_out, H_out, W_out] where
///   H_out = (H + 2*pad_h - kH) / stride_h + 1
///   W_out = (W + 2*pad_w - kW) / stride_w + 1
///
/// Returns the output as a flat Vec + shape.
pub fn conv2d(
    x: &[f32],
    x_shape: [usize; 4], // [B, C_in, H, W]
    w: &Conv2dWeight,
    bias: &Bias,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> (Vec<f32>, [usize; 4]) {
    let [b, c_in, h, width] = x_shape;
    let [c_out, _in, kh, kw] = [w.out_ch, w.in_ch, w.kh, w.kw];
    assert_eq!(c_in, w.in_ch, "conv2d channel mismatch");

    let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (width + 2 * pad_w - kw) / stride_w + 1;

    // im2col: build a matrix of shape [B*H_out*W_out, C_in*kH*kW]
    // Each row = the receptive field of one output position, unrolled.
    let patch_size = c_in * kh * kw;
    let n_rows = b * h_out * w_out;
    let mut col = vec![0.0f32; n_rows * patch_size];

    // im2col — parallel over output rows (each row independent).
    col.par_chunks_mut(patch_size).enumerate().for_each(|(row_idx, row_slice)| {
        let bi = row_idx / (h_out * w_out);
        let rem = row_idx % (h_out * w_out);
        let oh = rem / w_out;
        let ow = rem % w_out;
        for ci in 0..c_in {
            for dkh in 0..kh {
                for dkw in 0..kw {
                    let ih = oh * stride_h + dkh;
                    let iw = ow * stride_w + dkw;
                    // Padding: if ih/iw fall in the padded region, value is 0.
                    let ih_s = ih as isize - pad_h as isize;
                    let iw_s = iw as isize - pad_w as isize;
                    let val = if ih_s >= 0
                        && iw_s >= 0
                        && (ih_s as usize) < h
                        && (iw_s as usize) < width
                    {
                        x[((bi * c_in + ci) * h + ih_s as usize) * width + iw_s as usize]
                    } else {
                        0.0
                    };
                    row_slice[(ci * kh + dkh) * kw + dkw] = val;
                }
            }
        }
    });

    // GEMM: out[n_rows, c_out] = col[n_rows, patch_size] @ W^T[patch_size, c_out]
    // W is [c_out, patch_size] row-major. For gemm's rhs B[k,n] we want B[k,n]=W[n,k].
    // rhs_cs (n+1 stride) = patch_size (W[n+1,k] is patch_size away in row-major W)
    // rhs_rs (k+1 stride) = 1 (W[n,k+1] is 1 away in row-major W)
    let mut out = vec![0.0f32; n_rows * c_out];
    unsafe {
        gemm(
            n_rows,
            c_out,
            patch_size,
            out.as_mut_ptr(),
            1,
            c_out as isize,
            false,
            col.as_ptr(),
            1,
            patch_size as isize,
            w.data.as_ptr(),
            patch_size as isize, // rhs_cs
            1,                   // rhs_rs
            0.0,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }

    // Add bias to each row (parallel over rows).
    out.par_chunks_mut(c_out).for_each(|row| {
        for co in 0..c_out {
            row[co] += bias.data[co];
        }
    });

    // Reshape to [B, C_out, H_out, W_out].
    // GEMM produced [n_rows, c_out] = [B*H_out*W_out, C_out], row-major.
    // We need [B, C_out, H_out, W_out]. The rows iterate as (b, oh, ow), columns as co.
    // Reorder: for each b, for each co, for each (oh,ow): out[b,co,oh,ow] = gemm[b*H_out*W_out*..., co]
    let mut reshaped = vec![0.0f32; b * c_out * h_out * w_out];
    reshaped
        .par_chunks_mut(c_out * h_out * w_out)
        .enumerate()
        .for_each(|(bi, bplane)| {
            bplane.par_chunks_mut(h_out * w_out).enumerate().for_each(|(co, coplane)| {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let src_row = (bi * h_out + oh) * w_out + ow;
                        coplane[oh * w_out + ow] = out[src_row * c_out + co];
                    }
                }
            });
        });

    (reshaped, [b, c_out, h_out, w_out])
}

// ═══════════════════════════════════════════════════════════════════════
//  Conv1d (with dilation) via im2col + GEMM
// ═══════════════════════════════════════════════════════════════════════

/// Conv1d forward via im2col, supporting dilation.
///
/// Input:  x   [B, C_in, L] row-major
/// Weight: w   [C_out, C_in, K] row-major
/// Padding: `pad` applied symmetrically (left and right).
/// Stride:  always 1 (DConv uses stride-1 convs; for stride>1 use
///          [`conv1d_with_stride`]).
/// Dilation: spacing between kernel taps.
pub fn conv1d(
    x: &[f32],
    x_shape: [usize; 3], // [B, C_in, L]
    w: &Conv1dWeight,
    bias: &Bias,
    pad: usize,
    dilation: usize,
) -> (Vec<f32>, [usize; 3]) {
    conv1d_with_stride(x, x_shape, w, bias, pad, 1, dilation)
}

/// 1D convolution with explicit stride. Output length:
/// `L_out = (L + 2*pad - dilation*(k-1) - 1) / stride + 1`.
pub fn conv1d_with_stride(
    x: &[f32],
    x_shape: [usize; 3],
    w: &Conv1dWeight,
    bias: &Bias,
    pad: usize,
    stride: usize,
    dilation: usize,
) -> (Vec<f32>, [usize; 3]) {
    let [b, c_in, l] = x_shape;
    let k = w.k;
    let c_out = w.out_ch;
    assert_eq!(c_in, w.in_ch, "conv1d channel mismatch");

    let l_out = (l + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    let patch_size = c_in * k;
    let n_rows = b * l_out;

    // im2col: [n_rows, patch_size] — parallel over output rows.
    let mut col = vec![0.0f32; n_rows * patch_size];
    col.par_chunks_mut(patch_size).enumerate().for_each(|(row_idx, row_slice)| {
        let bi = row_idx / l_out;
        let ol = row_idx % l_out;
        for ci in 0..c_in {
            for dk in 0..k {
                let il = ol as isize * stride as isize
                    + dk as isize * dilation as isize
                    - pad as isize;
                let val = if il >= 0 && (il as usize) < l {
                    x[(bi * c_in + ci) * l + il as usize]
                } else {
                    0.0
                };
                row_slice[ci * k + dk] = val;
            }
        }
    });

    // GEMM: out[n_rows, c_out] = col[n_rows, patch_size] @ W^T[patch_size, c_out]
    let mut out = vec![0.0f32; n_rows * c_out];
    // out = col @ W^T. gemm signature: C[m,n] = A[m,k] @ B[k,n]. We pass B with
    // transposed strides so that B[k,n] reads W[n,k] from the [c_out, patch_size] row-major weight.
    unsafe {
        gemm(
            n_rows,
            c_out,
            patch_size,
            out.as_mut_ptr(),
            1,
            c_out as isize,
            false,
            col.as_ptr(),
            1,
            patch_size as isize,
            w.data.as_ptr(),
            // W is [c_out, patch_size] row-major. For B[k,n] = W[n,k]:
            // rhs_cs (n+1 → next column) = patch_size (W[n+1, k] is patch_size away)
            // rhs_rs (k+1 → next row)    = 1 (W[n, k+1] is 1 away)
            patch_size as isize, // rhs_cs
            1,                   // rhs_rs
            0.0,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }

    // Add bias (parallel over rows).
    out.par_chunks_mut(c_out).for_each(|row| {
        for co in 0..c_out {
            row[co] += bias.data[co];
        }
    });

    // Reshape [n_rows, c_out] → [B, C_out, L_out] — parallel over (b, co).
    let mut reshaped = vec![0.0f32; b * c_out * l_out];
    reshaped
        .par_chunks_mut(c_out * l_out)
        .enumerate()
        .for_each(|(bi, bplane)| {
            bplane.par_chunks_mut(l_out).enumerate().for_each(|(co, coplane)| {
                for ol in 0..l_out {
                    coplane[ol] = out[(bi * l_out + ol) * c_out + co];
                }
            });
        });

    (reshaped, [b, c_out, l_out])
}

// ═══════════════════════════════════════════════════════════════════════
//  ConvTranspose2d (im2col + GEMM, no dilation)
// ═══════════════════════════════════════════════════════════════════════

/// 2D transposed convolution (a.k.a. fractionally-strided conv).
///
/// PyTorch weight layout: `[C_in, C_out, kH, kW]` (note: in↔out swapped
/// compared to Conv2d). For each output position `(oh, ow)`, the contributing
/// input position is `ih = oh * stride_h - pad_h + dkh`, `iw = ow * stride_w
/// - pad_w + dkw`. Only `(ih, iw)` within `[0, H_in) × [0, W_in)` contribute.
///
/// Output shape: `H_out = (H_in - 1) * stride_h - 2 * pad_h + kH` (analogously W).
///
/// We use the im2col trick: build a matrix `col` of shape `[B*H_out*W_out,
/// C_in*kH*kW]` where each row is the input patch that contributes to one
/// output position. Then `out = col @ W_unrolled` (where W_unrolled is
/// `[C_in*kH*kW, C_out]` via a transposed-stride view). Reshape `out` to
/// `[B*H_out*W_out, C_out]` → permute → `[B, C_out, H_out, W_out]`.
pub fn conv_transpose2d(
    x: &[f32],
    x_shape: [usize; 4], // [B, C_in, H, W]
    w: &Conv2dWeight,    // [C_in, C_out, kH, kW] PyTorch ConvTranspose layout
    bias: &Bias,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> (Vec<f32>, [usize; 4]) {
    let [b, c_in, h_in, w_in] = x_shape;
    let [c_in_w, c_out, kh, kw] = [w.in_ch, w.out_ch, w.kh, w.kw];
    assert_eq!(c_in, c_in_w, "conv_transpose2d channel mismatch");

    // PyTorch ConvTranspose2d: H_out = (H_in - 1) * stride - 2 * pad +
    // dilation * (k - 1) + output_padding + 1. With dilation=1, output_padding=0,
    // H_out = (H_in - 1) * stride + (k - 1) - 2 * pad + 1.
    let h_out = (h_in - 1) * stride_h + (kh - 1) - 2 * pad_h + 1;
    let w_out = (w_in - 1) * stride_w + (kw - 1) - 2 * pad_w + 1;

    // im2col: [B*H_out*W_out, C_in*kH*kW]
    let patch = c_in * kh * kw;
    let n_rows = b * h_out * w_out;
    let mut col = vec![0.0f32; n_rows * patch];

    // im2col — parallel over output rows (each row independent). Same math,
    // just computed across the rayon pool. n_rows is large (4× upsampling).
    col.par_chunks_mut(patch).enumerate().for_each(|(row, row_slice)| {
        let bi = row / (h_out * w_out);
        let rem = row % (h_out * w_out);
        let oh = rem / w_out;
        let ow = rem % w_out;
        for ci in 0..c_in {
            for dkh in 0..kh {
                for dkw in 0..kw {
                    // ConvTranspose reverse index: which input position
                    // (ih, iw) contributes to (oh, ow) via (dkh, dkw)?
                    //   ih * stride_h + dkh - pad_h = oh
                    //   ih = (oh + pad_h - dkh) / stride_h
                    // Valid only when (oh + pad_h - dkh) is divisible
                    // by stride_h.
                    let oh_p = oh as isize + pad_h as isize - dkh as isize;
                    let ow_p = ow as isize + pad_w as isize - dkw as isize;
                    if oh_p < 0
                        || ow_p < 0
                        || oh_p % stride_h as isize != 0
                        || ow_p % stride_w as isize != 0
                    {
                        continue;
                    }
                    let ih_s = oh_p / stride_h as isize;
                    let iw_s = ow_p / stride_w as isize;
                    let val = if ih_s >= 0
                        && iw_s >= 0
                        && (ih_s as usize) < h_in
                        && (iw_s as usize) < w_in
                    {
                        x[((bi * c_in + ci) * h_in + ih_s as usize) * w_in + iw_s as usize]
                    } else {
                        0.0
                    };
                    row_slice[(ci * kh + dkh) * kw + dkw] = val;
                }
            }
        }
    });

    // GEMM: out[n_rows, c_out] = col[n_rows, patch] @ W_unrolled[patch, c_out]
    // After the load-time reorder in `take_conv_transpose2d`, the weight data
    // is stored as `[patch, c_out]` row-major (i.e. `reordered[i, oc] =
    // a[ic, oc, kh, kw]` at memory `i*c_out + oc`). Hence:
    //   rhs_rs (k+1 stride, across patch) = c_out
    //   rhs_cs (n+1 stride, across c_out) = 1
    let mut out = vec![0.0f32; n_rows * c_out];
    unsafe {
        gemm(
            n_rows,
            c_out,
            patch,
            out.as_mut_ptr(),
            1,
            c_out as isize,
            false,
            col.as_ptr(),
            1,
            patch as isize,
            w.data.as_ptr(),
            1,              // rhs_cs (n+1 stride, across c_out) = 1
            c_out as isize, // rhs_rs (k+1 stride, across patch) = c_out
            0.0,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }

    // Add bias per output position (parallel over rows).
    out.par_chunks_mut(c_out).for_each(|row| {
        for co in 0..c_out {
            row[co] += bias.data[co];
        }
    });

    // Reshape [B*H_out*W_out, C_out] → [B, C_out, H_out, W_out]
    // Rows iterate as (b, oh, ow); we need to reorder so the C_out dim is
    // axis 1 (PyTorch layout). Parallel over the destination's (b, co) planes.
    let mut reshaped = vec![0.0f32; b * c_out * h_out * w_out];
    reshaped
        .par_chunks_mut(c_out * h_out * w_out)
        .enumerate()
        .for_each(|(bi, bplane)| {
            bplane.par_chunks_mut(h_out * w_out).enumerate().for_each(|(co, coplane)| {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let src_row = (bi * h_out + oh) * w_out + ow;
                        coplane[oh * w_out + ow] = out[src_row * c_out + co];
                    }
                }
            });
        });

    (reshaped, [b, c_out, h_out, w_out])
}

// ═══════════════════════════════════════════════════════════════════════
//  ConvTranspose1d (im2col + GEMM, no dilation)
// ═══════════════════════════════════════════════════════════════════════

/// 1D transposed convolution.
///
/// PyTorch weight layout: `[C_in, C_out, k]`. Output length:
/// `L_out = (L_in - 1) * stride + k - 2 * pad`.
pub fn conv_transpose1d(
    x: &[f32],
    x_shape: [usize; 3], // [B, C_in, L]
    w: &Conv1dWeight,    // [C_in, C_out, k]
    bias: &Bias,
    pad: usize,
    stride: usize,
) -> (Vec<f32>, [usize; 3]) {
    let [b, c_in, l_in] = x_shape;
    let [c_in_w, c_out, k] = [w.in_ch, w.out_ch, w.k];
    assert_eq!(c_in, c_in_w, "conv_transpose1d channel mismatch");

    let l_out = (l_in - 1) * stride + (k - 1) - 2 * pad + 1;

    let patch = c_in * k;
    let n_rows = b * l_out;
    let mut col = vec![0.0f32; n_rows * patch];
    for bi in 0..b {
        for ol in 0..l_out {
            let row = bi * l_out + ol;
            let dst = row * patch;
            for ci in 0..c_in {
                for dk in 0..k {
                    // ConvTranspose reverse index: il = (ol + pad - dk) / stride
                    let ol_p = ol as isize + pad as isize - dk as isize;
                    if ol_p < 0 || ol_p % stride as isize != 0 {
                        continue;
                    }
                    let il_s = ol_p / stride as isize;
                    let val = if (il_s as usize) < l_in {
                        x[(bi * c_in + ci) * l_in + il_s as usize]
                    } else {
                        0.0
                    };
                    col[dst + ci * k + dk] = val;
                }
            }
        }
    }

    let mut out = vec![0.0f32; n_rows * c_out];
    unsafe {
        gemm(
            n_rows,
            c_out,
            patch,
            out.as_mut_ptr(),
            1,
            c_out as isize,
            false,
            col.as_ptr(),
            1,
            patch as isize,
            w.data.as_ptr(),
            1,              // rhs_cs (n+1 stride, across c_out) = 1
            c_out as isize, // rhs_rs (k+1 stride, across patch) = c_out
            0.0,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }

    for r in 0..n_rows {
        for co in 0..c_out {
            out[r * c_out + co] += bias.data[co];
        }
    }

    // Reshape [B*L_out, C_out] → [B, C_out, L_out]
    let mut reshaped = vec![0.0f32; b * c_out * l_out];
    for bi in 0..b {
        for co in 0..c_out {
            for ol in 0..l_out {
                reshaped[(bi * c_out + co) * l_out + ol] = out[(bi * l_out + ol) * c_out + co];
            }
        }
    }

    (reshaped, [b, c_out, l_out])
}

// ═══════════════════════════════════════════════════════════════════════
//  Element-wise activations and norms
// ═══════════════════════════════════════════════════════════════════════

/// GELU activation (exact erf-based, matching burn/PyTorch).
///
/// `GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))`
///
/// burn's ndarray backend uses libm::erf on f64, so we do the same for
/// bit-exact alignment.
pub fn gelu(x: &mut [f32]) {
    let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
    x.par_iter_mut().for_each(|v| {
        let erf_val = libm::erf((*v as f64) * (inv_sqrt2 as f64)) as f32;
        *v = 0.5 * *v * (1.0 + erf_val);
    });
}

/// GroupNorm with 1 group (= normalization over all channels jointly).
/// Input layout: [B, C, L]. Normalizes over (C, L) per batch element.
///
/// burn GroupNormConfig::new(1, num_channels) → 1 group.
pub fn groupnorm1(x: &mut [f32], shape: [usize; 3], gn: &GroupNorm1) {
    let [b, c, l] = shape;
    let cl = c * l;
    // Parallel over batch elements (each is independent). In the conv path
    // b = B*Fr which is large; matches torch GroupNorm math exactly.
    x.par_chunks_mut(cl).for_each(|slice| {
        // mean/variance in f32 — matches torch GroupNorm + the CUDA path.
        let mean = slice.iter().sum::<f32>() / cl as f32;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cl as f32;
        let inv_std = (var + 1e-5).recip().sqrt();
        // normalize + scale + shift
        for ci in 0..c {
            let g = gn.gamma[ci];
            let bt = gn.beta[ci];
            for li in 0..l {
                let idx = ci * l + li;
                slice[idx] = (slice[idx] - mean) * inv_std * g + bt;
            }
        }
    });
}

/// GLU activation along the channel dimension (dim=1 for [B, C, L]).
/// Splits channels in half: first half * sigmoid(second half).
/// Input: [B, 2*C, L] → Output: [B, C, L].
pub fn glu_channel(x: &[f32], shape: [usize; 3]) -> (Vec<f32>, [usize; 3]) {
    let [b, c2, l] = shape;
    assert_eq!(c2 % 2, 0, "GLU: channels must be even");
    let c = c2 / 2;
    let mut out = vec![0.0f32; b * c * l];
    // Parallel over (b, c) — each (b,c) writes a contiguous row of length l.
    out.par_chunks_mut(l).enumerate().for_each(|(idx, row)| {
        let bi = idx / c;
        let ci = idx % c;
        let a_base = (bi * c2 + ci) * l;
        let b_base = (bi * c2 + c + ci) * l;
        for li in 0..l {
            let a = x[a_base + li];
            let b_val = x[b_base + li];
            let sig = 1.0 / (1.0 + (-b_val).exp());
            row[li] = a * sig;
        }
    });
    (out, [b, c, l])
}

/// Apply LayerScale: x * scale[c] for each channel.
/// Input: [B, C, L], scale: [C].
pub fn layer_scale(x: &mut [f32], shape: [usize; 3], scale: &LayerScale) {
    let [b, c, l] = shape;
    x.par_chunks_mut(c * l).enumerate().for_each(|(bi, bplane)| {
        let _ = bi;
        bplane.par_chunks_mut(l).enumerate().for_each(|(ci, row)| {
            let s = scale.scale[ci];
            for v in row.iter_mut() {
                *v *= s;
            }
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════
//  DConvLayer + DConv forward
// ═══════════════════════════════════════════════════════════════════════

/// One DConvLayer forward.
///
/// Input/output: [B, C, L] row-major.
/// Dilation = `1 << layer_index` (layer 0 → dilation 1, layer 1 → dilation 2).
///
/// Forward:
///   residual = x
///   x = conv1(x, k=3, pad=dilation, dilation=dilation)  → [B, compress, L]
///   x = groupnorm1(x)
///   x = gelu(x)
///   x = conv2(x, k=1)                                    → [B, 2*C, L]
///   x = groupnorm2(x)
///   x = glu(x)                                           → [B, C, L]
///   x = layer_scale(x)
///   return x + residual
pub fn dconv_layer_forward(
    x: &[f32],
    shape: [usize; 3],
    layer: &DConvLayer,
    dilation: usize,
) -> (Vec<f32>, [usize; 3]) {
    let [b, c, l] = shape;
    // Save residual (trimmed to output length after conv1 may change it — but
    // with pad=dilation and k=3, conv1 preserves length).
    let residual = x.to_vec();

    // conv1: k=3, pad=dilation, dilation=dilation → same length
    let (mut h, h_shape) = conv1d(x, shape, &layer.conv1, &layer.conv1_bias, dilation, dilation);
    // groupnorm1
    groupnorm1(&mut h, h_shape, &layer.norm1);
    // gelu
    gelu(&mut h);
    // conv2: k=1, pad=0, dilation=1
    let (mut h2, h2_shape) = conv1d(&h, h_shape, &layer.conv2, &layer.conv2_bias, 0, 1);
    // groupnorm2
    groupnorm1(&mut h2, h2_shape, &layer.norm2);
    // glu → [B, C, L]
    let (mut h3, h3_shape) = glu_channel(&h2, h2_shape);
    // layer_scale
    layer_scale(&mut h3, h3_shape, &layer.scale);
    // residual add (trim residual to match output length if needed)
    let [_b, _c, l_out] = h3_shape;
    let mut out = h3;
    for bi in 0..b {
        for ci in 0..c {
            for li in 0..l_out {
                let idx = (bi * c + ci) * l_out + li;
                out[idx] += residual[(bi * c + ci) * l + li];
            }
        }
    }
    (out, h3_shape)
}

/// DConv forward (2 stacked DConvLayers, dilation 1 then 2).
pub fn dconv_forward(x: &[f32], shape: [usize; 3], dconv: &DConv) -> (Vec<f32>, [usize; 3]) {
    let mut data = x.to_vec();
    let mut s = shape;
    for (j, layer) in dconv.layers.iter().enumerate() {
        let dilation = 1 << j; // layer 0 → 1, layer 1 → 2
        let (out, out_shape) = dconv_layer_forward(&data, s, layer, dilation);
        data = out;
        s = out_shape;
    }
    (data, s)
}

// ═══════════════════════════════════════════════════════════════════════
//  HEncLayer forward (frequency encoder)
// ═══════════════════════════════════════════════════════════════════════

/// HEncLayer forward.
///
/// Input:  x [B, C_in, Fr, T] row-major (4D)
/// Output: [B, C_out, Fr_out, T] row-major + shape
///
/// Forward:
///   x = conv2d(x, k=[8,1], s=[4,1], p=[2,0])  → [B, C_out, Fr/4, T]
///   x = gelu(x)
///   reshape → [B*Fr_out, C_out, T]
///   x = dconv(x)
///   reshape back → [B, C_out, Fr_out, T]
///   x = conv2d(x, k=[1,1], s=[1,1], p=[0,0])  → [B, 2*C_out, Fr_out, T]
///   x = glu(x, dim=1)                         → [B, C_out, Fr_out, T]
pub fn henc_layer_forward(
    x: &[f32],
    x_shape: [usize; 4],
    layer: &HEncLayer,
) -> (Vec<f32>, [usize; 4]) {
    let [b, _c_in, _fr, _t] = x_shape;

    // 1. Conv2d [8,1] stride [4,1] pad [2,0]
    let (mut h, h_shape) = conv2d(
        x,
        x_shape,
        &layer.conv,
        &layer.conv_bias,
        2, // KERNEL_SIZE/4
        0,
        4, // STRIDE
        1,
    );
    // 2. GELU
    gelu(&mut h);

    // 3. Reshape [B, C_out, Fr_out, T] → [B*Fr_out, C_out, T]
    let [_, c_out, fr_out, t] = h_shape;
    let mut flat = vec![0.0f32; b * fr_out * c_out * t];
    for bi in 0..b {
        for ci in 0..c_out {
            for fi in 0..fr_out {
                for ti in 0..t {
                    // src [bi, ci, fi, ti] → dst [bi*fr_out + fi, ci, ti]
                    flat[((bi * fr_out + fi) * c_out + ci) * t + ti] =
                        h[((bi * c_out + ci) * fr_out + fi) * t + ti];
                }
            }
        }
    }

    // 4. DConv
    let (dconv_out, dconv_shape) = dconv_forward(&flat, [b * fr_out, c_out, t], &layer.dconv);

    // 5. Reshape back [B*Fr_out, C_out, T] → [B, C_out, Fr_out, T]
    let [n, c2, t2] = dconv_shape;
    assert_eq!(n, b * fr_out);
    assert_eq!(c2, c_out);
    let mut unflat = vec![0.0f32; b * c_out * fr_out * t2];
    for bi in 0..b {
        for fi in 0..fr_out {
            for ci in 0..c_out {
                for ti in 0..t2 {
                    unflat[((bi * c_out + ci) * fr_out + fi) * t2 + ti] =
                        dconv_out[((bi * fr_out + fi) * c_out + ci) * t2 + ti];
                }
            }
        }
    }

    // 6. Rewrite: Conv2d [1,1] → [B, 2*C_out, Fr_out, T]
    let rewrite_bias = Bias {
        data: layer.rewrite_bias.data.clone(),
        len: layer.rewrite_bias.len,
    };
    let (rewritten, rw_shape) = conv2d(
        &unflat,
        [b, c_out, fr_out, t2],
        &layer.rewrite,
        &rewrite_bias,
        0,
        0,
        1,
        1,
    );

    // 7. GLU along channel dim.
    // rewritten is [B, 2*C_out, Fr_out, T]. GLU splits channels in half.
    let [b2, c2x, fr2, t3] = rw_shape;
    assert_eq!(c2x, 2 * c_out);
    let c_glu = c2x / 2;
    let mut out = vec![0.0f32; b2 * c_glu * fr2 * t3];
    for bi in 0..b2 {
        for ci in 0..c_glu {
            for fi in 0..fr2 {
                for ti in 0..t3 {
                    let a = rewritten[((bi * c2x + ci) * fr2 + fi) * t3 + ti];
                    let b_val = rewritten[((bi * c2x + c_glu + ci) * fr2 + fi) * t3 + ti];
                    let sig = 1.0 / (1.0 + (-b_val).exp());
                    out[((bi * c_glu + ci) * fr2 + fi) * t3 + ti] = a * sig;
                }
            }
        }
    }

    (out, [b2, c_glu, fr2, t3])
}

// ═══════════════════════════════════════════════════════════════════════
//  FreqEncoder forward (4 HEncLayers + freq_emb)
// ═══════════════════════════════════════════════════════════════════════

/// Frequency encoder forward.
///
/// Input:  freq [B, 4, Fr, T]  (CaC format)
/// Output: [B, 384, Fr/256, T] + shape, plus per-layer skips for the decoder.
///
/// Matches burn htdemucs.rs:160-199:
///   1. layers[0].forward(freq)
///   2. freq_emb applied (* 0.2): freq[*, fi, *] += emb[fi] * 0.2
///   3. save skip[0] (after freq_emb)
///   4. layers[1..3].forward + save skip[1..3]
pub fn freq_encoder_forward(
    freq: &[f32],
    freq_shape: [usize; 4],
    enc: &FreqEncoder,
) -> (Vec<f32>, [usize; 4], Vec<(Vec<f32>, [usize; 4])>) {
    let mut skips: Vec<(Vec<f32>, [usize; 4])> = Vec::with_capacity(4);

    // Layer 0
    let (mut h, mut h_shape) = henc_layer_forward(freq, freq_shape, &enc.layers[0]);

    // Apply freq_emb AFTER encoder[0], BEFORE saving skip (matches burn).
    // emb[fi] is [dim], added to h[b, :, fi, t] for all b, t.
    // freq_emb_scale = 0.2 (applied here, not at load time).
    let [b, c, fr, t] = h_shape;
    let emb = &enc.freq_emb;
    let scale = 2.0;
    for bi in 0..b {
        for ci in 0..c {
            for fi in 0..fr {
                let emb_val = emb.data[fi * emb.dim + ci] * scale;
                for ti in 0..t {
                    h[((bi * c + ci) * fr + fi) * t + ti] += emb_val;
                }
            }
        }
    }

    // Save skip[0] (after freq_emb)
    skips.push((h.clone(), h_shape));

    // Layers 1..3
    for i in 1..4 {
        let (out, out_shape) = henc_layer_forward(&h, h_shape, &enc.layers[i]);
        h = out;
        h_shape = out_shape;
        skips.push((h.clone(), h_shape));
    }

    (h, h_shape, skips)
}

// ═══════════════════════════════════════════════════════════════════════
//  Transformer building blocks
// ═══════════════════════════════════════════════════════════════════════

/// Layer normalization along the last (feature) dimension.
///
/// Input layout: `[B, S, D]` row-major. Normalizes over the last dim per
/// (batch, sequence position) pair. Matches burn `LayerNorm` (default eps
/// 1e-5).
pub fn layernorm(x: &mut [f32], shape: [usize; 3], ln: &LayerNorm1) {
    let [b, s, d] = shape;
    assert_eq!(d, ln.dim, "layernorm dim mismatch");
    // Parallel over each (b, s) row — each normalizes independently over d.
    x.par_chunks_mut(d).for_each(|slice| {
        let mean = slice.iter().sum::<f32>() / d as f32;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
        let inv_std = (var + 1e-5).recip().sqrt();
        for di in 0..d {
            slice[di] = (slice[di] - mean) * inv_std * ln.gamma[di] + ln.beta[di];
        }
    });
    let _ = (b, s);
}

/// Linear layer on a 3D `[B, S, D_in]` tensor → `[B, S, D_out]`.
///
/// Weight is in **PyTorch layout**: `weight[D_out, D_in]` row-major.
/// We use gemm's stride trick to avoid an actual transpose:
///   `out = x @ W^T` is computed as `gemm(..., rhs with transposed strides)`.
pub fn linear(
    x: &[f32],
    x_shape: [usize; 3], // [B, S, D_in]
    w: &Weight2D,        // [D_out, D_in] PyTorch layout
    bias: &[f32],        // [D_out]
) -> (Vec<f32>, [usize; 3]) {
    let [b, s, d_in] = x_shape;
    let d_out = w.rows;
    assert_eq!(d_in, w.cols, "linear dim mismatch");
    let bs = b * s;

    // Treat x as [bs, d_in], output as [bs, d_out]
    let mut out = vec![0.0f32; bs * d_out];
    unsafe {
        gemm(
            bs,
            d_out,
            d_in,
            out.as_mut_ptr(),
            1,
            d_out as isize,
            false,
            x.as_ptr(),
            1,
            d_in as isize,
            w.data.as_ptr(),
            // W is [d_out, d_in] row-major. For B[k,n] = W[n,k] (rhs):
            //   rhs_cs (n+1 → next col) = d_in (W[n+1, k] is d_in away)
            //   rhs_rs (k+1 → next row) = 1 (W[n, k+1] is 1 away)
            d_in as isize, // rhs_cs
            1,             // rhs_rs
            0.0,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }

    // Add bias per row.
    for r in 0..bs {
        for o in 0..d_out {
            out[r * d_out + o] += bias[o];
        }
    }

    (out, [b, s, d_out])
}

/// Numerically-stable softmax along the sequence axis (axis=1) of `[B, S, D]`.
///
/// `axis` selects which dim to softmax over. 0 = batch, 1 = sequence, 2 = feature.
/// Specialised for the common transformer case (axis=1 over sequence).
pub fn softmax(x: &mut [f32], shape: [usize; 3], axis: usize) {
    let [b, s, d] = shape;
    match axis {
        1 => softmax_axis_s(x, b, s, d),
        _ => panic!("softmax: only axis=1 (sequence) is currently supported"),
    }
}

/// Softmax over axis=1 (sequence dim) of `[B, S, D]`. For each (b, d) pair,
/// softmax the s values. Layout: x[bi * s * d + si * d + di].
fn softmax_axis_s(x: &mut [f32], b: usize, s: usize, d: usize) {
    for bi in 0..b {
        for di in 0..d {
            // Gather max for stability.
            let mut max_val = f32::NEG_INFINITY;
            for si in 0..s {
                let v = x[bi * s * d + si * d + di];
                if v > max_val {
                    max_val = v;
                }
            }
            // Exponentiate and sum.
            let mut sum = 0.0f32;
            for si in 0..s {
                let idx = bi * s * d + si * d + di;
                x[idx] = (x[idx] - max_val).exp();
                sum += x[idx];
            }
            // Normalize.
            let inv_sum = 1.0 / sum;
            for si in 0..s {
                let idx = bi * s * d + si * d + di;
                x[idx] *= inv_sum;
            }
        }
    }
}

/// LayerScale for `[B, S, D]` tensors. Scale per last-dim, broadcast over
/// (batch, sequence). Matches burn's `LayerScale::forward_last` (which scales
/// each channel independently, batch and sequence broadcast).
pub fn layer_scale_last(x: &mut [f32], shape: [usize; 3], scale: &LayerScale) {
    let [b, s, d] = shape;
    // Last dim is contiguous; parallelize over (b, s) rows of length d.
    x.par_chunks_mut(d).for_each(|row| {
        for di in 0..d {
            row[di] *= scale.scale[di];
        }
    });
    let _ = (b, s);
}

/// 1D sinusoidal positional embedding matching burn's `create_sin_embedding`.
///
/// Layout: `[seq_len, d_model]`. First half = cos(phase), second half = sin(phase).
/// `phase[pos, i] = pos / 10000^(i / (half - 1))` for i = 0..half.
///
/// The result is unsqueezed by the caller as needed.
pub fn sin_embed_1d(seq_len: usize, d_model: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; seq_len * d_model];
    let half = d_model / 2;
    if half == 0 {
        return data;
    }
    let half_m1 = if half > 1 { (half - 1) as f32 } else { 1.0 };

    for pos in 0..seq_len {
        for i in 0..half {
            let angle = pos as f32 / (10000.0_f32).powf(i as f32 / half_m1);
            data[pos * d_model + i] = angle.cos(); // first half: cos
            data[pos * d_model + half + i] = angle.sin(); // second half: sin
        }
    }
    data
}

/// 2D sinusoidal positional embedding matching burn's `create_2d_sin_embedding`.
///
/// Returns `[width * height, d_model]` in **time-major** order (t varies
/// slowest). Layout within d_model:
///   - Channels 0, 2, 4, ..., half-2: sin(time_pos * div_term)
///   - Channels 1, 3, 5, ..., half-1: cos(time_pos * div_term)
///   - Channels half, half+2, ..., d_model-2: sin(freq_pos * div_term)
///   - Channels half+1, half+3, ..., d_model-1: cos(freq_pos * div_term)
///
/// `div_term[k] = exp(-2k * ln(10000) / half) = 1 / 10000^(2k/half)`
pub fn sin_embed_2d(d_model: usize, height: usize, width: usize) -> Vec<f32> {
    let half = d_model / 2;
    let quarter = half / 2;
    let seq_len = width * height;
    let mut data = vec![0.0f32; seq_len * d_model];

    if quarter == 0 {
        return data;
    }

    let div_terms: Vec<f32> = (0..quarter)
        .map(|k| (-2.0 * k as f32 * (10000.0_f32).ln() / half as f32).exp())
        .collect();

    for t in 0..width {
        for fr in 0..height {
            // time-major flatten: t varies slowest
            let s = t * height + fr;
            for k in 0..quarter {
                let w_angle = t as f32 * div_terms[k];
                let h_angle = fr as f32 * div_terms[k];
                // First half: width (time) encoding — interleaved sin/cos
                data[s * d_model + 2 * k] = w_angle.sin();
                data[s * d_model + 2 * k + 1] = w_angle.cos();
                // Second half: height (freq) encoding — interleaved sin/cos
                data[s * d_model + half + 2 * k] = h_angle.sin();
                data[s * d_model + half + 2 * k + 1] = h_angle.cos();
            }
        }
    }
    data
}

// ═══════════════════════════════════════════════════════════════════════
//  Multi-head attention (self + cross share the same op; only inputs differ)
// ═══════════════════════════════════════════════════════════════════════

/// Multi-head attention with Q from `q_in` and K, V from `kv_in`.
///
/// `q_in`  and `kv_in` are both `[B, S, D]` (S may differ between them).
/// `attn.in_proj_weight [3D, D]` is split into three `[D, D]` blocks:
///   Q from row range [0, D), K from [D, 2D), V from [2D, 3D).
/// The packed in_proj_bias [3D] is split the same way.
///
/// Returns the output `[B, q_seq_len, D]` after the out projection.
pub fn mha(
    q_in: &[f32],
    q_shape: [usize; 3],
    kv_in: &[f32],
    kv_shape: [usize; 3],
    attn: &MhaWeights,
) -> (Vec<f32>, [usize; 3]) {
    let d = attn.d_model;
    let h = attn.n_heads;
    let d_head = d / h;
    assert_eq!(d % h, 0, "mha: d_model must be divisible by n_heads");
    assert_eq!(q_shape[2], d, "mha: q last dim must equal d_model");
    assert_eq!(kv_shape[2], d, "mha: kv last dim must equal d_model");

    let [b, q_seq, _qd] = q_shape;
    let [_kb, kv_seq, _kd] = kv_shape;

    // ─── 1. Project Q from q_in ────────────────────────────────────────────
    let q_w: Vec<f32> = attn.in_proj_weight[0..d * d].to_vec();
    let q_b: Vec<f32> = attn.in_proj_bias[0..d].to_vec();
    let q_w_struct = Weight2D {
        data: q_w,
        rows: d,
        cols: d,
    };
    let (q_proj, _) = linear(q_in, q_shape, &q_w_struct, &q_b);
    // q_proj: [B, q_seq, D]

    // ─── 2. Project K, V from kv_in ────────────────────────────────────────
    let k_w: Vec<f32> = attn.in_proj_weight[d * d..2 * d * d].to_vec();
    let v_w: Vec<f32> = attn.in_proj_weight[2 * d * d..3 * d * d].to_vec();
    let k_b: Vec<f32> = attn.in_proj_bias[d..2 * d].to_vec();
    let v_b: Vec<f32> = attn.in_proj_bias[2 * d..3 * d].to_vec();
    let k_w_struct = Weight2D {
        data: k_w,
        rows: d,
        cols: d,
    };
    let v_w_struct = Weight2D {
        data: v_w,
        rows: d,
        cols: d,
    };
    let (k_proj, _) = linear(kv_in, kv_shape, &k_w_struct, &k_b);
    let (v_proj, _) = linear(kv_in, kv_shape, &v_w_struct, &v_b);
    // k_proj, v_proj: [B, kv_seq, D]

    // ─── 3. Compute attention: Q @ K^T / sqrt(d_head), softmax, @ V ───────
    let scale = 1.0 / (d_head as f32).sqrt();
    let mut out = vec![0.0f32; b * q_seq * d];

    for bi in 0..b {
        // For each head:
        for hi in 0..h {
            // Gather Q, K, V slices for this head: [q_seq, d_head] / [kv_seq, d_head].
            // Parallel over sequence positions (each copies d_head contiguous values).
            let mut q_head = vec![0.0f32; q_seq * d_head];
            let mut k_head = vec![0.0f32; kv_seq * d_head];
            let mut v_head = vec![0.0f32; kv_seq * d_head];
            q_head.par_chunks_mut(d_head).enumerate().for_each(|(si, row)| {
                let base = (bi * q_seq + si) * d + hi * d_head;
                for dh in 0..d_head {
                    row[dh] = q_proj[base + dh];
                }
            });
            k_head
                .par_chunks_mut(d_head)
                .zip(v_head.par_chunks_mut(d_head))
                .enumerate()
                .for_each(|(si, (krow, vrow))| {
                    let base = (bi * kv_seq + si) * d + hi * d_head;
                    for dh in 0..d_head {
                        krow[dh] = k_proj[base + dh];
                        vrow[dh] = v_proj[base + dh];
                    }
                });

            // scores[qi,ki] = sum_dh Q[qi,dh]·K[ki,dh]  →  [q_seq, kv_seq]
            // gemm: lhs=q_head[q_seq,d_head], rhs=k_head^T (k_head [kv_seq,d_head]).
            let mut scores = vec![0.0f32; q_seq * kv_seq];
            unsafe {
                gemm(
                    q_seq, kv_seq, d_head,
                    scores.as_mut_ptr(),
                    1, kv_seq as isize,
                    false,
                    q_head.as_ptr(), 1, d_head as isize,
                    k_head.as_ptr(), d_head as isize, 1,
                    scale, 0.0,
                    false, false, false,
                    Parallelism::Rayon(0),
                );
            }

            // Softmax along kv_seq axis — parallel over query rows.
            scores.par_chunks_mut(kv_seq).for_each(|row| {
                let mut max_v = f32::NEG_INFINITY;
                for &v in row.iter() {
                    if v > max_v {
                        max_v = v;
                    }
                }
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max_v).exp();
                    sum += *v;
                }
                let inv_sum = 1.0 / sum;
                for v in row.iter_mut() {
                    *v *= inv_sum;
                }
            });

            // out_head[qi,dh] = sum_ki scores[qi,ki]·V[ki,dh]  →  [q_seq, d_head]
            // gemm: lhs=scores[q_seq,kv_seq], rhs=v_head (non-transposed [kv_seq,d_head]).
            let mut out_head = vec![0.0f32; q_seq * d_head];
            unsafe {
                gemm(
                    q_seq, d_head, kv_seq,
                    out_head.as_mut_ptr(),
                    1, d_head as isize,
                    false,
                    scores.as_ptr(), 1, kv_seq as isize,
                    v_head.as_ptr(), 1, d_head as isize,
                    1.0, 0.0,
                    false, false, false,
                    Parallelism::Rayon(0),
                );
            }
            // Scatter back into [B, q_seq, D] at head slot `hi`. (Serial — the
            // writes are strided across the shared `out` buffer, so a par
            // closure can't borrow it; this is small: q_seq*d_head per head.)
            for qi in 0..q_seq {
                for dh in 0..d_head {
                    out[(bi * q_seq + qi) * d + hi * d_head + dh] = out_head[qi * d_head + dh];
                }
            }
        }
    }

    // ─── 4. Output projection: [B, q_seq, D] @ W_out^T + bias ─────────────
    let out_w = Weight2D {
        data: attn.out_proj_weight.clone(),
        rows: d,
        cols: d,
    };
    linear(&out, [b, q_seq, d], &out_w, &attn.out_proj_bias)
}

/// Convenience: self-attention where Q, K, V all come from the same input.
pub fn mha_self(x: &[f32], shape: [usize; 3], attn: &MhaWeights) -> (Vec<f32>, [usize; 3]) {
    mha(x, shape, x, shape, attn)
}

// ═══════════════════════════════════════════════════════════════════════
//  Transformer layer forward (self + cross)
// ═══════════════════════════════════════════════════════════════════════

/// Self-attention transformer layer forward (matches burn
/// `SelfAttentionLayer::forward`).
///
/// Input/output: `x [B, S, D]`.
///
///   residual = x
///   x_n = layernorm(x)
///   attn_out = mha_self(x_n)
///   x = residual + γ₁ · attn_out
///
///   residual = x
///   x_n = layernorm(x)
///   ffn = linear2(gelu(linear1(x_n) + linear1_bias) + linear2_bias)
///   x = residual + γ₂ · ffn
pub fn self_attn_layer_forward(
    x: &[f32],
    shape: [usize; 3],
    layer: &crate::model::SelfAttnLayer,
) -> (Vec<f32>, [usize; 3]) {
    let [b, s, d] = shape;
    assert_eq!(d, layer.norm1.dim, "self_attn d_model mismatch");
    assert_eq!(d, layer.attn.d_model, "self_attn attn d_model mismatch");

    // ─── Block 1: self-attention ───────────────────────────────────────────
    let mut x_n = x.to_vec();
    layernorm(&mut x_n, shape, &layer.norm1);
    let (attn_out, _) = mha_self(&x_n, shape, &layer.attn);
    let mut attn_scaled = attn_out;
    layer_scale_last(&mut attn_scaled, shape, &layer.gamma_1);
    let mut x_after_attn = x.to_vec();
    for i in 0..b * s * d {
        x_after_attn[i] += attn_scaled[i];
    }

    // ─── Block 2: FFN ─────────────────────────────────────────────────────
    let mut x_n2 = x_after_attn.clone();
    layernorm(&mut x_n2, shape, &layer.norm2);
    let (h1, h1_shape) = linear(&x_n2, shape, &layer.linear1, &layer.linear1_bias);
    let (mut h1_g, h1_g_shape) = (h1, h1_shape);
    gelu(&mut h1_g);
    let (h2, h2_shape) = linear(&h1_g, h1_g_shape, &layer.linear2, &layer.linear2_bias);
    let (mut h2_scaled, _) = (h2, h2_shape);
    layer_scale_last(&mut h2_scaled, shape, &layer.gamma_2);

    let mut out = x_after_attn;
    for i in 0..b * s * d {
        out[i] += h2_scaled[i];
    }

    // ─── Block 3: MyGroupNorm(1, d_model) — normalize over (S, D) per batch ─
    // swap to [B, D, S], apply groupnorm1, swap back.
    let mut swapped = vec![0.0f32; b * d * s];
    for bi in 0..b {
        for si in 0..s {
            for di in 0..d {
                swapped[bi * d * s + di * s + si] = out[bi * s * d + si * d + di];
            }
        }
    }
    groupnorm1(&mut swapped, [b, d, s], &layer.norm_out);
    for bi in 0..b {
        for si in 0..s {
            for di in 0..d {
                out[bi * s * d + si * d + di] = swapped[bi * d * s + di * s + si];
            }
        }
    }

    (out, shape)
}

/// Cross-attention transformer layer forward (matches burn
/// `CrossAttentionLayer::forward`).
///
/// `query` and `cross` are both `[B, S, D]` (S may differ).
///
///   residual = query
///   q_n = layernorm(query)
///   kv_n = layernorm(cross)
///   attn_out = mha(q_n, kv_n)
///   x = residual + γ₁ · attn_out
///
///   residual = x
///   x_n = layernorm(x)
///   ffn = linear2(gelu(linear1(x_n) + linear1_bias) + linear2_bias)
///   x = residual + γ₂ · ffn
pub fn cross_attn_layer_forward(
    query: &[f32],
    q_shape: [usize; 3],
    cross: &[f32],
    c_shape: [usize; 3],
    layer: &crate::model::CrossAttnLayer,
) -> (Vec<f32>, [usize; 3]) {
    let [b, s, d] = q_shape;
    assert_eq!(d, layer.norm1.dim, "cross_attn d_model mismatch");

    // ─── Block 1: cross-attention ─────────────────────────────────────────
    let mut q_n = query.to_vec();
    layernorm(&mut q_n, q_shape, &layer.norm1);
    let mut kv_n = cross.to_vec();
    layernorm(&mut kv_n, c_shape, &layer.norm2);
    let (attn_out, _) = mha(&q_n, q_shape, &kv_n, c_shape, &layer.attn);
    let mut attn_scaled = attn_out;
    layer_scale_last(&mut attn_scaled, q_shape, &layer.gamma_1);

    let mut x_after_attn = query.to_vec();
    for i in 0..b * s * d {
        x_after_attn[i] += attn_scaled[i];
    }

    // ─── Block 2: FFN ─────────────────────────────────────────────────────
    let mut x_n2 = x_after_attn.clone();
    layernorm(&mut x_n2, q_shape, &layer.norm3);
    let (h1, h1_shape) = linear(&x_n2, q_shape, &layer.linear1, &layer.linear1_bias);
    let (mut h1_g, h1_g_shape) = (h1, h1_shape);
    gelu(&mut h1_g);
    let (h2, h2_shape) = linear(&h1_g, h1_g_shape, &layer.linear2, &layer.linear2_bias);
    let (mut h2_scaled, _) = (h2, h2_shape);
    layer_scale_last(&mut h2_scaled, q_shape, &layer.gamma_2);

    let mut out = x_after_attn;
    for i in 0..b * s * d {
        out[i] += h2_scaled[i];
    }

    // ─── Block 3: MyGroupNorm(1, d_model) — normalize over (S, D) per batch ─
    let mut swapped = vec![0.0f32; b * d * s];
    for bi in 0..b {
        for si in 0..s {
            for di in 0..d {
                swapped[bi * d * s + di * s + si] = out[bi * s * d + si * d + di];
            }
        }
    }
    groupnorm1(&mut swapped, [b, d, s], &layer.norm_out);
    for bi in 0..b {
        for si in 0..s {
            for di in 0..d {
                out[bi * s * d + si * d + di] = swapped[bi * d * s + di * s + si];
            }
        }
    }

    (out, q_shape)
}

// ═══════════════════════════════════════════════════════════════════════
//  Cross-domain transformer top-level forward
// ═══════════════════════════════════════════════════════════════════════

/// Conv1d with k=1, pad=0, dilation=1 — i.e. a 1x1 conv, which is a per-position
/// linear. Wraps the existing `conv1d` to avoid paying im2col overhead.
pub fn conv1d_k1(
    x: &[f32],
    x_shape: [usize; 3], // [B, C_in, L]
    w: &crate::model::Conv1dWeight, // [C_out, C_in, 1]
    bias: &crate::model::Bias,
) -> (Vec<f32>, [usize; 3]) {
    conv1d(x, x_shape, w, bias, 0, 1)
}

/// Apply a 1x1 conv (channel resample) along the channel dim of a 4D tensor,
/// preserving spatial shape.
///
/// For 4-stem/ft, `freq [1, 384, Fr, T]` → `[1, 512, Fr, T]` via
/// `channel_upsampler` (a Conv1d [512, 384, 1] applied per-frequency-bin).
fn channel_upsample_4d(
    x: &[f32],
    x_shape: [usize; 4], // [B, C_in, Fr, T]
    w: &crate::model::Conv1dWeight,
    bias: &crate::model::Bias,
) -> (Vec<f32>, [usize; 4]) {
    // Conv1d k=1 on the per-frequency stream: treat each (b, fr) as a "batch"
    // and T as length. Reshape [B*C_in*Fr, 1, T] would be wasteful; instead
    // reuse the existing conv2d path by viewing it as conv2d with kH=1, kW=1.
    // But conv1d with k=1 is simplest — reshape and call conv1d_k1.
    let [b, c_in, fr, t] = x_shape;
    let bs = b * fr;
    // Reshape [B, C_in, Fr, T] (row-major) → [B*Fr, C_in, T] by swapping (fr, c_in)
    // Source layout: x[b][c][fr][t], index = ((b*c + c)*fr + fr)*t + t
    // Dest layout: y[br][c][t], index = (br*c + c)*t + t
    // where br = b*fr + fr.
    // Permute: src index (b, c, fr, t) → dst index (b*fr + fr, c, t)
    let mut reshaped = vec![0.0f32; bs * c_in * t];
    for bi in 0..b {
        for ci in 0..c_in {
            for fri in 0..fr {
                for ti in 0..t {
                    let src = ((bi * c_in + ci) * fr + fri) * t + ti;
                    let dst = ((bi * fr + fri) * c_in + ci) * t + ti;
                    reshaped[dst] = x[src];
                }
            }
        }
    }
    // Apply conv1d k=1.
    let (out, _) = conv1d_k1(&reshaped, [bs, c_in, t], w, bias);
    // out: [bs, c_out, t] = [B*Fr, C_out, T]
    let c_out = w.out_ch;
    // Reshape back [B*Fr, C_out, T] → [B, C_out, Fr, T].
    // Permute: src (br, co, t) → dst (b, co, fr, t) where br = b*fr + fr.
    let mut unflat = vec![0.0f32; b * c_out * fr * t];
    for bi in 0..b {
        for fri in 0..fr {
            for co in 0..c_out {
                for ti in 0..t {
                    let src = ((bi * fr + fri) * c_out + co) * t + ti;
                    let dst = ((bi * c_out + co) * fr + fri) * t + ti;
                    unflat[dst] = out[src];
                }
            }
        }
    }
    (unflat, [b, c_out, fr, t])
}

/// Channel downsample from `[B, C_in, Fr, T]` to `[B, C_out, Fr, T]` (4D).
/// Symmetric to `channel_upsample_4d` but with the resampling weights.
fn channel_downsample_4d(
    x: &[f32],
    x_shape: [usize; 4], // [B, C_in, Fr, T]
    w: &crate::model::Conv1dWeight,
    bias: &crate::model::Bias,
) -> (Vec<f32>, [usize; 4]) {
    channel_upsample_4d(x, x_shape, w, bias)
}

/// Channel upsample 3D `[B, C_in, T2]` → `[B, C_out, T2]`.
fn channel_upsample_3d(
    x: &[f32],
    x_shape: [usize; 3],
    w: &crate::model::Conv1dWeight,
    bias: &crate::model::Bias,
) -> (Vec<f32>, [usize; 3]) {
    conv1d_k1(x, x_shape, w, bias)
}

/// Channel downsample 3D `[B, C_in, T2]` → `[B, C_out, T2]`.
fn channel_downsample_3d(
    x: &[f32],
    x_shape: [usize; 3],
    w: &crate::model::Conv1dWeight,
    bias: &crate::model::Bias,
) -> (Vec<f32>, [usize; 3]) {
    conv1d_k1(x, x_shape, w, bias)
}

/// Top-level cross-domain transformer forward.
///
/// Inputs:
///   - `freq`: `[1, bottleneck_ch=384, Fr, T]`
///   - `time`: `[1, bottleneck_ch=384, T2]`
///
/// Outputs:
///   - `freq_out`: `[1, bottleneck_ch, Fr, T]`
///   - `time_out`: `[1, bottleneck_ch, T2]`
///
/// Mirrors `demucs-core/src/model/transformer.rs:CrossDomainTransformer::forward`.
pub fn cross_domain_transformer_forward(
    freq: &[f32],
    freq_shape: [usize; 4],
    time: &[f32],
    time_shape: [usize; 3],
    ct: &crate::model::CrossDomainTransformer,
) -> (
    Vec<f32>,
    [usize; 4],
    Vec<f32>,
    [usize; 3],
) {
    let [_b, _ch, fr, t] = freq_shape;
    let [_tb, _tc, t2] = time_shape;
    let d_model = ct.norm_in.dim;

    // ─── 1. Channel upsample freq (4D): [1, ch, Fr, T] → [1, d_model, Fr, T] ──
    let (freq_d, _freq_d_shape) = match (&ct.channel_upsampler, &ct.channel_upsampler_bias) {
        (Some(w), Some(b)) => {
            let bias = crate::model::Bias {
                data: b.clone(),
                len: b.len(),
            };
            channel_upsample_4d(freq, freq_shape, w, &bias)
        }
        _ => (freq.to_vec(), freq_shape),
    };

    // ─── 2. Channel upsample time (3D): [1, ch, T2] → [1, d_model, T2] ────
    let (time_d, _time_d_shape) = match (&ct.channel_upsampler_t, &ct.channel_upsampler_t_bias) {
        (Some(w), Some(b)) => {
            let bias = crate::model::Bias {
                data: b.clone(),
                len: b.len(),
            };
            channel_upsample_3d(time, time_shape, w, &bias)
        }
        _ => (time.to_vec(), time_shape),
    };

    // ─── 3. Flatten freq to sequence: [1, d_model, Fr, T] → [1, T*Fr, d_model] ──
    // Permute (0, 3, 2, 1): src (b, c, fr, t) → dst (b, t, fr, c)
    let mut freq_seq = vec![0.0f32; t * fr * d_model];
    for ti in 0..t {
        for fri in 0..fr {
            for ci in 0..d_model {
                let src = ci * fr * t + fri * t + ti;
                let dst = (ti * fr + fri) * d_model + ci;
                freq_seq[dst] = freq_d[src];
            }
        }
    }

    // ─── 4. Permute time: [1, d_model, T2] → [1, T2, d_model] ───────────
    let mut time_seq = vec![0.0f32; t2 * d_model];
    for ti in 0..t2 {
        for ci in 0..d_model {
            time_seq[ti * d_model + ci] = time_d[ci * t2 + ti];
        }
    }
    let freq_shape_seq = [1usize, t * fr, d_model];
    let time_shape_seq = [1usize, t2, d_model];

    // ─── 5. Input norms ───────────────────────────────────────────────
    let mut freq_seq_n = freq_seq.clone();
    layernorm(&mut freq_seq_n, freq_shape_seq, &ct.norm_in);
    let mut time_seq_n = time_seq.clone();
    layernorm(&mut time_seq_n, time_shape_seq, &ct.norm_in_t);

    // ─── 6. Sinusoidal positional embeddings (additive, after norm) ────
    // 2D for freq: shape (width=T, height=Fr, d_model), time-major.
    let freq_pe = sin_embed_2d(d_model, fr, t);
    // 1D for time: shape (T2, d_model).
    let time_pe = sin_embed_1d(t2, d_model);
    for i in 0..(t * fr * d_model) {
        freq_seq_n[i] += freq_pe[i];
    }
    for i in 0..(t2 * d_model) {
        time_seq_n[i] += time_pe[i];
    }
    let mut freq_seq = freq_seq_n;
    let mut time_seq = time_seq_n;

    // ─── 7. 5 transformer layers ──────────────────────────────────────
    for i in 0..crate::T_LAYERS {
        match (&ct.layers[i], &ct.layers_t[i]) {
            (
                crate::model::TransformerLayerWeights {
                    self_attn: Some(f_layer),
                    ..
                },
                crate::model::TransformerLayerWeights {
                    self_attn: Some(t_layer),
                    ..
                },
            ) => {
                // Self-attention for both freq and time, independently.
                let (f, _) = self_attn_layer_forward(&freq_seq, freq_shape_seq, f_layer);
                freq_seq = f;
                let (t_out, _) = self_attn_layer_forward(&time_seq, time_shape_seq, t_layer);
                time_seq = t_out;
            }
            (
                crate::model::TransformerLayerWeights {
                    cross_attn: Some(f_layer),
                    ..
                },
                crate::model::TransformerLayerWeights {
                    cross_attn: Some(t_layer),
                    ..
                },
            ) => {
                // Cross-attention: freq queries time, time queries freq.
                let (f_new, _) = cross_attn_layer_forward(
                    &freq_seq, freq_shape_seq, &time_seq, time_shape_seq, f_layer,
                );
                let (t_new, _) = cross_attn_layer_forward(
                    &time_seq, time_shape_seq, &freq_seq, freq_shape_seq, t_layer,
                );
                freq_seq = f_new;
                time_seq = t_new;
            }
            _ => panic!(
                "transformer layer {i}: freq and time layer types must match (both self or both cross)"
            ),
        }
    }

    // ─── 8. Unflatten freq: [1, T*Fr, d_model] → [1, d_model, Fr, T] ──
    // Permute (0, 2, 1, 3): src (b, t, fr, c) → dst (b, c, fr, t)
    let mut freq_unflat = vec![0.0f32; d_model * fr * t];
    for ti in 0..t {
        for fri in 0..fr {
            for ci in 0..d_model {
                let src = (ti * fr + fri) * d_model + ci;
                let dst = ci * fr * t + fri * t + ti;
                freq_unflat[dst] = freq_seq[src];
            }
        }
    }

    // ─── 9. Permute time: [1, T2, d_model] → [1, d_model, T2] ────────
    let mut time_unflat = vec![0.0f32; d_model * t2];
    for ti in 0..t2 {
        for ci in 0..d_model {
            time_unflat[ci * t2 + ti] = time_seq[ti * d_model + ci];
        }
    }

    // ─── 10. Channel downsample ──────────────────────────────────────
    let (freq_out, freq_out_shape) = match (&ct.channel_downsampler, &ct.channel_downsampler_bias) {
        (Some(w), Some(b)) => {
            let bias = crate::model::Bias {
                data: b.clone(),
                len: b.len(),
            };
            let freq_4d = [1usize, d_model, fr, t];
            channel_downsample_4d(&freq_unflat, freq_4d, w, &bias)
        }
        _ => (freq_unflat, [1, d_model, fr, t]),
    };
    let (time_out, time_out_shape) = match (&ct.channel_downsampler_t, &ct.channel_downsampler_t_bias) {
        (Some(w), Some(b)) => {
            let bias = crate::model::Bias {
                data: b.clone(),
                len: b.len(),
            };
            let time_3d = [1usize, d_model, t2];
            channel_downsample_3d(&time_unflat, time_3d, w, &bias)
        }
        _ => (time_unflat, [1, d_model, t2]),
    };

    (freq_out, freq_out_shape, time_out, time_out_shape)
}

// ═══════════════════════════════════════════════════════════════════════
//  HDecLayer / TDecLayer forward
// ═══════════════════════════════════════════════════════════════════════

/// Frequency decoder layer forward (mirrors
/// `demucs-core::model::conv::HDecLayer::forward`).
///
/// Inputs:
///   - `x`: `[B, chin, Fr, T]`
///   - `skip`: `[B, chin, Fr_skip, T]` (may have a different Fr if encoder/decoder shapes don't match)
///   - `freq_target`: trim the output freq dim to this many bins
///
/// Output: `[B, chout, freq_target, T]`.
pub fn hdec_layer_forward(
    x: &[f32],
    x_shape: [usize; 4], // [B, C_in, Fr, T]
    skip: &[f32],
    skip_shape: [usize; 4],
    freq_target: usize,
    layer: &HDecLayer,
) -> (Vec<f32>, [usize; 4]) {
    let [b, chin, fr, t] = x_shape;
    assert_eq!(chin, layer.rewrite.in_ch, "hdec rewrite in_ch mismatch");
    assert_eq!(2 * chin, layer.rewrite.out_ch, "hdec rewrite out_ch mismatch");

    // ─── 1. Residual: x = x + skip (sizes should match here) ────────────
    assert_eq!(x_shape, skip_shape, "hdec: x and skip must have same shape for residual add");
    let mut h: Vec<f32> = x.iter().zip(skip.iter()).map(|(a, b)| a + b).collect();

    // ─── 2. Conv2d(3,3) pad=(1,1) ──────────────────────────────────────
    let (h2, h2_shape) = conv2d(
        &h,
        x_shape,
        &layer.rewrite,
        &layer.rewrite_bias,
        1,
        1,
        1,
        1,
    );
    // ─── 3. GLU on dim=1: 2*chin → chin (4D in-place) ─────────────────
    let [b2, c2x, fr2, t2] = h2_shape;
    assert_eq!(c2x, 2 * chin, "hdec GLU channel must be 2*chin");
    let c_glu = chin;
    let mut h3 = vec![0.0f32; b2 * c_glu * fr2 * t2];
    for bi in 0..b2 {
        for ci in 0..c_glu {
            for fri in 0..fr2 {
                for ti in 0..t2 {
                    let a = h2[((bi * c2x + ci) * fr2 + fri) * t2 + ti];
                    let b_val = h2[((bi * c2x + c_glu + ci) * fr2 + fri) * t2 + ti];
                    let sig = 1.0 / (1.0 + (-b_val).exp());
                    h3[((bi * c_glu + ci) * fr2 + fri) * t2 + ti] = a * sig;
                }
            }
        }
    }
    let h3_shape = [b2, c_glu, fr2, t2];
    // ─── 4. DConv (per-frequency flatten) ─────────────────────────────
    let [_, c3, fr3, t3] = h3_shape;
    let mut flat = vec![0.0f32; b * fr3 * c3 * t3];
    for bi in 0..b {
        for ci in 0..c3 {
            for fri in 0..fr3 {
                for ti in 0..t3 {
                    flat[((bi * fr3 + fri) * c3 + ci) * t3 + ti] =
                        h3[((bi * c3 + ci) * fr3 + fri) * t3 + ti];
                }
            }
        }
    }
    let (dconv_out, dconv_shape) = dconv_forward(&flat, [b * fr3, c3, t3], &layer.dconv);
    // Reshape back [B*Fr, C, T] → [B, C, Fr, T]
    let [n, c4, t4] = dconv_shape;
    assert_eq!(n, b * fr3);
    let mut unflat = vec![0.0f32; b * c4 * fr3 * t4];
    for bi in 0..b {
        for fri in 0..fr3 {
            for ci in 0..c4 {
                for ti in 0..t4 {
                    unflat[((bi * c4 + ci) * fr3 + fri) * t4 + ti] =
                        dconv_out[((bi * fr3 + fri) * c4 + ci) * t4 + ti];
                }
            }
        }
    }

    // ─── 5. ConvTranspose2d([8,1], stride=[4,1], pad=[2,0]) ────────────
    let (mut h5, h5_shape) = conv_transpose2d(
        &unflat,
        [b, c4, fr3, t4],
        &layer.conv_tr,
        &layer.conv_tr_bias,
        2,
        0,
        4,
        1,
    );
    // ─── 6. Trim freq dim if > freq_target ────────────────────────────
    let [_, _, h5_fr, _] = h5_shape;
    if h5_fr > freq_target {
        let [b, c, _, t] = h5_shape;
        let mut trimmed = vec![0.0f32; b * c * freq_target * t];
        for bi in 0..b {
            for ci in 0..c {
                for fri in 0..freq_target {
                    for ti in 0..t {
                        trimmed[((bi * c + ci) * freq_target + fri) * t + ti] =
                            h5[((bi * c + ci) * h5_fr + fri) * t + ti];
                    }
                }
            }
        }
        h5 = trimmed;
    }
    // Report the actual output shape: either the trimmed shape (when
    // h5_fr > freq_target) or h5_shape unchanged. The previous version
    // always reported [b, c, freq_target, t], which is a lie when the
    // natural conv-transpose output is smaller than freq_target (e.g. test
    // inputs with fr < 2048). That lie caused downstream
    // `denormalize_freq` to walk past the actual buffer.
    let out_shape = if h5_fr > freq_target {
        let [b, c, _, t] = h5_shape;
        [b, c, freq_target, t]
    } else {
        h5_shape
    };
    // ─── 7. Optional GELU (skip on last layer) ─────────────────────────
    if !layer.last {
        gelu(&mut h5);
    }

    (h5, out_shape)
}

/// Time decoder layer forward.
///
/// Inputs:
///   - `x`: `[B, chin, T]`
///   - `skip`: `[B, chin, T_skip]`
///   - `time_target`: trim output time dim to this
///
/// Output: `[B, chout, time_target]`.
pub fn tdec_layer_forward(
    x: &[f32],
    x_shape: [usize; 3],
    skip: &[f32],
    skip_shape: [usize; 3],
    time_target: usize,
    layer: &TDecLayer,
) -> (Vec<f32>, [usize; 3]) {
    let [b, chin, t] = x_shape;
    assert_eq!(chin, layer.rewrite.in_ch, "tdec rewrite in_ch mismatch");

    // ─── 1. Trim skip to match x.time (Python: skip[..., :x.shape[-1]]) ─
    let skip_t = skip_shape[2].min(t);
    let mut h: Vec<f32> = vec![0.0f32; b * chin * t];
    for bi in 0..b {
        for ci in 0..chin {
            for ti in 0..skip_t {
                h[(bi * chin + ci) * t + ti] =
                    x[(bi * chin + ci) * t + ti] + skip[(bi * chin + ci) * skip_shape[2] + ti];
            }
            // tail of x (beyond skip_t) just passes through
            for ti in skip_t..t {
                h[(bi * chin + ci) * t + ti] = x[(bi * chin + ci) * t + ti];
            }
        }
    }

    // ─── 2. Conv1d(3) pad=1 ────────────────────────────────────────────
    let (h2, h2_shape) = conv1d(
        &h,
        [b, chin, t],
        &layer.rewrite,
        &layer.rewrite_bias,
        1,
        1,
    );
    // ─── 3. GLU on dim=1: 2*chin → chin ───────────────────────────────
    let (h3, h3_shape) = glu_channel(&h2, h2_shape);
    // ─── 4. DConv (no flatten for time) ──────────────────────────────
    let (mut h4, h4_shape) = dconv_forward(&h3, h3_shape, &layer.dconv);
    // ─── 5. ConvTranspose1d: chin → chout, 4× upsample ─────────────────
    let (mut h5, h5_shape) = conv_transpose1d(
        &h4,
        h4_shape,
        &layer.conv_tr,
        &layer.conv_tr_bias,
        2,
        4,
    );
    // ─── 6. Trim time dim if > time_target ────────────────────────────
    let [_b, _c, h5_t] = h5_shape;
    if h5_t > time_target {
        let [b, c, _] = h5_shape;
        let mut trimmed = vec![0.0f32; b * c * time_target];
        for bi in 0..b {
            for ci in 0..c {
                for ti in 0..time_target {
                    trimmed[(bi * c + ci) * time_target + ti] =
                        h5[(bi * c + ci) * h5_t + ti];
                }
            }
        }
        h5 = trimmed;
    }
    // Report the actual output shape: either trimmed (when h5_t >
    // time_target) or h5_shape unchanged. The previous version always
    // reported `time_target`, which over-reports when the natural
    // conv-transpose output is smaller than the requested target.
    let out_shape = if h5_t > time_target {
        let [b, c, _] = h5_shape;
        [b, c, time_target]
    } else {
        h5_shape
    };
    // ─── 7. Optional GELU ──────────────────────────────────────────────
    if !layer.last {
        gelu(&mut h5);
    }

    (h5, out_shape)
}

/// Time encoder layer forward (mirrors
/// `demucs-core::model::conv::TEncLayer::forward`).
///
/// Input: `x [B, chin, T]`. Output: `x [B, chout, T_out]` where
/// `T_out = T_padded / STRIDE` (right-pad so length is divisible by 4).
pub fn tenc_layer_forward(
    x: &[f32],
    x_shape: [usize; 3],
    layer: &TEncLayer,
) -> (Vec<f32>, [usize; 3]) {
    let [b, _chin, t] = x_shape;
    let stride = 4; // STRIDE constant
    let chout = layer.conv.out_ch;

    // ─── 1. Right-pad so length is divisible by STRIDE ─────────────────
    let pad_right = if t % stride == 0 { 0 } else { stride - (t % stride) };
    let t_padded = t + pad_right;
    let mut x_padded = vec![0.0f32; b * _chin * t_padded];
    for bi in 0..b {
        for ci in 0.._chin {
            for ti in 0..t {
                x_padded[(bi * _chin + ci) * t_padded + ti] =
                    x[(bi * _chin + ci) * t + ti];
            }
            // tail is already 0
        }
    }

    // ─── 2. Conv1d(k=8, stride=4, pad=2) chin → chout ──────────────────
    let (h, h_shape) = conv1d_with_stride(
        &x_padded,
        [b, _chin, t_padded],
        &layer.conv,
        &layer.conv_bias,
        2,
        stride,
        1,
    );
    // h_shape: [B, chout, t_padded/stride]

    // ─── 3. GELU ───────────────────────────────────────────────────────
    let mut h2 = h;
    gelu(&mut h2);

    // ─── 4. DConv (1D, no flatten) ────────────────────────────────────
    let (mut h3, h3_shape) = dconv_forward(&h2, h_shape, &layer.dconv);

    // ─── 5. Rewrite Conv1d(k=1) chout → 2*chout ──────────────────────
    // Reshape via conv1d with k=1, pad=0, dilation=1.
    let (h4, h4_shape) = conv1d(
        &h3,
        h3_shape,
        &layer.rewrite,
        &layer.rewrite_bias,
        0,
        1,
    );

    // ─── 6. GLU on dim=1: 2*chout → chout ─────────────────────────────
    let (mut h5, h5_shape) = glu_channel(&h4, h4_shape);
    assert_eq!(h5_shape[1], chout, "tenc GLU output must be chout");
    let _ = h3; // silence unused

    (h5, h5_shape)
}

// ═══════════════════════════════════════════════════════════════════════
//  HTDemucs top-level forward
// ═══════════════════════════════════════════════════════════════════════

/// Per-batch mean and std of a tensor flattened across non-batch dims.
/// Returns `(mean, std)` each as `[B, 1, ...]` broadcastable against the
/// original shape (with the appropriate number of trailing 1s).
///
/// For 4D `[B, C, H, W]` → `mean, std` are each `[B, 1, 1, 1]`.
/// For 3D `[B, C, L]`   → `mean, std` are each `[B, 1, 1]`.
fn per_batch_mean_std<const N: usize>(x: &[f32], shape: [usize; N]) -> (Vec<f32>, Vec<f32>) {
    let b = shape[0];
    let per_batch: usize = shape[1..].iter().product();
    let mut mean = vec![0.0f32; b];
    let mut std = vec![0.0f32; b];
    for bi in 0..b {
        let slice = &x[bi * per_batch..(bi + 1) * per_batch];
        let m: f32 = slice.iter().sum::<f32>() / per_batch as f32;
        let v: f32 = slice.iter().map(|x| (x - m).powi(2)).sum::<f32>() / per_batch as f32;
        mean[bi] = m;
        std[bi] = v.sqrt();
    }
    (mean, std)
}

/// Normalize a 4D `[B, C, H, W]` tensor along the per-batch mean/std:
/// `x = (x - mean) / (std + 1e-5)`. Returns normalized data and the
/// `[B, 1, 1, 1]` mean/std for later denormalization.
pub fn normalize_freq(
    x: &[f32],
    shape: [usize; 4],
) -> (Vec<f32>, [usize; 4], Vec<f32>, [usize; 4], Vec<f32>, [usize; 4]) {
    let (mean, std) = per_batch_mean_std(x, shape);
    let mut out = x.to_vec();
    let b = shape[0];
    let per_batch: usize = shape[1..].iter().product();
    for bi in 0..b {
        let m = mean[bi];
        let s = std[bi] + 1e-5;
        for j in 0..per_batch {
            out[bi * per_batch + j] = (out[bi * per_batch + j] - m) / s;
        }
    }
    (out, shape, mean, [b, 1, 1, 1], std, [b, 1, 1, 1])
}

/// Normalize a 3D `[B, C, L]` tensor analogously.
pub fn normalize_time(
    x: &[f32],
    shape: [usize; 3],
) -> (Vec<f32>, [usize; 3], Vec<f32>, [usize; 3], Vec<f32>, [usize; 3]) {
    let (mean, std) = per_batch_mean_std(x, shape);
    let mut out = x.to_vec();
    let b = shape[0];
    let per_batch: usize = shape[1..].iter().product();
    for bi in 0..b {
        let m = mean[bi];
        let s = std[bi] + 1e-5;
        for j in 0..per_batch {
            out[bi * per_batch + j] = (out[bi * per_batch + j] - m) / s;
        }
    }
    (out, shape, mean, [b, 1, 1], std, [b, 1, 1])
}

/// Denormalize a 4D freq tensor: `x = x * std + mean` (raw std, NOT std+eps).
///
/// NOTE: per_batch is derived from `x.len()/b`, NOT from `shape[1..]`. The
/// forward output has n_sources*4 channels while the input shape has 4 —
/// using shape[1..] would only denormalize the first n_sources fraction
/// of channels (e.g. only stem 0, leaving vocals at channels 12-15 in
/// normalized space → iSTFT explodes).
pub fn denormalize_freq(x: &mut [f32], shape: [usize; 4], mean: &[f32], std: &[f32]) {
    let b = shape[0];
    let per_batch = x.len() / b;
    for bi in 0..b {
        let m = mean[bi];
        let s = std[bi];
        for j in 0..per_batch {
            x[bi * per_batch + j] = x[bi * per_batch + j] * s + m;
        }
    }
}

/// Denormalize a 3D time tensor.
pub fn denormalize_time(x: &mut [f32], shape: [usize; 3], mean: &[f32], std: &[f32]) {
    let b = shape[0];
    let per_batch = x.len() / b; // actual size (output has n_sources*2 ch, input has 2)
    for bi in 0..b {
        let m = mean[bi];
        let s = std[bi];
        for j in 0..per_batch {
            x[bi * per_batch + j] = x[bi * per_batch + j] * s + m;
        }
    }
}

/// HTDemucs top-level forward (mirrors
/// `demucs-core::model::htdemucs::HTDemucs::forward_with_listener`).
///
/// Inputs:
///   - `freq`: `[1, 4, 2048, T]` CaC (4 channels = 2 stereo × 2 (real, imag))
///   - `time`: `[1, 2, samples]` stereo waveform
///
/// Outputs:
///   - `freq_out`: `[1, n_sources*4, 2048, T]`
///   - `time_out`: `[1, n_sources*2, samples]`
pub fn htdemucs_forward(
    freq: &[f32],
    freq_shape: [usize; 4],
    time: &[f32],
    time_shape: [usize; 3],
    model: &HTDemucs,
) -> (Vec<f32>, [usize; 4], Vec<f32>, [usize; 3]) {
    let depth = model.encoders.len();

    // ─── 1. Normalize inputs ───────────────────────────────────────────
    let (freq_n, freq_shape, freq_mean, _, freq_std, _) =
        normalize_freq(freq, freq_shape);
    let (time_n, time_shape, time_mean, _, time_std, _) =
        normalize_time(time, time_shape);

    // ─── 2. Freq encoder chain (4 layers) ──────────────────────────────
    let mut freq_skips: Vec<(Vec<f32>, [usize; 4])> = Vec::with_capacity(depth);
    // Layer 0 first, then apply freq_emb, then save skip.
    let (h, h_shape) = henc_layer_forward(&freq_n, freq_shape, &model.encoders[0]);
    let mut freq = h;
    let mut freq_shape = h_shape;
    // Apply freq_emb AFTER encoder[0], BEFORE saving skip (matches burn).
    let [b, c, fr, t] = freq_shape;
    let emb = &model.freq_emb;
    let scale = 2.0;
    for bi in 0..b {
        for ci in 0..c {
            for fi in 0..fr {
                let emb_val = emb.data[fi * emb.dim + ci] * scale;
                for ti in 0..t {
                    freq[((bi * c + ci) * fr + fi) * t + ti] += emb_val;
                }
            }
        }
    }
    freq_skips.push((freq.clone(), freq_shape));

    // Layers 1..depth-1
    for i in 1..depth {
        let (out, out_shape) = henc_layer_forward(&freq, freq_shape, &model.encoders[i]);
        freq = out;
        freq_shape = out_shape;
        freq_skips.push((freq.clone(), freq_shape));
    }

    // ─── 3. Time encoder chain (4 layers) ──────────────────────────────
    let mut time_skips: Vec<(Vec<f32>, [usize; 3])> = Vec::with_capacity(depth);
    let mut time_lengths: Vec<usize> = Vec::with_capacity(depth);
    let mut time = time_n;
    let mut time_shape = time_shape;
    for i in 0..depth {
        time_lengths.push(time_shape[2]);
        let (out, out_shape) = tenc_layer_forward(&time, time_shape, &model.tencoders[i]);
        time = out;
        time_shape = out_shape;
        time_skips.push((time.clone(), time_shape));
    }

    // ─── 4. Cross-domain Transformer ──────────────────────────────────
    let (mut freq, mut freq_shape, mut time, mut time_shape) = cross_domain_transformer_forward(
        &freq,
        freq_shape,
        &time,
        time_shape,
        &model.crosstransformer,
    );

    // ─── 5. Freq decoder chain (reverse order, with skips) ────────────
    // freq_dims[i] = Fr dim of freq_skips[i] (in push order: shallow→deep).
    let freq_dims: Vec<usize> = freq_skips.iter().map(|(_, s)| s[2]).collect();
    for i in 0..depth {
        let (skip, skip_shape) = freq_skips.pop().expect("freq skip stack");
        let target = if i + 1 < freq_dims.len() {
            freq_dims[freq_dims.len() - 2 - i]
        } else {
            crate::N_FFT / 2
        };
        let (out, out_shape) = hdec_layer_forward(
            &freq,
            freq_shape,
            &skip,
            skip_shape,
            target,
            &model.decoders[i],
        );
        freq = out;
        freq_shape = out_shape;
    }

    // ─── 6. Time decoder chain (reverse order, with skips) ────────────
    for i in 0..depth {
        let (skip, skip_shape) = time_skips.pop().expect("time skip stack");
        let target = time_lengths[time_lengths.len() - 1 - i];
        let (out, out_shape) = tdec_layer_forward(
            &time,
            time_shape,
            &skip,
            skip_shape,
            target,
            &model.tdecoders[i],
        );
        time = out;
        time_shape = out_shape;
    }

    // ─── 7. Denormalize outputs (raw std, not std+eps) ────────────────
    denormalize_freq(&mut freq, freq_shape, &freq_mean, &freq_std);
    denormalize_time(&mut time, time_shape, &time_mean, &time_std);

    (freq, freq_shape, time, time_shape)
}

/// Traced variant of `cross_domain_transformer_forward`. Same semantics, but
/// also returns per-layer (5 layers × 2 domains) intermediate tensors.
///
/// Layout of returned `Vec<(Vec<f32>, Vec<f32>)>`:
///   - index 0..T_LAYERS: (freq, time) after each layer's residual block
pub fn cross_domain_transformer_forward_traced(
    freq: &[f32],
    freq_shape: [usize; 4],
    time: &[f32],
    time_shape: [usize; 3],
    ct: &crate::model::CrossDomainTransformer,
) -> (
    Vec<f32>,
    [usize; 4],
    Vec<f32>,
    [usize; 3],
    Vec<(Vec<f32>, Vec<f32>)>,
) {
    let t_layers = crate::T_LAYERS;
    let (freq_d, _freq_d_shape) = match (&ct.channel_upsampler, &ct.channel_upsampler_bias) {
        (Some(w), Some(b)) => {
            let bias = crate::model::Bias {
                data: b.clone(),
                len: b.len(),
            };
            channel_upsample_4d(freq, freq_shape, w, &bias)
        }
        _ => (freq.to_vec(), freq_shape),
    };
    let (time_d, _time_d_shape) = match (&ct.channel_upsampler_t, &ct.channel_upsampler_t_bias) {
        (Some(w), Some(b)) => {
            let bias = crate::model::Bias {
                data: b.clone(),
                len: b.len(),
            };
            channel_upsample_3d(time, time_shape, w, &bias)
        }
        _ => (time.to_vec(), time_shape),
    };
    let [_, _, fr, t] = freq_shape;
    let d_model = ct.norm_in.dim;
    let mut freq_seq = vec![0.0f32; t * fr * d_model];
    for ti in 0..t {
        for fri in 0..fr {
            for ci in 0..d_model {
                let src = ci * fr * t + fri * t + ti;
                let dst = (ti * fr + fri) * d_model + ci;
                freq_seq[dst] = freq_d[src];
            }
        }
    }
    let [_tb, _tc, t2] = time_shape;
    let mut time_seq = vec![0.0f32; t2 * d_model];
    for ti in 0..t2 {
        for ci in 0..d_model {
            time_seq[ti * d_model + ci] = time_d[ci * t2 + ti];
        }
    }
    let freq_shape_seq = [1usize, t * fr, d_model];
    let time_shape_seq = [1usize, t2, d_model];
    let mut freq_seq_n = freq_seq.clone();
    layernorm(&mut freq_seq_n, freq_shape_seq, &ct.norm_in);
    let mut time_seq_n = time_seq.clone();
    layernorm(&mut time_seq_n, time_shape_seq, &ct.norm_in_t);
    let freq_pe = sin_embed_2d(d_model, fr, t);
    let time_pe = sin_embed_1d(t2, d_model);
    for i in 0..(t * fr * d_model) {
        freq_seq_n[i] += freq_pe[i];
    }
    for i in 0..(t2 * d_model) {
        time_seq_n[i] += time_pe[i];
    }
    let mut freq_seq = freq_seq_n;
    let mut time_seq = time_seq_n;
    let mut trace: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(t_layers);
    for i in 0..t_layers {
        match (&ct.layers[i], &ct.layers_t[i]) {
            (
                crate::model::TransformerLayerWeights {
                    self_attn: Some(f_layer),
                    ..
                },
                crate::model::TransformerLayerWeights {
                    self_attn: Some(t_layer),
                    ..
                },
            ) => {
                let (f, _) = self_attn_layer_forward(&freq_seq, freq_shape_seq, f_layer);
                freq_seq = f;
                let (t_out, _) = self_attn_layer_forward(&time_seq, time_shape_seq, t_layer);
                time_seq = t_out;
            }
            (
                crate::model::TransformerLayerWeights {
                    cross_attn: Some(f_layer),
                    ..
                },
                crate::model::TransformerLayerWeights {
                    cross_attn: Some(t_layer),
                    ..
                },
            ) => {
                let (f_new, _) = cross_attn_layer_forward(
                    &freq_seq, freq_shape_seq, &time_seq, time_shape_seq, f_layer,
                );
                let (t_new, _) = cross_attn_layer_forward(
                    &time_seq, time_shape_seq, &freq_seq, freq_shape_seq, t_layer,
                );
                freq_seq = f_new;
                time_seq = t_new;
            }
            _ => panic!(
                "transformer layer {i}: freq and time layer types must match"
            ),
        }
        trace.push((freq_seq.clone(), time_seq.clone()));
    }
    let mut freq_unflat = vec![0.0f32; d_model * fr * t];
    for ti in 0..t {
        for fri in 0..fr {
            for ci in 0..d_model {
                let src = (ti * fr + fri) * d_model + ci;
                let dst = ci * fr * t + fri * t + ti;
                freq_unflat[dst] = freq_seq[src];
            }
        }
    }
    let mut time_unflat = vec![0.0f32; d_model * t2];
    for ti in 0..t2 {
        for ci in 0..d_model {
            time_unflat[ci * t2 + ti] = time_seq[ti * d_model + ci];
        }
    }
    let (freq_out, freq_out_shape) = match (&ct.channel_downsampler, &ct.channel_downsampler_bias) {
        (Some(w), Some(b)) => {
            let bias = crate::model::Bias {
                data: b.clone(),
                len: b.len(),
            };
            let freq_4d = [1usize, d_model, fr, t];
            channel_downsample_4d(&freq_unflat, freq_4d, w, &bias)
        }
        _ => (freq_unflat, [1, d_model, fr, t]),
    };
    let (time_out, time_out_shape) = match (&ct.channel_downsampler_t, &ct.channel_downsampler_t_bias) {
        (Some(w), Some(b)) => {
            let bias = crate::model::Bias {
                data: b.clone(),
                len: b.len(),
            };
            let time_3d = [1usize, d_model, t2];
            channel_downsample_3d(&time_unflat, time_3d, w, &bias)
        }
        _ => (time_unflat, [1, d_model, t2]),
    };
    (freq_out, freq_out_shape, time_out, time_out_shape, trace)
}

// ═══════════════════════════════════════════════════════════════════════
//  Extract per-stem WAV via iSTFT + time add
// ═══════════════════════════════════════════════════════════════════════

/// Extract all stems from `htdemucs_forward` outputs (mirrors
/// `demucs-core/src/lib.rs::extract_all_stems`).
///
/// `freq_out` has shape `[1, n_sources*4, n_bins, n_frames]`; we trim it to
/// `n_frames` frames and split into per-stem CaC channel groups of 4. For each
/// stem, we iSTFT both left and right channels and add the corresponding
/// time-domain stem (per-sample sum).
///
/// Returns one `Stem` per source.
pub fn extract_stems(
    freq_out: &[f32],
    freq_out_shape: [usize; 4],
    time_out: &[f32],
    time_out_shape: [usize; 3],
    n_frames: usize,
    padded_len: usize,
    n_samples: usize,
    stft: &mut crate::dsp::stft::Stft,
) -> Vec<crate::Stem> {
    use crate::metadata::StemId;
    let [_, total_ch, n_bins, _t_full] = freq_out_shape;
    assert_eq!(total_ch % 4, 0, "freq_out channels must be n_sources*4");
    let n_sources = total_ch / 4;
    let ch_stride = n_bins * n_frames;
    let stem_stride = 4 * ch_stride;

    // Stem order: matches the model construction — Drums (0), Bass (1),
    // Other (2), Vocals (3) for htdemucs_ft; for 6-stem model: Drums, Bass,
    // Other, Vocals, Guitar, Piano.
    let stem_order: Vec<StemId> = match n_sources {
        4 => vec![
            StemId::Drums,
            StemId::Bass,
            StemId::Other,
            StemId::Vocals,
        ],
        6 => vec![
            StemId::Drums,
            StemId::Bass,
            StemId::Other,
            StemId::Vocals,
            StemId::Guitar,
            StemId::Piano,
        ],
        other => panic!("unsupported n_sources = {other}"),
    };

    let mut stems = Vec::with_capacity(n_sources);

    for s in 0..n_sources {
        let base = s * stem_stride;
        // Pack [reals, imags] for left and right into a contiguous buffer.
        let mut left_cac = vec![0.0f32; 2 * ch_stride];
        let mut right_cac = vec![0.0f32; 2 * ch_stride];
        // Source layout [b=0, c in [s*4 .. s*4+4], bin, frame]
        //   left real: c = s*4+0
        //   left imag: c = s*4+1
        //   right real: c = s*4+2
        //   right imag: c = s*4+3
        for bin in 0..n_bins {
            for frame in 0..n_frames {
                let src = base + 0 * ch_stride + bin * n_frames + frame;
                left_cac[bin * n_frames + frame] = freq_out[src];
                let src = base + 1 * ch_stride + bin * n_frames + frame;
                left_cac[ch_stride + bin * n_frames + frame] = freq_out[src];
                let src = base + 2 * ch_stride + bin * n_frames + frame;
                right_cac[bin * n_frames + frame] = freq_out[src];
                let src = base + 3 * ch_stride + bin * n_frames + frame;
                right_cac[ch_stride + bin * n_frames + frame] = freq_out[src];
            }
        }

        let left_spec = crate::dsp::cac::cac_data_to_complex(&left_cac, n_bins, n_frames);
        let right_spec = crate::dsp::cac::cac_data_to_complex(&right_cac, n_bins, n_frames);

        let left_wav = stft.inverse(&left_spec, padded_len).expect("iSTFT left failed");
        let right_wav = stft
            .inverse(&right_spec, padded_len)
            .expect("iSTFT right failed");

        // Time-domain stem.
        let [_tb, _tc, time_len] = time_out_shape;
        assert!(time_len >= n_samples, "time_out shorter than n_samples");
        let left_time = &time_out[s * 2 * time_len..s * 2 * time_len + n_samples];
        let right_time = &time_out[s * 2 * time_len + time_len..s * 2 * time_len + time_len + n_samples];

        let mut left = vec![0.0f32; n_samples];
        let mut right = vec![0.0f32; n_samples];
        for i in 0..n_samples {
            left[i] = left_wav[i] + left_time[i];
            right[i] = right_wav[i] + right_time[i];
        }
        stems.push(crate::Stem {
            id: stem_order[s],
            left,
            right,
        });
    }
    stems
}

