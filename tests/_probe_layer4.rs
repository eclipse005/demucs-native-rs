//! Isolated test: run self_attn_layer_forward on layers 0, 2, 4 with the
//! same input std to see which layer explodes.

use demucs_core_native::model::{SelfAttnLayer, HTDemucs};
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

fn stats(name: &str, x: &[f32]) {
    if x.is_empty() {
        return;
    }
    let n = x.len() as f32;
    let min_v = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_v = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    eprintln!("{name}: range=[{:.4}, {:.4}] mean={:.4} std={:.4}", min_v, max_v, mean, var.sqrt());
}

#[test]
#[ignore]
fn probe_naive_conv_transpose2d_known_answer() {
    // Simplest possible: B=1, C_in=1, C_out=1, H=1, W=1, kH=2, kW=1,
    // stride=1, pad=0. X = [[[[1.0]]]]; W (post-reorder) = [1.0].
    // Output: H_out = (1-1)*1 + 2 - 0 = 2, W_out = 1.
    // Y[0, 0, 0, 0] = X[0, 0, 0, 0] = 1 (dkh=0).
    // Y[0, 0, 1, 0] = X[0, 0, 0, 0] = 1 (dkh=1, ih=1-0+1=1 → wait stride=1
    //   pad=0, ih=oh*1 - 0 + dkh = oh + dkh. dkh=1: oh=0, ih=1 (oob, X only at ih=0).
    // Actually for oh=1, dkh=0: ih=1, oob. dkh=1: ih=2, oob.
    // So Y[1, 0] = 0. Y[0, 0] = 1.
    let x = vec![1.0f32];
    let w = vec![1.0f32];
    let w2 = vec![1.0f32, 1.0f32];
    let b = vec![0.0f32];
    let (out, shape) = naive_conv_transpose2d(
        &x,
        [1, 1, 1, 1],
        &w2,
        [1, 1, 2, 1],
        &b,
        0,
        0,
        1,
        1,
    );
    eprintln!(
        "naive [1,1,1,1] k=2 stride=1 pad=0: shape={:?} out={:?}",
        shape, out
    );

    // Same with stride=2: H_out = (1-1)*2 + 2 - 0 = 2, W_out = 1.
    // Y[0,0,0,0]: ih=oh*2-0+dkh, dkh=0: ih=0 (valid), X=1.0 → Y=1
    // Y[0,0,0,0]: dkh=1: ih=1 (oob) → 0
    // Y[0,0,1,0]: ih=2+dkh, all oob → 0
    let (out, shape) = naive_conv_transpose2d(
        &x,
        [1, 1, 1, 1],
        &w2,
        [1, 1, 2, 1],
        &b,
        0,
        0,
        2,
        1,
    );
    eprintln!(
        "naive [1,1,1,1] k=2 stride=2 pad=0: shape={:?} out={:?}",
        shape, out
    );

    // ops_cpu comparison: stride=1 (use conv_transpose1d wrapper for simplicity).
    let (out1, shape1) = demucs_core_native::ops_cpu::conv_transpose1d(
        &x,
        [1, 1, 1],
        &demucs_core_native::model::Conv1dWeight {
            data: w.clone(),
            out_ch: 1,
            in_ch: 1,
            k: 2,
        },
        &demucs_core_native::model::Bias {
            data: b.clone(),
            len: 1,
        },
        0,
        1,
    );
    eprintln!(
        "ops  1D [1,1] k=2 stride=1 pad=0: shape={:?} out={:?}",
        shape1, out1
    );
    let (out1, shape1) = demucs_core_native::ops_cpu::conv_transpose1d(
        &x,
        [1, 1, 1],
        &demucs_core_native::model::Conv1dWeight {
            data: w.clone(),
            out_ch: 1,
            in_ch: 1,
            k: 2,
        },
        &demucs_core_native::model::Bias {
            data: b.clone(),
            len: 1,
        },
        0,
        2,
    );
    eprintln!(
        "ops  1D [1,1] k=2 stride=2 pad=0: shape={:?} out={:?}",
        shape1, out1
    );
}

/// Naive conv2d for [B, C_in, H, W] with explicit shapes.
fn naive_conv2d_shaped(
    x: &[f32],
    x_shape: [usize; 4],
    w: &[f32],
    w_shape: [usize; 4], // [out_ch, in_ch, kH, kW]
    b: &[f32],
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> (Vec<f32>, [usize; 4]) {
    let [bs, c_in, h, width] = x_shape;
    let [out_ch, _, kh, kw] = w_shape;
    let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (width + 2 * pad_w - kw) / stride_w + 1;
    let mut out = vec![0.0f32; bs * out_ch * h_out * w_out];
    for bi in 0..bs {
        for oc in 0..out_ch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = b[oc];
                    for ic in 0..c_in {
                        for dkh in 0..kh {
                            for dkw in 0..kw {
                                let ih = (oh * stride_h + dkh) as isize - pad_h as isize;
                                let iw = (ow * stride_w + dkw) as isize - pad_w as isize;
                                if ih >= 0 && iw >= 0 && (ih as usize) < h && (iw as usize) < width {
                                    acc += w[((oc * c_in + ic) * kh + dkh) * kw + dkw]
                                        * x[((bi * c_in + ic) * h + ih as usize) * width + iw as usize];
                                }
                            }
                        }
                    }
                    out[((bi * out_ch + oc) * h_out + oh) * w_out + ow] = acc;
                }
            }
        }
    }
    (out, [bs, out_ch, h_out, w_out])
}

/// Naive ConvTranspose2d for [B, C_in, H, W] with weight [in, out, kH, kW]
/// (PyTorch layout, already reordered into [out, in, kH, kW] by the loader).
fn naive_conv_transpose2d(
    x: &[f32],
    x_shape: [usize; 4],
    w: &[f32],         // [out_ch, in_ch, kH, kW] (post-reorder)
    w_shape: [usize; 4],
    b: &[f32],
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> (Vec<f32>, [usize; 4]) {
    let [bs, c_in, h, width] = x_shape;
    let [out_ch, _, kh, kw] = w_shape;
    let h_out = (h - 1) * stride_h + kh - 2 * pad_h;
    let w_out = (width - 1) * stride_w + kw - 2 * pad_w;
    let mut out = vec![0.0f32; bs * out_ch * h_out * w_out];
    // Add bias.
    for bi in 0..bs {
        for oc in 0..out_ch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    out[((bi * out_ch + oc) * h_out + oh) * w_out + ow] = b[oc];
                }
            }
        }
    }
    // Scatter-add contributions. Reverse formulation: iterate over output
    // positions and find which input positions contribute.
    for bi in 0..bs {
        for oc in 0..out_ch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    for ic in 0..c_in {
                        for dkh in 0..kh {
                            for dkw in 0..kw {
                                // Reverse index: which input position
                                // (ih, iw) contributes to (oh, ow) via
                                // (dkh, dkw)?
                                //   ih * stride_h + dkh = oh + pad_h
                                //   ih = (oh + pad_h - dkh) / stride_h
                                // Valid only when (oh + pad_h - dkh) is
                                // divisible by stride_h.
                                let oh_p = oh as isize + pad_h as isize - dkh as isize;
                                let ow_p = ow as isize + pad_w as isize - dkw as isize;
                                if oh_p < 0
                                    || ow_p < 0
                                    || oh_p % stride_h as isize != 0
                                    || ow_p % stride_w as isize != 0
                                {
                                    continue;
                                }
                                let ih = (oh_p / stride_h as isize) as usize;
                                let iw = (ow_p / stride_w as isize) as usize;
                                if ih >= h || iw >= width {
                                    continue;
                                }
                                let x_val = x[((bi * c_in + ic) * h + ih) * width + iw];
                                out[((bi * out_ch + oc) * h_out + oh) * w_out + ow] +=
                                    w[((oc * c_in + ic) * kh + dkh) * kw + dkw] * x_val;
                            }
                        }
                    }
                }
            }
        }
    }
    (out, [bs, out_ch, h_out, w_out])
}

/// Naive GLU on dim=1 of [B, 2*C, H, W] -> [B, C, H, W]:
/// a = first half, b = second half, out = a * sigmoid(b).
fn naive_glu_4d(x: &[f32], shape: [usize; 4]) -> (Vec<f32>, [usize; 4]) {
    let [bs, c2, h, w] = shape;
    let c = c2 / 2;
    let mut out = vec![0.0f32; bs * c * h * w];
    for bi in 0..bs {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let a = x[((bi * c2 + ci) * h + hi) * w + wi];
                    let b = x[((bi * c2 + c + ci) * h + hi) * w + wi];
                    let sig = 1.0 / (1.0 + (-b).exp());
                    out[((bi * c + ci) * h + hi) * w + wi] = a * sig;
                }
            }
        }
    }
    (out, [bs, c, h, w])
}

/// Run hdec_layer_forward in isolation with the same input as the
/// hdec_forward.rs test, but using the naive reference ops.
#[test]
#[ignore]
fn probe_hdec_layer0_naive_vs_ops() {
    let model_path = std::path::PathBuf::from("../models/htdemucs_ft.safetensors");
    if !model_path.exists() {
        eprintln!("skipping");
        return;
    }
    let store = demucs_core_native::weights::WeightStore::load(&model_path).unwrap();
    let layer = demucs_core_native::model::HDecLayer::from_store(
        &store, "04573f0d", "decoder.0", 384, 192, false,
    )
    .expect("load HDecLayer decoder.0");

    // Same input as hdec_forward.rs.
    let b = 1;
    let chin = 384;
    let fr = 8;
    let t = 4;
    let x: Vec<f32> = (0..b * chin * fr * t)
        .map(|i| (i as f32 * 0.0017 - 0.5).sin() * 0.2)
        .collect();
    let skip: Vec<f32> = (0..b * chin * fr * t)
        .map(|i| (i as f32 * 0.0023 + 0.1).cos() * 0.1)
        .collect();
    let target = 32;

    // ─── Reference: ops_cpu ─────────────────────────────────────────
    let (ops_out, ops_shape) = demucs_core_native::ops_cpu::hdec_layer_forward(
        &x, [b, chin, fr, t], &skip, [b, chin, fr, t], target, &layer,
    );
    eprintln!(
        "ops_cpu hdec layer: in x range [{:.4}, {:.4}], skip [{:.4}, {:.4}], out range [{:.4}, {:.4}]",
        x.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        x.iter().cloned().fold(f32::INFINITY, f32::min),
        skip.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        skip.iter().cloned().fold(f32::INFINITY, f32::min),
        ops_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        ops_out.iter().cloned().fold(f32::INFINITY, f32::min),
    );
    eprintln!("ops_cpu out shape = {:?}", ops_shape);

    // ─── Reference: naive (no DConv yet — just the conv2d+GLU+conv_tr path) ─
    // 1. residual = x + skip
    let mut h: Vec<f32> = x.iter().zip(skip.iter()).map(|(a, b)| a + b).collect();
    let mut h_shape = [b, chin, fr, t];
    // 2. Conv2d(3,3) pad=(1,1) — chin -> 2*chin
    let (h2, h2_shape) = naive_conv2d_shaped(
        &h, h_shape,
        &layer.rewrite.data,
        [layer.rewrite.out_ch, layer.rewrite.in_ch, 3, 3],
        &layer.rewrite_bias.data,
        1, 1, 1, 1,
    );
    eprintln!("naive after conv2d: range [{:.4}, {:.4}]", h2.iter().cloned().fold(f32::NEG_INFINITY, f32::max), h2.iter().cloned().fold(f32::INFINITY, f32::min));
    // 3. GLU: 2*chin -> chin
    let (h3, h3_shape) = naive_glu_4d(&h2, h2_shape);
    eprintln!("naive after GLU: range [{:.4}, {:.4}]", h3.iter().cloned().fold(f32::NEG_INFINITY, f32::max), h3.iter().cloned().fold(f32::INFINITY, f32::min));
    // 4. DConv: per-frequency flatten + 2 DConvLayers.
    // Flatten [B, C, Fr, T] -> [B*Fr, C, T].
    let [bs, c3, fr3, t3] = h3_shape;
    let mut flat = vec![0.0f32; bs * fr3 * c3 * t3];
    for bi in 0..bs {
        for ci in 0..c3 {
            for fri in 0..fr3 {
                for ti in 0..t3 {
                    flat[((bi * fr3 + fri) * c3 + ci) * t3 + ti] =
                        h3[((bi * c3 + ci) * fr3 + fri) * t3 + ti];
                }
            }
        }
    }
    // Apply ops_cpu's dconv_forward on the flat tensor. We can't easily
    // naive-replicate DConv's internal GroupNorm + GELU + scale, so just
    // call ops_cpu and see the range.
    let (dconv_out, dconv_shape) = demucs_core_native::ops_cpu::dconv_forward(
        &flat, [bs * fr3, c3, t3], &layer.dconv,
    );
    eprintln!(
        "naive after DConv (via ops_cpu): range [{:.4}, {:.4}], shape {:?}",
        dconv_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        dconv_out.iter().cloned().fold(f32::INFINITY, f32::min),
        dconv_shape
    );
    // Unflatten [B*Fr, C, T] -> [B, C, Fr, T]
    let [n, c4, t4] = dconv_shape;
    let mut h3_after_dconv = vec![0.0f32; bs * c4 * fr3 * t4];
    for bi in 0..bs {
        for fri in 0..fr3 {
            for ci in 0..c4 {
                for ti in 0..t4 {
                    h3_after_dconv[((bi * c4 + ci) * fr3 + fri) * t4 + ti] =
                        dconv_out[((bi * fr3 + fri) * c4 + ci) * t4 + ti];
                }
            }
        }
    }
    let h3_shape_dconv = [bs, c4, fr3, t4];
    // 5. ConvTranspose2d([8,1], stride=4, pad=2)
    let (h4_raw, h4_shape) = naive_conv_transpose2d(
        &h3_after_dconv, h3_shape_dconv,
        &layer.conv_tr.data,
        [layer.conv_tr.out_ch, layer.conv_tr.in_ch, 8, 1],
        &layer.conv_tr_bias.data,
        2, 0, 4, 1,
    );
    let mut h4 = h4_raw;
    let (h4_raw, h4_shape) = naive_conv_transpose2d(
        &h3_after_dconv, h3_shape_dconv,
        &layer.conv_tr.data,
        [layer.conv_tr.out_ch, layer.conv_tr.in_ch, 8, 1],
        &layer.conv_tr_bias.data,
        2, 0, 4, 1,
    );
    let mut h4 = h4_raw;
    let [_, _, h4_fr, _] = h4_shape;
    if h4_fr > target {
        let mut trimmed = vec![0.0f32; b * layer.conv_tr.out_ch * target * t];
        for bi in 0..b {
            for co in 0..layer.conv_tr.out_ch {
                for fri in 0..target {
                    for ti in 0..t {
                        trimmed[((bi * layer.conv_tr.out_ch + co) * target + fri) * t + ti] =
                            h4[((bi * layer.conv_tr.out_ch + co) * h4_fr + fri) * t + ti];
                    }
                }
            }
        }
        h4 = trimmed;
    }
    let naive_shape = [b, layer.conv_tr.out_ch, target, t];
    eprintln!(
        "naive hdec layer (full, DConv via ops_cpu): out range [{:.4}, {:.4}], shape {:?}",
        h4.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        h4.iter().cloned().fold(f32::INFINITY, f32::min),
        naive_shape
    );
    let max_diff = ops_out.iter().zip(h4.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("max abs diff (ops vs naive, with DConv) = {:.6e}", max_diff);
}

#[test]
#[ignore]
fn probe_hdec_layer0_isolated_with_dummy() {
    // Run hdec_layer_forward in isolation with a unit-magnitude input matching
    // the transformer's actual output (Fr=8, max=180 ish, std=27).
    let model_path = std::path::PathBuf::from("../models/htdemucs_ft.safetensors");
    if !model_path.exists() {
        eprintln!("skipping");
        return;
    }
    let store = demucs_core_native::weights::WeightStore::load(&model_path).unwrap();
    let model = demucs_core_native::model::HTDemucs::from_store(&store, "04573f0d", 4, 512).unwrap();
    let layer = &model.decoders[0]; // chin=384, chout=192

    // Input: [1, 384, 8, 4] mimicking transformer output scale.
    let b = 1;
    let chin = 384;
    let fr = 8;
    let t = 4;
    let x: Vec<f32> = (0..b * chin * fr * t)
        .map(|i| (i as f32 * 0.013).sin() * 27.0)  // std ~ 19, range [-27, 27]
        .collect();
    // Skip: same shape.
    let skip: Vec<f32> = (0..b * chin * fr * t)
        .map(|i| (i as f32 * 0.011).cos() * 0.13)  // std ~ 0.09, range [-0.13, 0.13]
        .collect();

    let target = 32;
    let (out, out_shape) = demucs_core_native::ops_cpu::hdec_layer_forward(
        &x,
        [b, chin, fr, t],
        &skip,
        [b, chin, fr, t],
        target,
        layer,
    );
    let max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = out.iter().cloned().fold(f32::INFINITY, f32::min);
    let rms = (out.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / out.len() as f64).sqrt();
    eprintln!(
        "hdec layer 0 isolated: in_x range=[{:.4}, {:.4}], in_skip range=[{:.4}, {:.4}], out range=[{:.4}, {:.4}] rms={:.4}",
        x.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        x.iter().cloned().fold(f32::INFINITY, f32::min),
        skip.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        skip.iter().cloned().fold(f32::INFINITY, f32::min),
        max, min, rms,
    );
    eprintln!("out shape = {:?}", out_shape);

    // ─── Per-stage trace ────────────────────────────────────────────
    // Stage 1: x + skip
    let h1: Vec<f32> = x.iter().zip(skip.iter()).map(|(a, b)| a + b).collect();
    eprintln!(
        "  stage1 (x+skip): range=[{:.4}, {:.4}] rms={:.4}",
        h1.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        h1.iter().cloned().fold(f32::INFINITY, f32::min),
        (h1.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / h1.len() as f64).sqrt()
    );
    // Stage 2: conv2d (3,3) chin=384, 2*chin=768
    let rewrite_bias = demucs_core_native::model::Bias {
        data: layer.rewrite_bias.data.clone(),
        len: layer.rewrite_bias.len,
    };
    let (h2, h2_shape) = demucs_core_native::ops_cpu::conv2d(
        &h1, [b, chin, fr, t],
        &layer.rewrite, &rewrite_bias, 1, 1, 1, 1,
    );
    eprintln!(
        "  stage2 (conv2d 3x3): shape={:?} range=[{:.4}, {:.4}] rms={:.4}",
        h2_shape,
        h2.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        h2.iter().cloned().fold(f32::INFINITY, f32::min),
        (h2.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / h2.len() as f64).sqrt()
    );
    // Stage 3: GLU (a*sigmoid(b), 2*chin → chin)
    let [b2, c2x, fr2, t2] = h2_shape;
    let c_glu = c2x / 2;
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
    eprintln!(
        "  stage3 (GLU): shape=[{},{},{},{}] range=[{:.4}, {:.4}] rms={:.4}",
        b2, c_glu, fr2, t2,
        h3.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        h3.iter().cloned().fold(f32::INFINITY, f32::min),
        (h3.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / h3.len() as f64).sqrt()
    );
    // Stage 4: DConv (after per-frequency flatten)
    let mut flat = vec![0.0f32; b * fr * c_glu * t];
    for bi in 0..b {
        for ci in 0..c_glu {
            for fri in 0..fr {
                for ti in 0..t {
                    flat[((bi * fr + fri) * c_glu + ci) * t + ti] =
                        h3[((bi * c_glu + ci) * fr + fri) * t + ti];
                }
            }
        }
    }
    let (dconv_out, dconv_shape) = demucs_core_native::ops_cpu::dconv_forward(
        &flat, [b * fr, c_glu, t], &layer.dconv,
    );
    eprintln!(
        "  stage4 (DConv): shape={:?} range=[{:.4}, {:.4}] rms={:.4}",
        dconv_shape,
        dconv_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        dconv_out.iter().cloned().fold(f32::INFINITY, f32::min),
        (dconv_out.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / dconv_out.len() as f64).sqrt()
    );
    // Stage 5: ConvTranspose2d
    let [n, c4, t4] = dconv_shape;
    let mut unflat = vec![0.0f32; b * c4 * fr * t4];
    for bi in 0..b {
        for fri in 0..fr {
            for ci in 0..c4 {
                for ti in 0..t4 {
                    unflat[((bi * c4 + ci) * fr + fri) * t4 + ti] =
                        dconv_out[((bi * fr + fri) * c4 + ci) * t4 + ti];
                }
            }
        }
    }
    let (h5, h5_shape) = demucs_core_native::ops_cpu::conv_transpose2d(
        &unflat, [b, c4, fr, t4],
        &layer.conv_tr, &layer.conv_tr_bias, 2, 0, 4, 1,
    );
    eprintln!(
        "  stage5 (ConvTranspose2d k=8 s=4 p=2): shape={:?} range=[{:.4}, {:.4}] rms={:.4}",
        h5_shape,
        h5.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        h5.iter().cloned().fold(f32::INFINITY, f32::min),
        (h5.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / h5.len() as f64).sqrt()
    );
}

#[test]
#[ignore]
fn probe_extract_stems_zero_input() {
    use demucs_core_native::dsp::stft::Stft;
    use demucs_core_native::ops_cpu;

    let n_bins = 2048;
    let n_frames = 336;
    let padded_len = 343980;
    // freq_out all zeros, shape [1, 16, n_bins, n_frames]
    let freq_out = vec![0.0f32; 1 * 16 * n_bins * n_frames];
    let time_out = vec![0.0f32; 1 * 8 * padded_len];
    let mut stft = Stft::new(demucs_core_native::N_FFT, demucs_core_native::HOP_LENGTH);
    let stems = ops_cpu::extract_stems(
        &freq_out,
        [1, 16, n_bins, n_frames],
        &time_out,
        [1, 8, padded_len],
        n_frames,
        padded_len,
        padded_len,
        &mut stft,
    );
    for s in &stems {
        let max = s.left.iter().cloned().fold(0.0f32, f32::max);
        let min = s.left.iter().cloned().fold(0.0f32, f32::min);
        eprintln!(
            "stem {:?} left (zero input): range=[{:.4}, {:.4}] rms={:.4}",
            s.id, min, max,
            (s.left.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / s.left.len() as f64).sqrt()
        );
    }
    // Zero input should give zero output (linearity).
    for s in &stems {
        assert!(s.left.iter().all(|&v| v.abs() < 1e-3), "expected zero output for zero input, got max={}",
            s.left.iter().cloned().fold(0.0f32, f32::max));
        assert!(s.right.iter().all(|&v| v.abs() < 1e-3), "expected zero output for zero input, got max={}",
            s.right.iter().cloned().fold(0.0f32, f32::max));
    }
}

#[test]
#[ignore]
fn probe_extract_stems_known_freq() {
    use demucs_core_native::dsp::cac::cac_data_to_complex;
    use demucs_core_native::dsp::stft::Stft;
    use demucs_core_native::ops_cpu;
    // Construct freq_out = 1.0 in all 16 channels, time_out = 0.
    // After iSTFT (per channel), should produce a 343980-sample wav per stem.
    // This is a linearity test: iSTFT(δ_freq) + 0_time → known per-stem WAV.
    let n_bins = 2048;
    let n_frames = 336;
    let padded_len = 343980;
    let freq_out = vec![0.0f32; 1 * 16 * n_bins * n_frames];
    let time_out = vec![0.0f32; 1 * 8 * padded_len];
    let mut stft = Stft::new(demucs_core_native::N_FFT, demucs_core_native::HOP_LENGTH);
    let stems = ops_cpu::extract_stems(
        &freq_out,
        [1, 16, n_bins, n_frames],
        &time_out,
        [1, 8, padded_len],
        n_frames,
        padded_len,
        padded_len,
        &mut stft,
    );
    // Now construct a freq tensor where stem 0 (Drums) is all 1.0 in its
    // 4 CaC channels, others 0. Verify the Drums stem is non-zero and
    // the others are zero (iSTFT linearity).
    let mut freq_out = vec![0.0f32; 1 * 16 * n_bins * n_frames];
    let ch_stride = n_bins * n_frames;
    for bin in 0..n_bins {
        for frame in 0..n_frames {
            // Drums: channels 0..4.
            for ch in 0..4 {
                let idx = (0 * 4 + ch) * ch_stride + bin * n_frames + frame;
                freq_out[idx] = 0.1;
            }
        }
    }
    let stems = ops_cpu::extract_stems(
        &freq_out,
        [1, 16, n_bins, n_frames],
        &time_out,
        [1, 8, padded_len],
        n_frames,
        padded_len,
        padded_len,
        &mut stft,
    );
    for s in &stems {
        let max = s.left.iter().cloned().fold(0.0f32, f32::max);
        let min = s.left.iter().cloned().fold(0.0f32, f32::min);
        let rms = (s.left.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / s.left.len() as f64).sqrt();
        eprintln!("stem {:?} left: range=[{:.4}, {:.4}] rms={:.4}", s.id, min, max, rms);
    }
}

#[test]
#[ignore]
fn probe_self_attn_per_layer() {
    let model_path = std::path::PathBuf::from("../models/htdemucs_ft.safetensors");
    if !model_path.exists() {
        eprintln!("skipping");
        return;
    }
    let store = WeightStore::load(&model_path).expect("load model");
    let model = HTDemucs::from_store(&store, "04573f0d", 4, 512).expect("load HTDemucs");

    // Run all 3 self-attn layers (0, 2, 4) with the same input to see the
    // amplification pattern.
    for &layer_idx in &[0usize, 2, 4] {
        let layer = match &model.crosstransformer.layers[layer_idx].self_attn {
            Some(l) => l,
            None => panic!("layer {layer_idx} should be self_attn"),
        };
        eprintln!("=== layer {layer_idx} ===");

        let b = 1;
        let seq = 336;
        let d = 512;
        let x: Vec<f32> = (0..b * seq * d)
            .map(|i| (i as f32 * 0.0037).sin() * 3.0)
            .collect();
        stats("INPUT x", &x);

        // Just MHA self, no layernorm/residual.
        let (mha_out, _) = ops_cpu::mha_self(&x, [b, seq, d], &layer.attn);
        stats("MHA self (raw)", &mha_out);

        // Full self-attention layer.
        let (out, _) = ops_cpu::self_attn_layer_forward(&x, [b, seq, d], layer);
        stats("self_attn_layer_forward OUT", &out);
    }
}

#[test]
#[ignore]
fn probe_mha_self_uniform_random_input() {
    // Sanity: with uniform random weights and small input std, MHA output
    // should be approximately same magnitude as input × sqrt(d_head) (because
    // softmax @ V is a weighted average of V).
    let b = 1;
    let seq = 8;
    let d = 8;
    let h = 2;
    let d_head = 4;
    let x: Vec<f32> = (0..b * seq * d).map(|i| (i as f32 * 0.07).sin()).collect();
    let attn = demucs_core_native::model::MhaWeights {
        in_proj_weight: (0..3 * d * d).map(|i| ((i * 7919) % 100) as f32 / 1000.0 - 0.05).collect(),
        in_proj_bias: vec![0.0; 3 * d],
        out_proj_weight: (0..d * d).map(|i| ((i * 6151) % 100) as f32 / 1000.0 - 0.05).collect(),
        out_proj_bias: vec![0.0; d],
        d_model: d,
        n_heads: h,
    };
    stats("INPUT", &x);
    let (out, _) = ops_cpu::mha_self(&x, [b, seq, d], &attn);
    stats("MHA out", &out);
}

#[test]
#[ignore]
fn probe_mha_self_known_answer() {
    // Hand-derivable: B=1, seq=2, d=2, h=1, d_head=2.
    // x = [[1, 0], [0, 1]] (identity at positions 0, 1).
    // W_in_proj: split Q/K/V as 2x2 each, all ones. Q = x @ W_Q^T = x.
    //   (W_Q = identity times 2 = [[2, 0], [0, 2]]?)
    // Actually: Q = x @ W_Q^T + b_Q. W_Q layout [out, in] = [2, 2] row-major.
    // For y = x @ W^T, in W^T we want Q[i, o] = x[i, k] * W[o, k].
    // Set W_Q = identity: W_Q = [[1,0],[0,1]], so Q = x.
    // Similarly K = x, V = x. Then QK^T = x x^T = [[1,0],[0,1]].
    // softmax(QK^T / sqrt(2)) per row:
    //   row 0: [softmax(1/sqrt2, 0)] ≈ [0.622, 0.378]
    //   row 1: [softmax(0, 1/sqrt2)] ≈ [0.378, 0.622]
    // attn @ V:
    //   row 0: 0.622 * [1,0] + 0.378 * [0,1] = [0.622, 0.378]
    //   row 1: 0.378 * [1,0] + 0.622 * [0,1] = [0.378, 0.622]
    // Out proj = I (identity): out = attn @ V (no change).
    let b = 1;
    let seq = 2;
    let d = 2;
    let h = 1;
    let d_head = 2;
    // in_proj_weight: 3 chunks of 2*2 = 4 each, all identity in PyTorch layout.
    // PyTorch Linear: weight [out, in]. For identity x@W^T to give x: W = I.
    let mut in_proj_weight = vec![0.0f32; 3 * d * d];
    // Q chunk: W_Q = identity → W[0,0]=1, W[1,1]=1.
    in_proj_weight[0 * d * d + 0] = 1.0; // W_Q[0,0]
    in_proj_weight[0 * d * d + 1 * d + 1] = 1.0; // W_Q[1,1]
    // K chunk: identity.
    in_proj_weight[1 * d * d + 0] = 1.0;
    in_proj_weight[1 * d * d + 1 * d + 1] = 1.0;
    // V chunk: identity.
    in_proj_weight[2 * d * d + 0] = 1.0;
    in_proj_weight[2 * d * d + 1 * d + 1] = 1.0;
    let attn = demucs_core_native::model::MhaWeights {
        in_proj_weight,
        in_proj_bias: vec![0.0; 3 * d],
        out_proj_weight: vec![1.0, 0.0, 0.0, 1.0], // identity in PyTorch [out,in] = I
        out_proj_bias: vec![0.0; d],
        d_model: d,
        n_heads: h,
    };
    // x = [[1, 0], [0, 1]]
    let x = vec![1.0, 0.0, 0.0, 1.0];
    let (out, _) = ops_cpu::mha_self(&x, [b, seq, d], &attn);
    eprintln!("out = {:?}", out);
    // Expected ≈ [0.670, 0.330, 0.330, 0.670] (softmax of [1/sqrt2, 0])
    let expected = [0.670, 0.330, 0.330, 0.670];
    for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        eprintln!("[{i}] got={:.4} expected={:.4} diff={:.4}", got, exp, diff);
        assert!(diff < 1e-3, "mha self known-answer failed at {i}: {got} vs {exp}");
    }
}
