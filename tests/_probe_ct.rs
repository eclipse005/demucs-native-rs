//! Focused comparison: conv_transpose2d/1d ops_cpu vs a from-scratch naive
//! reference, on a real-weight decoder.0 conv_tr tensor. Helps locate any
//! im2col / GEMM stride bug that the unit tests (which all use c_out=1) miss.

use demucs_core_native::model::{Bias, HDecLayer};
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
    let rms = (x.iter().map(|v| ((*v - mean) as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    eprintln!("{name}: range=[{min_v:.4}, {max_v:.4}] rms={rms:.4}");
}

#[test]
#[ignore]
fn probe_conv_transpose2d_ops_vs_naive() {
    let model_path = std::path::PathBuf::from("../models/htdemucs_ft.safetensors");
    if !model_path.exists() {
        eprintln!("skipping");
        return;
    }
    let store = WeightStore::load(&model_path).unwrap();
    let layer = HDecLayer::from_store(&store, "04573f0d", "decoder.0", 384, 192, false).unwrap();

    // ConvTranspose2d layer: chin=384 (matches DConv output), chout=192,
    // kH=8, kW=1, stride=4, pad=2. Input: [1, 384, 8, 4].
    let b = 1;
    let c_in = 384;
    let c_out = 192;
    let kh = 8;
    let kw = 1;
    let h_in = 8;
    let w_in = 4;
    let pad_h = 2;
    let pad_w = 0;
    let stride_h = 4;
    let stride_w = 1;

    let h_out = (h_in - 1) * stride_h + (kh - 1) - 2 * pad_h + 1;
    let w_out = (w_in - 1) * stride_w + (kw - 1) - 2 * pad_w + 1;
    eprintln!("ConvTranspose2d config: in=[{b},{c_in},{h_in},{w_in}], out=[{b},{c_out},{h_out},{w_out}]");

    // Random input.
    let x: Vec<f32> = (0..b * c_in * h_in * w_in)
        .map(|i| (i as f32 * 0.013).sin() * 0.5 + 0.1)
        .collect();
    stats("input x", &x);

    // ─── Reference: ops_cpu::conv_transpose2d ────────────────────────
    let (ops_out, ops_shape) = ops_cpu::conv_transpose2d(
        &x,
        [b, c_in, h_in, w_in],
        &layer.conv_tr,
        &layer.conv_tr_bias,
        pad_h, pad_w, stride_h, stride_w,
    );
    eprintln!("ops: shape={:?}", ops_shape);
    stats("ops_conv_transpose2d", &ops_out);

    // ─── Reference: from-scratch naive (operates directly on
    // take_conv_transpose2d's reordered `[patch, c_out]` row-major memory:
    // `reordered[i, oc] = a[ic, oc, kh, kw]` at memory `i*c_out + oc` where
    // i = ic*kh*kw + kh*kw + kw).
    let mut naive_out = vec![0.0f32; b * c_out * h_out * w_out];
    // Add bias.
    for bi in 0..b {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    naive_out[((bi * c_out + oc) * h_out + oh) * w_out + ow] =
                        layer.conv_tr_bias.data[oc];
                }
            }
        }
    }
    // Scatter-add.
    for bi in 0..b {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    for ic in 0..c_in {
                        for dkh in 0..kh {
                            for dkw in 0..kw {
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
                                if ih >= h_in || iw >= w_in {
                                    continue;
                                }
                                let x_val = x[((bi * c_in + ic) * h_in + ih) * w_in + iw];
                                let i = ic * kh * kw + dkh * kw + dkw;
                                naive_out[((bi * c_out + oc) * h_out + oh) * w_out + ow] +=
                                    layer.conv_tr.data[i * c_out + oc] * x_val;
                            }
                        }
                    }
                }
            }
        }
    }
    stats("naive_conv_transpose2d", &naive_out);
    let max_diff = ops_out
        .iter()
        .zip(naive_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_diff: f32 = ops_out
        .iter()
        .zip(naive_out.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / ops_out.len() as f32;
    eprintln!("max_abs_diff = {max_diff:.6e}, mean_diff = {mean_diff:.6e}");

    // Find first divergent entry to localize the bug.
    for (i, (a, n)) in ops_out.iter().zip(naive_out.iter()).enumerate() {
        if (a - n).abs() > 1e-3 {
            let b = i / (c_out * h_out * w_out);
            let rest = i % (c_out * h_out * w_out);
            let oc = rest / (h_out * w_out);
            let r2 = rest % (h_out * w_out);
            let oh = r2 / w_out;
            let ow = r2 % w_out;
            eprintln!(
                "  first diff at i={i}: ops[{},{},{},{}]={} naive={} diff={}",
                b, oc, oh, ow, a, n, a - n
            );
            if i > 100 {
                break;
            }
        }
    }
}
