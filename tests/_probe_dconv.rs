//! Targeted: dconv_layer_forward stage-by-stage on real weights, with
//! synthetic input, to see if any internal stage explodes.

use demucs_core_native::model::{DConv, DConvLayer, HTDemucs};
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
    eprintln!("{name}: range=[{min_v:.4}, {max_v:.4}] mean={mean:.4} rms={rms:.4}");
}

#[test]
#[ignore]
fn probe_dconv_layer0_isolated() {
    let model_path = std::path::PathBuf::from("../models/htdemucs_ft.safetensors");
    if !model_path.exists() {
        eprintln!("skipping");
        return;
    }
    let store = WeightStore::load(&model_path).unwrap();
    let model = HTDemucs::from_store(&store, "04573f0d", 4, 512).unwrap();
    let dconv = &model.decoders[0].dconv;

    // Match hdec decoder 0 internal: input is GLU output, shape [B*Fr, C=384, T=4].
    // Use large amplitude input to see if normalize works.
    let b = 1;
    let c = 384;
    let l = 4;
    let x: Vec<f32> = (0..b * c * l)
        .map(|i| (i as f32 * 0.013).sin() * 100.0 + 50.0)  // big mean + large amp
        .collect();
    stats("INPUT x", &x);

    // Single DConvLayer (j=0, dilation=1).
    let (h, h_shape) = ops_cpu::dconv_layer_forward(
        &x, [b, c, l], &dconv.layers[0], 1,
    );
    stats(&format!("after dconv_layer[0] (dilation=1), shape={:?}", h_shape), &h);

    // Both layers via dconv_forward.
    let (h2, h2_shape) = ops_cpu::dconv_forward(&x, [b, c, l], dconv);
    stats(&format!("after dconv_forward (both layers), shape={:?}", h2_shape), &h2);
}

#[test]
#[ignore]
fn probe_dconv_layer0_step_by_step() {
    let model_path = std::path::PathBuf::from("../models/htdemucs_ft.safetensors");
    if !model_path.exists() {
        eprintln!("skipping");
        return;
    }
    let store = WeightStore::load(&model_path).unwrap();
    let model = HTDemucs::from_store(&store, "04573f0d", 4, 512).unwrap();
    let layer = &model.decoders[0].dconv.layers[0]; // dilation=1, chin=384, compress=48

    // Replicate dconv_layer_forward with stage-by-stage stats.
    let b = 1;
    let c = 384;
    let l = 4;
    let x: Vec<f32> = (0..b * c * l)
        .map(|i| (i as f32 * 0.013).sin() * 100.0 + 50.0)
        .collect();
    stats("INPUT x", &x);

    // conv1: k=3, pad=1, dilation=1
    let (h, h_shape) = ops_cpu::conv1d(
        &x, [b, c, l], &layer.conv1, &layer.conv1_bias, 1, 1,
    );
    stats(&format!("stage1 (conv1 k=3 d=1 p=1) [B,C,L={l}→{}], shape={:?}", h_shape[2], h_shape), &h);
    // groupnorm1: per-batch along (C=48, L)
    let mut h = h;
    ops_cpu::groupnorm1(&mut h, h_shape, &layer.norm1);
    stats(&format!("stage2 (groupnorm1 on compress=48), shape={:?}", h_shape), &h);
    // gelu
    ops_cpu::gelu(&mut h);
    stats(&format!("stage3 (gelu), shape={:?}", h_shape), &h);
    // conv2: k=1, pad=0, dilation=1
    let (h2, h2_shape) = ops_cpu::conv1d(
        &h, h_shape, &layer.conv2, &layer.conv2_bias, 0, 1,
    );
    stats(&format!("stage4 (conv2 k=1) [B,C,L={l}→{}], shape={:?}", h2_shape[2], h2_shape), &h2);
    // groupnorm2
    let mut h2 = h2;
    ops_cpu::groupnorm1(&mut h2, h2_shape, &layer.norm2);
    stats(&format!("stage5 (groupnorm2 on 2*ch=768), shape={:?}", h2_shape), &h2);
    // glu
    let (mut h3, h3_shape) = ops_cpu::glu_channel(&h2, h2_shape);
    stats(&format!("stage6 (glu) [B,C,L={l}→{}], shape={:?}", h3_shape[2], h3_shape), &h3);
    // layer_scale
    ops_cpu::layer_scale(&mut h3, h3_shape, &layer.scale);
    stats(&format!("stage7 (layer_scale), shape={:?}", h3_shape), &h3);
    // +residual
    for bi in 0..b {
        for ci in 0..c {
            for li in 0..h3_shape[2] {
                let idx = (bi * c + ci) * h3_shape[2] + li;
                h3[idx] += x[(bi * c + ci) * l + li];
            }
        }
    }
    stats("stage8 (+residual)", &h3);
}
