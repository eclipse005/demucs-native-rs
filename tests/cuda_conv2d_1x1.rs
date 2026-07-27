//! Test conv2d_1x1 against CPU using actual henc rewrite weights.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::{CudaState, GpuTensor};
use demucs_core_native::gpu_model::{GpuBias, GpuConv2dWeight};
use demucs_core_native::model::{Bias, Conv2dWeight, HTDemucs};
use demucs_core_native::weights::WeightStore;
use demucs_core_native::ops_cpu;

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
#[ignore]
fn cuda_conv2d_1x1_henc0_rewrite_matches_cpu() {
    let model_path = std::path::Path::new("../models/htdemucs.safetensors");
    let store = WeightStore::load(model_path).expect("load model");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("from_store");

    let layer = &cpu_model.encoders[0];
    // conv2d_1x1: in [B=1, C=48, Fr=16, T=32] → out [B, 96, 16, 32]
    let b = 1;
    let c_in = 48;
    let fr = 16;
    let t = 32;
    let input: Vec<f32> = (0..b * c_in * fr * t)
        .map(|i| ((i as f32) * 0.013 - 0.3) * 0.5)
        .collect();

    // CPU conv2d (k=1, s=1, p=0)
    let (cpu_out, cpu_shape) = ops_cpu::conv2d(
        &input, [b, c_in, fr, t],
        &layer.rewrite, &layer.rewrite_bias,
        0, 0, 1, 1,
    );
    eprintln!("cpu conv2d_1x1 shape = {:?}", cpu_shape);

    // GPU conv2d_1x1
    let state = Arc::new(CudaState::new(0).expect("cuda init"));
    let gpu_w = GpuConv2dWeight::from_cpu(&state, &layer.rewrite).expect("up w");
    let gpu_b = GpuBias::from_cpu(&state, &Bias { data: layer.rewrite_bias.data.clone(), len: layer.rewrite_bias.len }).expect("up b");
    let gpu_in = state.upload_f32(&input, vec![b, c_in, fr, t]).expect("up");
    let gpu_out: GpuTensor = demucs_core_native::cuda_ops::conv2d_1x1(&state, &gpu_in, &gpu_w, &gpu_b)
        .expect("conv2d_1x1");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");

    let diff = max_diff(&cpu_out, &gpu_dl);
    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let rms = (cpu_out.iter().zip(&gpu_dl).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
        / cpu_out.len() as f32)
        .sqrt();
    eprintln!("conv2d_1x1 cpu vs gpu: max_diff={:.4}, rms={:.4}, max_val={:.2}", diff, rms, max_val);
    eprintln!("cpu[0..5]={:?}", &cpu_out[..5]);
    eprintln!("gpu[0..5]={:?}", &gpu_dl[..5]);
    let tol = (max_val * 0.05).max(5e-1);
    assert!(diff < tol, "conv2d_1x1 cpu vs gpu: max_diff={:.4} exceeds tol={:.4}", diff, tol);
}
