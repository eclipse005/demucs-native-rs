//! Test conv2d_8x1_s4p2 with actual henc weights.

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
fn cuda_conv2d_8x1_henc0_matches_cpu() {
    let model_path = std::path::Path::new("../models/htdemucs.safetensors");
    let store = WeightStore::load(model_path).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("from_store");
    let layer = &cpu_model.encoders[0];

    let b = 1;
    let c_in = 4;
    let fr = 64;
    let t = 32;
    let input: Vec<f32> = (0..b*c_in*fr*t).map(|i| ((i as f32)*0.013 - 0.3)*0.5).collect();

    let (cpu_out, cpu_shape) = ops_cpu::conv2d(
        &input, [b, c_in, fr, t],
        &layer.conv, &layer.conv_bias,
        2, 0, 4, 1,
    );
    eprintln!("cpu conv shape = {:?}", cpu_shape);

    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_w = GpuConv2dWeight::from_cpu(&state, &layer.conv).expect("up w");
    let gpu_b = GpuBias::from_cpu(&state, &Bias { data: layer.conv_bias.data.clone(), len: layer.conv_bias.len }).expect("up b");
    let gpu_in = state.upload_f32(&input, vec![b, c_in, fr, t]).expect("up");
    let gpu_out: GpuTensor = demucs_core_native::cuda_ops::conv2d_8x1_s4p2(&state, &gpu_in, &gpu_w, &gpu_b).expect("conv");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");

    let diff = max_diff(&cpu_out, &gpu_dl);
    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let rms = (cpu_out.iter().zip(&gpu_dl).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / cpu_out.len() as f32).sqrt();
    eprintln!("max_diff={:.4}, rms={:.4}, max_val={:.2}", diff, rms, max_val);
    eprintln!("cpu[0..5]={:?}", &cpu_out[..5]);
    eprintln!("gpu[0..5]={:?}", &gpu_dl[..5]);
    let tol = (max_val * 0.05).max(5e-1);
    assert!(diff < tol, "max_diff={:.4} exceeds tol={:.4}", diff, tol);
}
