//! Isolated conv_transpose2d GPU vs CPU comparison.
//! Same input + same weights, compare outputs. Locates whether the GPU
//! conv_transpose2d has an independent bug (vs being polluted by upstream
//! henc differences).

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_conv_transpose2d_isolated_matches_cpu() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("model");
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");

    // hdec[0].conv_tr: ConvTranspose2d [8,1] stride [4,1] pad [2,0].
    // chin=384, chout=192 (hdec[0] is deepest: 384→192). Input Fr=8 (real chunk1 bottleneck).
    let layer = &cpu_model.decoders[0];
    let b = 1; let c_in = 384; let fr = 8; let t = 336;
    // Synthetic input (deterministic, moderate amplitude).
    let input: Vec<f32> = (0..b * c_in * fr * t).map(|i| ((i as f32) * 1e-3).sin() * 1.0).collect();
    let x_shape = [b, c_in, fr, t];

    // CPU conv_transpose2d
    let (cpu_out, cpu_shape) = ops_cpu::conv_transpose2d(
        &input, x_shape, &layer.conv_tr, &layer.conv_tr_bias, 2, 0, 4, 1,
    );
    let cpu_max = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("CPU convTr: shape={:?} max={:.4} len={}", cpu_shape, cpu_max, cpu_out.len());

    // GPU conv_transpose2d_8x1_s4p2
    let gpu_in = state.upload_f32(&input, vec![b, c_in, fr, t]).expect("up");
    let gpu_out = demucs_core_native::cuda_ops::conv_transpose2d_8x1_s4p2(
        &state, &gpu_in, &gpu_model.decoders[0].conv_tr, &gpu_model.decoders[0].conv_tr_bias,
    ).expect("convTr");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    let gpu_max = gpu_dl.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("GPU convTr: shape={:?} max={:.4} len={}", gpu_out.shape(), gpu_max, gpu_dl.len());

    // Compare element-wise (both should have same shape after trim).
    let n = cpu_out.len().min(gpu_dl.len());
    let mut max_diff = 0.0f32;
    let mut sum_sq = 0.0f64;
    for i in 0..n {
        let d = (cpu_out[i] - gpu_dl[i]).abs();
        max_diff = max_diff.max(d);
        sum_sq += (d * d) as f64;
    }
    let rms = (sum_sq / n as f64).sqrt();
    eprintln!("convTr cpu vs gpu: max_diff={:.4} rms={:.4} (gpu/cpu max ratio={:.2})",
              max_diff, rms, gpu_max / cpu_max.max(1e-9));
}
