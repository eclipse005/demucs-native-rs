//! Isolated conv2d_8x1_s4p2 GPU vs PyTorch ground truth.
//! Same deterministic sin*0.3 input as tests/py_henc_probe.py.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_conv2d8x1_matches_pytorch() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("model");
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");

    // Deterministic input matching py_henc_probe.py: (i * 1e-4).sin() * 0.3
    let (b, cin, fr, t) = (1, 4, 2048, 336);
    let input: Vec<f32> = (0..b * cin * fr * t)
        .map(|i| ((i as f32) * 1e-4).sin() * 0.3)
        .collect();

    let gpu_in = state.upload_f32(&input, vec![b, cin, fr, t]).expect("up");
    let gpu_out = demucs_core_native::cuda_ops::conv2d_8x1_s4p2(
        &state, &gpu_in, &gpu_model.encoders[0].conv, &gpu_model.encoders[0].conv_bias,
    ).expect("conv2d_8x1");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    let gpu_max = gpu_dl.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let gpu_mean = gpu_dl.iter().sum::<f32>() as f64 / gpu_dl.len() as f64;
    let gpu_std = (gpu_dl.iter().map(|x| (*x as f64 - gpu_mean).powi(2)).sum::<f64>()
        / gpu_dl.len() as f64).sqrt();
    eprintln!("GPU conv2d_8x1: shape={:?} max={:.4} mean={:.4} std={:.4}",
              gpu_out.shape(), gpu_max, gpu_mean, gpu_std);
    eprintln!("GPU out[0..5] = {:?}", &gpu_dl[..5]);
    eprintln!("PyTorch expected: max=0.7965 mean=-0.0010 std=0.1658");
    eprintln!("PyTorch out[0..5] = [0.053950, 0.053934, 0.053918, 0.053902, 0.053887]");
    // f16 expected ~1% error. If GPU >> PyTorch, conv2d_8x1 has a bug.
    assert!(gpu_max < 1.5, "GPU conv2d_8x1 max {} >> pytorch 0.7965", gpu_max);
}
