//! Isolated conv2d_1x1 (rewrite) GPU vs PyTorch.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_conv1x1_matches_pytorch() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("model");
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");

    let (cin, fr, t) = (48, 512, 336);
    let input: Vec<f32> = (0..1 * cin * fr * t).map(|i| ((i as f32) * 1e-4).sin() * 1.0).collect();
    let gpu_in = state.upload_f32(&input, vec![1, cin, fr, t]).expect("up");
    let gpu_out = demucs_core_native::cuda_ops::conv2d_1x1(
        &state, &gpu_in, &gpu_model.encoders[0].rewrite, &gpu_model.encoders[0].rewrite_bias,
    ).expect("conv1x1");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    let gpu_max = gpu_dl.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("GPU conv1x1: shape={:?} max={:.4}", gpu_out.shape(), gpu_max);
    eprintln!("GPU out[0..5] = {:?}", &gpu_dl[..5]);
    eprintln!("PyTorch: run tests/py_conv1x1_probe.py");
}
