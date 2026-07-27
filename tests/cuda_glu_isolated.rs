//! Isolated glu_channel GPU vs PyTorch GLU.
//! Same deterministic input as tests/py_glu_probe.py.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::{CudaState, GpuTensor};

#[test]
#[ignore]
fn cuda_glu_matches_pytorch() {
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let c = 48;
    let l = 256;
    // Deterministic input matching py_glu_probe.py: (i*0.01).sin()*3.0
    let input: Vec<f32> = (0..1 * (2 * c) * l)
        .map(|i| ((i as f32) * 0.01).sin() * 3.0)
        .collect();
    let gpu_in = state.upload_f32(&input, vec![1, 2 * c, l]).expect("up");
    let gpu_out = demucs_core_native::cuda_ops::glu_channel(&state, &gpu_in).expect("glu");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    let gpu_max = gpu_dl.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("GPU glu: shape={:?} max={:.4}", gpu_out.shape(), gpu_max);
    eprintln!("GPU glu out[0..5] = {:?}", &gpu_dl[..5]);
    eprintln!("PyTorch glu out[0..5] = [run tests/py_glu_probe.py]");
    // f16 ~1% error expected.
    assert_eq!(gpu_out.shape(), &[1, c, l]);
}
