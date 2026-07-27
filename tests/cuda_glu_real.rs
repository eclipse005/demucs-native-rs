//! Test glu_channel with realistic input data similar to henc rewritten.

#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::{CudaState, GpuTensor};
use demucs_core_native::ops_cpu;

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
#[ignore]
fn cuda_glu_real_matches_cpu() {
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let b = 1;
    let c2 = 96;
    let l = 16 * 32; // simulating [B, 2C, H*W] flatten of [B, 96, 16, 32]
    let input: Vec<f32> = (0..b*c2*l).map(|i| (i as f32)*0.013 - 0.3).collect();

    // CPU reference.
    let (cpu, cpu_shape) = ops_cpu::glu_channel(&input, [b, c2, l]);
    eprintln!("cpu glu shape: {:?}", cpu_shape);

    let gpu = state.upload_f32(&input, vec![b, c2, l]).expect("up");
    let gpu_out: GpuTensor = demucs_core_native::cuda_ops::glu_channel(&state, &gpu).expect("glu");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");

    let diff = max_diff(&cpu, &gpu_dl);
    let max_val = cpu.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let rms = (cpu.iter().zip(&gpu_dl).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / cpu.len() as f32).sqrt();
    eprintln!("glu cpu vs gpu: max_diff={:.4}, rms={:.4}, max_val={:.4}", diff, rms, max_val);
    eprintln!("cpu[0..5]={:?}", &cpu[..5]);
    eprintln!("gpu[0..5]={:?}", &gpu_dl[..5]);
    let tol = (max_val * 0.05).max(5e-2);
    assert!(diff < tol, "max_diff={:.4} exceeds tol={:.4}", diff, tol);
}
