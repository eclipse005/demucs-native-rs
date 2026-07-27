//! MHA GPU vs CPU correctness. Random weights + input, compare within
//! f16 tolerance (softmax amplifies, so use rms not just max_diff).

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::{CudaState, GpuTensor};
use demucs_core_native::gpu_model::GpuMhaWeights;
use demucs_core_native::model::MhaWeights;
use demucs_core_native::ops_cpu;

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}
fn rms(a: &[f32], b: &[f32]) -> f32 {
    (a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / a.len() as f32).sqrt()
}

#[test]
#[ignore]
fn cuda_mha_self_matches_cpu() {
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let b = 1;
    let s = 8;
    let d = 16;
    let h = 4;
    let input: Vec<f32> = (0..b * s * d).map(|i| ((i as f32) * 0.07 - 0.3) * 0.5).collect();

    let in_proj_weight: Vec<f32> = (0..3 * d * d).map(|i| ((i as f32) * 0.013 - 0.3) * 0.1).collect();
    let in_proj_bias: Vec<f32> = (0..3 * d).map(|i| 0.1 * (i as f32) - 0.2).collect();
    let out_proj_weight: Vec<f32> = (0..d * d).map(|i| ((i as f32) * 0.011 - 0.3) * 0.1).collect();
    let out_proj_bias: Vec<f32> = (0..d).map(|i| 0.05 * (i as f32) - 0.05).collect();
    let mha = MhaWeights {
        in_proj_weight,
        in_proj_bias,
        out_proj_weight,
        out_proj_bias,
        d_model: d,
        n_heads: h,
    };
    let gpu_mha = GpuMhaWeights::from_cpu(&state, &mha).expect("up mha");

    // CPU reference.
    let (cpu_out, _) = ops_cpu::mha_self(&input, [b, s, d], &mha);

    // GPU.
    let gpu_in = state.upload_f32(&input, vec![b, s, d]).expect("up");
    let gpu_out: GpuTensor =
        demucs_core_native::cuda_ops::mha(&state, &gpu_in, &gpu_in, &gpu_mha).expect("mha");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");

    let diff = max_diff(&cpu_out, &gpu_dl);
    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let r = rms(&cpu_out, &gpu_dl);
    eprintln!(
        "mha self: max_diff={:.4} rms={:.4} max_val={:.4}",
        diff, r, max_val
    );
    eprintln!("cpu[0..5]={:?}", &cpu_out[..5]);
    eprintln!("gpu[0..5]={:?}", &gpu_dl[..5]);
    // Softmax amplifies f16 noise; rms is the honest metric.
    let rms_tol = (max_val * 0.05).max(1e-2);
    assert!(r < rms_tol, "rms={:.4} > tol={:.4}", r, rms_tol);
}
