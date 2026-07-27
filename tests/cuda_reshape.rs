//! Test reshape ops.

#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::{CudaState, GpuTensor};

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
#[ignore]
fn cuda_reshape_bcft_to_bfct_matches_cpu() {
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let b = 2; let c = 3; let fr = 4; let t = 5;
    let input: Vec<f32> = (0..b*c*fr*t).map(|i| i as f32).collect();

    // CPU reference.
    let mut cpu_out = vec![0.0f32; b*c*fr*t];
    for bi in 0..b {
        for ci in 0..c {
            for fri in 0..fr {
                for ti in 0..t {
                    let src = ((bi*c + ci)*fr + fri)*t + ti;
                    let dst = ((bi*fr + fri)*c + ci)*t + ti;
                    cpu_out[dst] = input[src];
                }
            }
        }
    }

    // GPU.
    let gpu_in = state.upload_f32(&input, vec![b, c, fr, t]).expect("up");
    let gpu_out: GpuTensor = demucs_core_native::cuda_ops::reshape_bcft_to_bfct(&state, &gpu_in).expect("rs");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    let diff = max_diff(&cpu_out, &gpu_dl);
    eprintln!("reshape_bcft_to_bfct max_diff={:.4}", diff);
    eprintln!("cpu[0..5]={:?}", &cpu_out[..5]);
    eprintln!("gpu[0..5]={:?}", &gpu_dl[..5]);
    assert!(diff < 1e-4, "max_diff={:.6}", diff);
}

#[test]
#[ignore]
fn cuda_transpose_bchw_bhwc_matches_cpu() {
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let b = 2; let c = 3; let h = 4; let w = 5;
    let input: Vec<f32> = (0..b*c*h*w).map(|i| i as f32).collect();

    // CPU: [B, C, H, W] → [B, H, W, C]
    let mut cpu_out = vec![0.0f32; b*c*h*w];
    for bi in 0..b {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src = ((bi*c + ci)*h + hi)*w + wi;
                    let dst = ((bi*h + hi)*w + wi)*c + ci;
                    cpu_out[dst] = input[src];
                }
            }
        }
    }

    // GPU.
    let gpu_in = state.upload_f32(&input, vec![b, c, h, w]).expect("up");
    let gpu_out: GpuTensor = demucs_core_native::cuda_ops::transpose_bchw_to_bhwc(&state, &gpu_in).expect("rs");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    let diff = max_diff(&cpu_out, &gpu_dl);
    eprintln!("transpose_bchw_to_bhwc max_diff={:.4}", diff);
    eprintln!("cpu[0..5]={:?}", &cpu_out[..5]);
    eprintln!("gpu[0..5]={:?}", &gpu_dl[..5]);
    assert!(diff < 1e-4, "max_diff={:.6}", diff);
}
