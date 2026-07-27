//! Small conv2d_1x1 test to isolate the bug.

#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::{CudaState, GpuTensor};
use demucs_core_native::gpu_model::{GpuBias, GpuConv2dWeight};
use demucs_core_native::model::{Bias, Conv2dWeight};

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
#[ignore]
fn cuda_conv2d_1x1_small_matches_cpu() {
    let state = Arc::new(CudaState::new(0).expect("cuda init"));
    let b = 1;
    let c_in = 3;
    let h = 2;
    let w = 4;
    let c_out = 2;
    let input: Vec<f32> = (0..b*c_in*h*w).map(|i| (i as f32)*0.1).collect();
    let w_data: Vec<f32> = (0..c_out*c_in*1*1).map(|i| (i as f32)*0.5 - 0.3).collect();
    let b_data: Vec<f32> = (0..c_out).map(|i| 0.1*(i as f32)).collect();

    // CPU conv2d k=1 s=1 p=0.
    let mut cpu_out = vec![0.0f32; b*c_out*h*w];
    for bi in 0..b {
        for co in 0..c_out {
            for ih in 0..h {
                for iw in 0..w {
                    let mut sum = b_data[co];
                    for ci in 0..c_in {
                        let x = input[((bi*c_in + ci)*h + ih)*w + iw];
                        let wt = w_data[((co*c_in + ci)*1 + 0)*1 + 0];
                        sum += x * wt;
                    }
                    cpu_out[((bi*c_out + co)*h + ih)*w + iw] = sum;
                }
            }
        }
    }

    // GPU
    let gpu_w = GpuConv2dWeight::from_cpu(&state, &Conv2dWeight { data: w_data.clone(), out_ch: c_out, in_ch: c_in, kh: 1, kw: 1 }).expect("up w");
    let gpu_b = GpuBias::from_cpu(&state, &Bias { data: b_data.clone(), len: b_data.len() }).expect("up b");
    let gpu_in = state.upload_f32(&input, vec![b, c_in, h, w]).expect("up");
    let gpu_out: GpuTensor = demucs_core_native::cuda_ops::conv2d_1x1(&state, &gpu_in, &gpu_w, &gpu_b).expect("conv");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");

    eprintln!("cpu[0..5]={:?}", &cpu_out[..5]);
    eprintln!("gpu[0..5]={:?}", &gpu_dl[..5]);
    let diff = max_diff(&cpu_out, &gpu_dl);
    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("max_diff={:.4}, max_val={:.4}", diff, max_val);
    let tol = (max_val * 0.05).max(5e-2);
    assert!(diff < tol, "max_diff={:.4} exceeds tol={:.4}", diff, tol);
}
