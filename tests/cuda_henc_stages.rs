//! Stage-by-stage henc test.

#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::{CudaState, GpuTensor};
use demucs_core_native::gpu_model::{GpuBias, GpuConv2dWeight, GpuHTDemucs};
use demucs_core_native::model::{Bias, HTDemucs};
use demucs_core_native::weights::WeightStore;
use demucs_core_native::ops_cpu;

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
#[ignore]
fn cuda_henc0_after_rewrite_pre_glu_matches_cpu() {
    let model_path = std::path::Path::new("../models/htdemucs.safetensors");
    let store = WeightStore::load(model_path).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("from_store");

    let b = 1;
    let c_in = 4;
    let fr = 64;
    let t = 32;
    let input: Vec<f32> = (0..b*c_in*fr*t).map(|i| ((i as f32)*0.013 - 0.3)*0.5).collect();
    let layer = &cpu_model.encoders[0];

    // Do CPU forward through conv2d_8x1_s4p2 + gelu + reshape + dconv + reshape + rewrite.
    // For simplicity, just do conv2d_8x1_s4p2 + gelu + CPU dconv + CPU rewrite.
    let (mut h, h_shape) = ops_cpu::conv2d(&input, [b, c_in, fr, t], &layer.conv, &layer.conv_bias, 2, 0, 4, 1);
    eprintln!("after conv2d: shape {:?}", h_shape);
    ops_cpu::gelu(&mut h);
    let [_, c_out, fr_out, _] = h_shape;
    // Reshape [B, C_out, Fr_out, T] → [B*Fr_out, C_out, T]
    let mut flat = vec![0.0f32; b * fr_out * c_out * t];
    for bi in 0..b {
        for ci in 0..c_out {
            for fri in 0..fr_out {
                for ti in 0..t {
                    flat[((bi * fr_out + fri) * c_out + ci) * t + ti] =
                        h[((bi * c_out + ci) * fr_out + fri) * t + ti];
                }
            }
        }
    }
    let (mut dconv_out, dconv_shape) = ops_cpu::dconv_forward(&flat, [b * fr_out, c_out, t], &layer.dconv);
    eprintln!("after dconv: shape {:?}", dconv_shape);
    let [n2, c2, t2] = dconv_shape;
    assert_eq!(n2, b * fr_out);
    // Reshape back [B*Fr_out, C, T] → [B, C, Fr, T]
    let mut unflat = vec![0.0f32; b * c2 * fr_out * t2];
    for bi in 0..b {
        for fri in 0..fr_out {
            for ci in 0..c2 {
                for ti in 0..t2 {
                    unflat[((bi * c2 + ci) * fr_out + fri) * t2 + ti] =
                        dconv_out[((bi * fr_out + fri) * c2 + ci) * t2 + ti];
                }
            }
        }
    }
    // Conv2d 1x1
    let (cpu_rewritten, rw_shape) = ops_cpu::conv2d(&unflat, [b, c2, fr_out, t2], &layer.rewrite, &layer.rewrite_bias, 0, 0, 1, 1);
    eprintln!("after rewrite: shape {:?}", rw_shape);
    eprintln!("cpu rewritten[0..5]={:?}", &cpu_rewritten[..5]);

    // GPU: do the same flow.
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");
    let gpu_layer = &gpu_model.encoders[0];
    let gpu_in = state.upload_f32(&input, vec![b, c_in, fr, t]).expect("up");
    let gpu_h: GpuTensor = demucs_core_native::cuda_ops::conv2d_8x1_s4p2(&state, &gpu_in, &gpu_layer.conv, &gpu_layer.conv_bias).expect("conv2d");
    let gpu_h = demucs_core_native::cuda_ops::gelu_inplace(&state, gpu_h).expect("gelu");
    let gpu_h = demucs_core_native::cuda_ops::reshape_bcft_to_bfct(&state, &gpu_h).expect("rs");
    let gpu_h = demucs_core_native::cuda_ops::dconv(&state, gpu_h, &gpu_layer.dconv).expect("dconv");
    let gpu_h = demucs_core_native::cuda_ops::reshape_bfct_to_bcft(&state, &gpu_h, b).expect("rs2");
    let gpu_rw: GpuTensor = demucs_core_native::cuda_ops::conv2d_1x1(&state, &gpu_h, &gpu_layer.rewrite, &gpu_layer.rewrite_bias).expect("conv2d_1x1");
    let gpu_dl = state.download_to_f32(&gpu_rw).expect("dl");
    eprintln!("gpu rewritten[0..5]={:?}", &gpu_dl[..5]);

    let diff = max_diff(&cpu_rewritten, &gpu_dl);
    let max_val = cpu_rewritten.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let rms = (cpu_rewritten.iter().zip(&gpu_dl).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / cpu_rewritten.len() as f32).sqrt();
    let mut max_idx = 0;
    let mut max_d = 0.0f32;
    for (i, (&a, &b)) in cpu_rewritten.iter().zip(&gpu_dl).enumerate() {
        let d = (a - b).abs();
        if d > max_d {
            max_d = d;
            max_idx = i;
        }
    }
    eprintln!("post-rewrite max_diff at idx {}: cpu={} gpu={}", max_idx, cpu_rewritten[max_idx], gpu_dl[max_idx]);
    eprintln!("post-rewrite cpu vs gpu: max_diff={:.4}, rms={:.4}, max_val={:.2}", diff, rms, max_val);
    let tol = (max_val * 0.05).max(5e-1);
    assert!(diff < tol, "max_diff={:.4} exceeds tol={:.4}", diff, tol);
}
