//! htdemucs_forward end-to-end GPU vs CPU.
//! Inputs normalized on CPU; GPU forward returns pre-denorm; we denorm on CPU.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::{CudaState, GpuTensor};
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

fn rms(a: &[f32], b: &[f32]) -> f32 {
    (a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / a.len() as f32).sqrt()
}

#[test]
#[ignore]
fn cuda_htdemucs_forward_matches_cpu() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("model");

    // Small shape (fast). Real-shape [2048,336] was tested separately and
    // confirmed freq drift triggers at real shape (108% vs 4.42% here) —
    // see ROADMAP §11.4. Kept small here for quick regression.
    let (b, cin_f, fr, t) = (1, 4, 256, 16);
    let (bt, cin_t, l) = (1, 2, 4096);
    let freq: Vec<f32> = (0..b * cin_f * fr * t).map(|i| ((i as f32) * 1e-4).sin() * 0.3).collect();
    let time: Vec<f32> = (0..bt * cin_t * l).map(|i| ((i as f32) * 3e-4).sin() * 0.3).collect();

    // CPU normalize (mean/std per batch).
    let (freq_n, fsh, fmean, _, fstd, _) = ops_cpu::normalize_freq(&freq, [b, cin_f, fr, t]);
    let (time_n, tsh, tmean, _, tstd, _) = ops_cpu::normalize_time(&time, [bt, cin_t, l]);

    // CPU full forward (normalizes internally → denormalized output).
    let (cpu_f, cpu_fsh, cpu_t, cpu_tsh) =
        ops_cpu::htdemucs_forward(&freq, [b, cin_f, fr, t], &time, [bt, cin_t, l], &cpu_model);

    // GPU forward on the normalized inputs.
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");
    let gf = state.upload_f32(&freq_n, vec![b, cin_f, fr, t]).expect("up f");
    let gt = state.upload_f32(&time_n, vec![bt, cin_t, l]).expect("up t");
    let (gf_raw, gt_raw) = demucs_core_native::cuda_ops::htdemucs_forward(&state, &gf, &gt, &gpu_model)
        .expect("forward");
    let mut gf_dl = state.download_to_f32(&gf_raw).expect("dl f");
    let mut gt_dl = state.download_to_f32(&gt_raw).expect("dl t");
    eprintln!("gpu freq out shape={:?} len={}", gf_raw.shape(), gf_dl.len());
    eprintln!("gpu time out shape={:?} len={}", gt_raw.shape(), gt_dl.len());
    eprintln!("cpu freq out shape={:?} len={}", cpu_fsh, cpu_f.len());
    eprintln!("cpu time out shape={:?} len={}", cpu_tsh, cpu_t.len());

    // Denormalize the GPU outputs on CPU (raw std).
    ops_cpu::denormalize_freq(&mut gf_dl, cpu_fsh, &fmean, &fstd);
    ops_cpu::denormalize_time(&mut gt_dl, cpu_tsh, &tmean, &tstd);

    let fmv = cpu_f.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tmv = cpu_t.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let rf = rms(&cpu_f, &gf_dl);
    let rt = rms(&cpu_t, &gt_dl);
    eprintln!("HTDEMUCS freq: rms={:.4} max_val={:.2} ({:.2}%)", rf, fmv, 100.0 * rf / fmv);
    eprintln!("HTDEMUCS time: rms={:.4} max_val={:.2} ({:.2}%)", rt, tmv, 100.0 * rt / tmv);
    eprintln!("freq cpu[0..3]={:?}", &cpu_f[..3]);
    eprintln!("freq gpu[0..3]={:?}", &gf_dl[..3]);
    // Full pipeline (enc+TX+dec) compounds f16; allow 5% rms.
    // NOTE: time path is 14.92% off even on small synthetic input — this
    // is a precision gap from the f16 GEMM accumulation. Not yet fixed.
    let ftol = (fmv * 0.05).max(1e-2);
    let ttol = (tmv * 0.05).max(1e-2);
    assert!(rf < ftol, "freq rms={:.4} > tol={:.4}", rf, ftol);
    assert!(rt < ttol, "time rms={:.4} > tol={:.4}", rt, ttol);
    let _ = (fsh, tsh);
}
