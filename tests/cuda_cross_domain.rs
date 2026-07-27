//! Cross-domain transformer GPU vs CPU (full 5+5 layers).

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
fn cuda_cross_domain_transformer_matches_cpu() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("model");
    let ct = &cpu_model.crosstransformer;
    let d_model = ct.norm_in.dim;
    let bottleneck = ct.channel_upsampler.as_ref().expect("upsampler").in_ch;

    let fr = 4;
    let t = 16;
    let t2 = 64; // time sequence length (independent of fr*t)
    let freq: Vec<f32> = (0..bottleneck * fr * t).map(|i| ((i as f32) * 0.0002 - 0.1)).collect();
    let time: Vec<f32> = (0..bottleneck * t2).map(|i| ((i as f32) * 0.0003 - 0.1)).collect();

    // Sinusoidal PEs (CPU-computed, uploaded).
    let freq_pe = ops_cpu::sin_embed_2d(d_model, fr, t); // [t*fr, d_model]
    let time_pe = ops_cpu::sin_embed_1d(t2, d_model);    // [t2, d_model]

    // CPU reference.
    let (cpu_f, _, cpu_t, _) =
        ops_cpu::cross_domain_transformer_forward(&freq, [1, bottleneck, fr, t], &time, [1, bottleneck, t2], ct);

    // GPU.
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");
    let gfreq = state.upload_f32(&freq, vec![1, bottleneck, fr, t]).expect("up f");
    let gtime = state.upload_f32(&time, vec![1, bottleneck, t2]).expect("up t");
    let gfpe = state.upload_f32(&freq_pe, vec![1, t * fr, d_model]).expect("up fpe");
    let gtpe = state.upload_f32(&time_pe, vec![1, t2, d_model]).expect("up tpe");
    let (gf, gt) = demucs_core_native::cuda_ops::cross_domain_transformer(
        &state, &gfreq, &gtime, &gpu_model.crosstransformer, &gfpe, &gtpe,
    )
    .expect("cdt");
    let gf_dl = state.download_to_f32(&gf).expect("dl f");
    let gt_dl = state.download_to_f32(&gt).expect("dl t");

    let fmv = cpu_f.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tmv = cpu_t.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let rf = rms(&cpu_f, &gf_dl);
    let rt = rms(&cpu_t, &gt_dl);
    eprintln!("CDT freq: rms={:.4} max_val={:.2} ({:.2}%)", rf, fmv, 100.0 * rf / fmv);
    eprintln!("CDT time: rms={:.4} max_val={:.2} ({:.2}%)", rt, tmv, 100.0 * rt / tmv);
    eprintln!("freq cpu[0..3]={:?}", &cpu_f[..3]);
    eprintln!("freq gpu[0..3]={:?}", &gf_dl[..3]);
    // 10 transformer layers compound f16 error; allow ~15%.
    let ftol = (fmv * 0.15).max(1e-2);
    let ttol = (tmv * 0.15).max(1e-2);
    assert!(rf < ftol, "freq rms={:.4} > tol={:.4}", rf, ftol);
    assert!(rt < ttol, "time rms={:.4} > tol={:.4}", rt, ttol);
}
