//! hdec[3] dconv (real shape [512,48,336]) GPU vs CPU, per-freq bins 279-372.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_dconv_hdec3_matches_cpu() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("model");
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");

    // hdec[3] dconv input: flattened [B*Fr, C, T] = [512, 48, 336].
    let (bf, c, t) = (512, 48, 336);
    let input: Vec<f32> = (0..bf * c * t).map(|i| ((i as f32) * 1e-3).sin() * 1.0).collect();

    let gpu_in = state.upload_f32(&input, vec![bf, c, t]).expect("up");
    let gpu_out = demucs_core_native::cuda_ops::dconv(&state, gpu_in, &gpu_model.decoders[3].dconv).expect("dconv");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");

    let (cpu_out, _) = ops_cpu::dconv_forward(&input, [bf, c, t], &cpu_model.decoders[3].dconv);

    // both are [512*48*336] flat. Compare per freq-bin (freq = row index / (c*t))
    let per_freq = c * t;
    let bin_energy = |out: &[f32], fi: usize| -> f64 {
        let s = fi * per_freq;
        out[s..s + per_freq].iter().map(|v| (*v as f64).powi(2)).sum()
    };
    let cpu_279_372: f64 = (279..372).map(|fi| bin_energy(&cpu_out, fi)).sum();
    let gpu_279_372: f64 = (279..372).map(|fi| bin_energy(&gpu_dl, fi)).sum();
    let cpu_low: f64 = (0..100).map(|fi| bin_energy(&cpu_out, fi)).sum();
    let gpu_low: f64 = (0..100).map(|fi| bin_energy(&gpu_dl, fi)).sum();
    let md = cpu_out.iter().zip(&gpu_dl).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    eprintln!("dconv hdec[3] GPU vs CPU: overall max_diff={:.4}", md);
    eprintln!("  low(0-100) cpu={:.2e} gpu={:.2e} ratio={:.2}", cpu_low, gpu_low, gpu_low/cpu_low);
    eprintln!("  12-16k(279-372) cpu={:.2e} gpu={:.2e} ratio={:.2}", cpu_279_372, gpu_279_372, gpu_279_372/cpu_279_372);
}
