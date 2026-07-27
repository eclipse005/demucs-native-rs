//! conv_transpose2d at hdec[3] shape (fr 512->2048) — GPU vs CPU,
//! checking the 12-16kHz output bins (1114-1486) where native has 23x python.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_convtr_hdec3_12k_bins() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("model");
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");

    // hdec[3]: chin=48, chout=16, conv_tr [8,1] stride4 pad2, fr 512->2048.
    let b = 1; let c_in = 48; let fr = 512; let t = 336;
    let input: Vec<f32> = (0..b * c_in * fr * t).map(|i| ((i as f32) * 1e-3).sin() * 1.0).collect();
    let x_shape = [b, c_in, fr, t];

    let (cpu_out, _) = ops_cpu::conv_transpose2d(
        &input, x_shape, &cpu_model.decoders[3].conv_tr, &cpu_model.decoders[3].conv_tr_bias, 2, 0, 4, 1,
    );
    // cpu_out shape [1, 16, 2048, 336]. Check 12-16kHz bins (freq dim 1114-1486) energy vs low bins.
    let fr_out = 2048;
    let c_out = 16;
    let bins_energy = |out: &[f32], lo: usize, hi: usize| -> f64 {
        let mut e = 0.0f64;
        for bi in 0..c_out {
            for fi in lo..hi {
                for ti in 0..t {
                    let v = out[(bi * fr_out + fi) * t + ti] as f64;
                    e += v * v;
                }
            }
        }
        e
    };
    let cpu_low = bins_energy(&cpu_out, 0, 512);
    let cpu_mid = bins_energy(&cpu_out, 512, 1114);
    let cpu_hi = bins_energy(&cpu_out, 1114, 1486);  // 12-16kHz
    eprintln!("CPU convTr bins: low(0-512)={:.2e} mid(512-1114)={:.2e} hi12-16k(1114-1486)={:.2e} ratio_hi/low={:.3}", cpu_low, cpu_mid, cpu_hi, cpu_hi/cpu_low);

    let gpu_in = state.upload_f32(&input, vec![b, c_in, fr, t]).expect("up");
    let gpu_out = demucs_core_native::cuda_ops::conv_transpose2d_8x1_s4p2(
        &state, &gpu_in, &gpu_model.decoders[3].conv_tr, &gpu_model.decoders[3].conv_tr_bias,
    ).expect("convTr");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    let gpu_low = bins_energy(&gpu_dl, 0, 512);
    let gpu_mid = bins_energy(&gpu_dl, 512, 1114);
    let gpu_hi = bins_energy(&gpu_dl, 1114, 1486);
    eprintln!("GPU convTr bins: low={:.2e} mid={:.2e} hi12-16k={:.2e} ratio_hi/low={:.3}", gpu_low, gpu_mid, gpu_hi, gpu_hi/gpu_low);
    eprintln!("GPU/CPU hi12-16k ratio = {:.2}x (if >>1, convTr injects high-freq noise)", gpu_hi / cpu_hi);
}
