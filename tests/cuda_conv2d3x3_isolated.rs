//! conv2d_3x3 (hdec rewrite) isolated GPU vs CPU — check 12-16kHz bins.
//! hdec[3] rewrite: chin=48 -> 96, k=[3,3] s=1 p=1.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_conv2d3x3_hdec3_bins() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("model");
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");

    let (b, c_in, fr, t) = (1, 48, 512, 336);
    let input: Vec<f32> = (0..b * c_in * fr * t).map(|i| ((i as f32) * 1e-3).sin() * 1.0).collect();
    let (cpu_out, _) = ops_cpu::conv2d(&input, [b, c_in, fr, t], &cpu_model.decoders[3].rewrite, &cpu_model.decoders[3].rewrite_bias, 1, 1, 1, 1);
    let gpu_in = state.upload_f32(&input, vec![b, c_in, fr, t]).expect("up");
    let gpu_out = demucs_core_native::cuda_ops::conv2d_3x3_s1p1(&state, &gpu_in, &gpu_model.decoders[3].rewrite, &gpu_model.decoders[3].rewrite_bias).expect("conv");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");

    let c_out = 96;
    // energy in freq bins 279-372 (maps to 12-16kHz after hdec[3] convTr) vs low bins
    let be = |out: &[f32], lo: usize, hi: usize| -> f64 {
        let mut e = 0.0f64;
        for ci in 0..c_out { for fi in lo..hi { for ti in 0..t {
            let v = out[(ci * fr + fi) * t + ti] as f64; e += v * v;
        }}}
        e
    };
    eprintln!("conv2d_3x3 GPU vs CPU bins:");
    eprintln!("  low(0-100)  cpu={:.2e} gpu={:.2e} ratio={:.2}", be(&cpu_out,0,100), be(&gpu_dl,0,100), be(&gpu_dl,0,100)/be(&cpu_out,0,100));
    eprintln!("  12-16k(279-372) cpu={:.2e} gpu={:.2e} ratio={:.2}", be(&cpu_out,279,372), be(&gpu_dl,279,372), be(&gpu_dl,279,372)/be(&cpu_out,279,372));
    // element-wise max diff
    let md = cpu_out.iter().zip(&gpu_dl).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    eprintln!("  overall max_diff={:.4}", md);
}
