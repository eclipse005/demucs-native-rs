//! hdec[3] whole layer at real shape, GPU vs CPU, 12-16kHz output bins.
//! Catches chain-interaction bugs that per-op isolated tests miss.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_hdec3_real_12k() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("model");
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");

    let (b, c, fr, t) = (1, 48, 512, 336);
    let x: Vec<f32> = (0..b*c*fr*t).map(|i| ((i as f32)*1e-3).sin()).collect();
    let skip: Vec<f32> = (0..b*c*fr*t).map(|i| ((i as f32)*1e-3 + 0.5).sin()).collect();
    let target = demucs_core_native::N_FFT / 2; // 2048

    let (cpu_out, _) = ops_cpu::hdec_layer_forward(&x, [b,c,fr,t], &skip, [b,c,fr,t], target, &cpu_model.decoders[3]);
    let gx = state.upload_f32(&x, vec![b,c,fr,t]).expect("up x");
    let gskip = state.upload_f32(&skip, vec![b,c,fr,t]).expect("up s");
    let gout = demucs_core_native::cuda_ops::hdec_layer(&state, gx, &gskip, target, &gpu_model.decoders[3]).expect("hdec");
    let gdl = state.download_to_f32(&gout).expect("dl");

    let fr_out = target;
    let c_out = 16;
    let be = |o:&[f32], lo:usize, hi:usize| -> f64 {
        let mut e=0.0; for ci in 0..c_out { for fi in lo..hi { for ti in 0..t {
            let v = o[(ci*fr_out+fi)*t+ti] as f64; e += v*v;
        }}} e
    };
    eprintln!("hdec[3] GPU vs CPU bins:");
    eprintln!("  low(0-512)  cpu={:.2e} gpu={:.2e} ratio={:.2}", be(&cpu_out,0,512), be(&gdl,0,512), be(&gdl,0,512)/be(&cpu_out,0,512));
    eprintln!("  12-16k(1114-1486) cpu={:.2e} gpu={:.2e} ratio={:.2}", be(&cpu_out,1114,1486), be(&gdl,1114,1486), be(&gdl,1114,1486)/be(&cpu_out,1114,1486));
}
