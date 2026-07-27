#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::{ops_cpu, weights::WeightStore, N_FFT};

#[test]
#[ignore]
fn cuda_hdec01_real() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("l");
    let m = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("m");
    let state = Arc::new(CudaState::new(0).expect("c"));
    let gm = GpuHTDemucs::from_cpu(&state, &m).expect("g");
    for idx in 0..2 {
        let chin = 384 / (1 << idx); // hdec[0]:384, hdec[1]:192
        let chout = 384 / (1 << (idx+1)); // hdec[0]:192, hdec[1]:96
        let fr_in = 8 * (1 << idx); // hdec[0]:8, hdec[1]:32
        let fr_out = 8 * (1 << (idx+1)); // hdec[0]:32, hdec[1]:128
        let t = 336;
        let x: Vec<f32> = (0..chin*fr_in*t).map(|i| ((i as f32)*1e-3).sin()).collect();
        let skip: Vec<f32> = (0..chin*fr_in*t).map(|i| ((i as f32)*1e-3+0.5).sin()).collect();
        let (cpu_out,_) = ops_cpu::hdec_layer_forward(&x,[1,chin,fr_in,t],&skip,[1,chin,fr_in,t],fr_out,&m.decoders[idx]);
        let gx = state.upload_f32(&x,vec![1,chin,fr_in,t]).expect("x");
        let gs = state.upload_f32(&skip,vec![1,chin,fr_in,t]).expect("s");
        let go = demucs_core_native::cuda_ops::hdec_layer(&state,gx,&gs,fr_out,&gm.decoders[idx]).expect("h");
        let gdl = state.download_to_f32(&go).expect("d");
        let md = cpu_out.iter().zip(&gdl).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
        let rms = (cpu_out.iter().zip(&gdl).map(|(a,b)|(a-b).powi(2)).sum::<f32>()/cpu_out.len() as f32).sqrt();
        let cm = cpu_out.iter().map(|v|v.abs()).fold(0.0f32,f32::max);
        eprintln!("hdec[{}] real: max_diff={:.4} rms={:.4} cpu_max={:.2} rms/cpu_max={:.1}%", idx, md, rms, cm, 100.0*rms/cm);
    }
}
