#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::{ops_cpu, weights::WeightStore, N_FFT};

#[test]
#[ignore]
fn cuda_tx_real_bin45() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("l");
    let m = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("m");
    let state = Arc::new(CudaState::new(0).expect("c"));
    let gm = GpuHTDemucs::from_cpu(&state, &m).expect("g");
    let (fr, t, t2) = (8, 336, 1344);
    let d = m.crosstransformer.norm_in.dim;
    let freq: Vec<f32> = (0..1*384*fr*t).map(|i| ((i as f32)*1e-3).sin()).collect();
    let time: Vec<f32> = (0..1*384*t2).map(|i| ((i as f32)*1e-3).sin()).collect();
    let fpe = ops_cpu::sin_embed_2d(d, fr, t);
    let tpe = ops_cpu::sin_embed_1d(t2, d);
    // CPU
    let (cf, _, ct_, _) = ops_cpu::cross_domain_transformer_forward(&freq,[1,384,fr,t],&time,[1,384,t2],&m.crosstransformer);
    // GPU
    let gf = state.upload_f32(&freq,vec![1,384,fr,t]).expect("f");
    let gt = state.upload_f32(&time,vec![1,384,t2]).expect("t");
    let gfpe = state.upload_f32(&fpe,vec![1,t*fr,d]).expect("fpe");
    let gtpe = state.upload_f32(&tpe,vec![1,t2,d]).expect("tpe");
    let (gf_o, _) = demucs_core_native::cuda_ops::cross_domain_transformer(&state,&gf,&gt,&gm.crosstransformer,&gfpe,&gtpe).expect("tx");
    let gdl = state.download_to_f32(&gf_o).expect("d");
    // cf shape [1,384,8,336]. Compare per freq-bin energy (bin 4-5 -> 12-16kHz).
    let be = |o:&[f32], fi:usize| -> f64 { let mut e=0.0; for ci in 0..384 { for ti in 0..t { let v=o[((ci*fr+fi)*t+ti)] as f64; e+=v*v; }} e };
    for fi in 0..fr {
        eprintln!("TX bin {}: cpu={:.2e} gpu={:.2e} ratio={:.2}", fi, be(&cf,fi), be(&gdl,fi), be(&gdl,fi)/be(&cf,fi));
    }
}
