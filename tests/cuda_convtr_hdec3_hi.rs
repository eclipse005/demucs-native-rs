#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_convtr_hdec3_16_22k() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("l");
    let m = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("m");
    let state = Arc::new(CudaState::new(0).expect("c"));
    let gm = GpuHTDemucs::from_cpu(&state, &m).expect("g");
    let (b,c_in,fr,t) = (1,48,512,336);
    let input: Vec<f32> = (0..b*c_in*fr*t).map(|i| ((i as f32)*1e-3).sin()).collect();
    let (cpu_out,_) = ops_cpu::conv_transpose2d(&input,[b,c_in,fr,t],&m.decoders[3].conv_tr,&m.decoders[3].conv_tr_bias,2,0,4,1);
    let gi = state.upload_f32(&input,vec![b,c_in,fr,t]).expect("i");
    let go = demucs_core_native::cuda_ops::conv_transpose2d_8x1_s4p2(&state,&gi,&gm.decoders[3].conv_tr,&gm.decoders[3].conv_tr_bias).expect("c");
    let gdl = state.download_to_f32(&go).expect("d");
    let c_out = 16; let fr_out = 2048;
    let be = |o:&[f32], lo:usize, hi:usize| -> f64 {
        let mut e=0.0; for ci in 0..c_out { for fi in lo..hi.min(fr_out) { for ti in 0..t {
            let v = o[(ci*fr_out+fi)*t+ti] as f64; e += v*v;
        }}} e
    };
    for (nm,lo,hi) in [("low",0,512),("mid",512,1114),("12-16k",1114,1486),("16-22k",1486,2048)] {
        let ce=be(&cpu_out,lo,hi); let ge=be(&gdl,lo,hi);
        eprintln!("convTr {:<8}({}-{}): cpu={:e} gpu={:e} gpu/cpu={:.2}x", nm, lo, hi, ce, ge, ge/ce);
    }
}
