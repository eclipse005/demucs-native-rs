#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_hdec2_real_12k() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("l");
    let m = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("m");
    let state = Arc::new(CudaState::new(0).expect("c"));
    let gm = GpuHTDemucs::from_cpu(&state, &m).expect("g");
    // hdec[2]: chin=96, chout=48, input fr=128, output fr=512 (convTr 128->512)
    let (b,c,fr,t) = (1,96,128,336);
    let x: Vec<f32> = (0..b*c*fr*t).map(|i| ((i as f32)*1e-3).sin()).collect();
    let skip: Vec<f32> = (0..b*c*fr*t).map(|i| ((i as f32)*1e-3+0.5).sin()).collect();
    let target = 512;
    let (cpu_out,_) = ops_cpu::hdec_layer_forward(&x,[b,c,fr,t],&skip,[b,c,fr,t],target,&m.decoders[2]);
    let gx = state.upload_f32(&x,vec![b,c,fr,t]).expect("x");
    let gs = state.upload_f32(&skip,vec![b,c,fr,t]).expect("s");
    let go = demucs_core_native::cuda_ops::hdec_layer(&state,gx,&gs,target,&gm.decoders[2]).expect("h");
    let gdl = state.download_to_f32(&go).expect("d");
    let be = |o:&[f32],lo:usize,hi:usize|->f64 { let mut e=0.0; for ci in 0..48 { for fi in lo.min(512)..hi.min(512) { for ti in 0..t {
        let v = o[(ci*512+fi)*t+ti] as f64; e+=v*v; }}} e };
    eprintln!("hdec[2] GPU vs CPU: low(0-100)={:.2}/{:.2} 279-372={:.2e}/{:.2e} ratio={:.2}",
        be(&cpu_out,0,100),be(&gdl,0,100),be(&cpu_out,279,372),be(&gdl,279,372),be(&gdl,279,372)/be(&cpu_out,279,372));
}
