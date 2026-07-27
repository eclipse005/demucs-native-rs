#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_tdec3_real() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("l");
    let m = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("m");
    let state = Arc::new(CudaState::new(0).expect("c"));
    let gm = GpuHTDemucs::from_cpu(&state, &m).expect("g");
    // tdec[3]: chin=48, chout=8, input t=85995, output t=343980
    let (b,c,ti) = (1,48,85995);
    let x: Vec<f32> = (0..b*c*ti).map(|i| ((i as f32)*1e-5).sin()).collect();
    let skip: Vec<f32> = (0..b*c*ti).map(|i| ((i as f32)*1e-5+0.3).sin()).collect();
    let target = 343980;
    let (cpu_out,_) = ops_cpu::tdec_layer_forward(&x,[b,c,ti],&skip,[b,c,ti],target,&m.tdecoders[3]);
    let gx = state.upload_f32(&x,vec![b,c,ti]).expect("x");
    let gs = state.upload_f32(&skip,vec![b,c,ti]).expect("s");
    let go = demucs_core_native::cuda_ops::tdec_layer(&state,gx,&gs,target,&gm.tdecoders[3]).expect("t");
    let gdl = state.download_to_f32(&go).expect("d");
    let md = cpu_out.iter().zip(&gdl).map(|(a,b)| (a-b).abs()).fold(0.0f32,f32::max);
    let rms = (cpu_out.iter().zip(&gdl).map(|(a,b)|(a-b).powi(2)).sum::<f32>()/cpu_out.len() as f32).sqrt();
    let cm = cpu_out.iter().map(|v|v.abs()).fold(0.0f32,f32::max);
    eprintln!("tdec[3] real GPU vs CPU: max_diff={:.4} rms={:.4} cpu_max={:.2} (rms/cpu_max={:.1}%)", md, rms, cm, 100.0*rms/cm);
}
