#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_henc0_real() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("l");
    let m = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("m");
    let state = Arc::new(CudaState::new(0).expect("c"));
    let gm = GpuHTDemucs::from_cpu(&state, &m).expect("g");
    // henc[0]: input [1,4,2048,336] (real shape), output [1,48,512,336]
    let (b,c,fr,t) = (1,4,2048,336);
    let input: Vec<f32> = (0..b*c*fr*t).map(|i| ((i as f32)*1e-4).sin()*100.0).collect();
    let (cpu_out,_) = ops_cpu::henc_layer_forward(&input,[b,c,fr,t],&m.encoders[0]);
    let gi = state.upload_f32(&input,vec![b,c,fr,t]).expect("i");
    let go = demucs_core_native::cuda_ops::henc_layer(&state,gi,&gm.encoders[0]).expect("h");
    let gdl = state.download_to_f32(&go).expect("d");
    let md = cpu_out.iter().zip(&gdl).map(|(a,b)| (a-b).abs()).fold(0.0f32,f32::max);
    let rms = (cpu_out.iter().zip(&gdl).map(|(a,b)|(a-b).powi(2)).sum::<f32>()/cpu_out.len() as f32).sqrt();
    let cm = cpu_out.iter().map(|v|v.abs()).fold(0.0f32,f32::max);
    eprintln!("henc[0] real GPU vs CPU: max_diff={:.4} rms={:.4} cpu_max={:.2} (rms/cpu_max={:.1}%)", md, rms, cm, 100.0*rms/cm);
}
