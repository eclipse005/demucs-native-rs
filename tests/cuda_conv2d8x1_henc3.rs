#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_conv2d8x1_henc3_shape() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("l");
    let m = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("m");
    let state = Arc::new(CudaState::new(0).expect("c"));
    let gm = GpuHTDemucs::from_cpu(&state, &m).expect("g");
    // henc[3] conv: in [1,192,32,336] → out [1,384,8,336]
    let (b, cin, fr, t) = (1, 192, 32, 336);
    let input: Vec<f32> = (0..b*cin*fr*t).map(|i| ((i as f32)*1e-4).sin()*0.05).collect();
    // CPU
    let (cpu_out, _) = ops_cpu::conv2d(&input, [b,cin,fr,t], &m.encoders[3].conv, &m.encoders[3].conv_bias, 2, 0, 4, 1);
    // GPU
    let gi = state.upload_f32(&input, vec![b,cin,fr,t]).expect("up");
    let go = demucs_core_native::cuda_ops::conv2d_8x1_s4p2(&state, &gi, &gm.encoders[3].conv, &gm.encoders[3].conv_bias).expect("c");
    let gdl = state.download_to_f32(&go).expect("dl");
    let md = cpu_out.iter().zip(&gdl).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    let rms = (cpu_out.iter().zip(&gdl).map(|(a,b)|(a-b).powi(2)).sum::<f32>()/cpu_out.len() as f32).sqrt();
    let cm = cpu_out.iter().map(|v|v.abs()).fold(0.0f32, f32::max);
    eprintln!("conv2d_8x1 henc[3] shape: max_diff={:.6} rms={:.6} cpu_max={:.4} rms/cpu_max={:.2}%", md, rms, cm, 100.0*rms/cm);
}
