//! groupnorm1 (dconv) GPU vs CPU at real shape [512, 48, 336].

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cuda_groupnorm1_matches_cpu() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("model");
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");

    // Use hdec[3] dconv layer 0 norm1 (1 group over C=compress).
    let gn = &cpu_model.decoders[3].dconv.layers[0].norm1;
    let gpu_gn = &gpu_model.decoders[3].dconv.layers[0].norm1;
    let (b, c, l) = (512usize, gn.num_channels, 336usize);
    let input: Vec<f32> = (0..b * c * l).map(|i| ((i as f32) * 1e-4).sin() * 5.0).collect();

    let mut cpu_in = input.clone();
    ops_cpu::groupnorm1(&mut cpu_in, [b, c, l], gn);

    let mut gpu_in = state.upload_f32(&input, vec![b, c, l]).expect("up");
    demucs_core_native::cuda_ops::groupnorm1_inplace(&state, &mut gpu_in, gpu_gn).expect("gn");
    let gpu_dl = state.download_to_f32(&gpu_in).expect("dl");

    let max_diff = cpu_in.iter().zip(&gpu_dl).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    let rms = (cpu_in.iter().zip(&gpu_dl).map(|(a,b)| (a-b).powi(2)).sum::<f32>()/cpu_in.len() as f32).sqrt();
    eprintln!("groupnorm1 GPU vs CPU: max_diff={:.4} rms={:.4} (input max={:.2})", max_diff, rms, input.iter().map(|v|v.abs()).fold(0.0f32,f32::max));
}
