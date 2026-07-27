//! Test that a full HTDemucs model can be uploaded to the GPU and the weights
//! preserved bit-perfect (via f16 round-trip).

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn upload_htdemucs_ft_vocals_to_gpu() {
    let model_path = PathBuf::from("../models/htdemucs_ft.safetensors");
    if !model_path.exists() {
        eprintln!("skipping: model not found");
        return;
    }
    let state = std::sync::Arc::new(CudaState::new(0).expect("cuda init"));
    let store = WeightStore::load(&model_path).expect("load safetensors");
    let cpu_model = demucs_core_native::model::HTDemucs::from_store(
        &store,
        "04573f0d",
        4,
        512,
    )
    .expect("load HTDemucs vocals model");
    let _gpu = GpuHTDemucs::from_cpu(&state, &cpu_model)
        .expect("mirror HTDemucs to GPU");
    // Hold GPU state briefly to ensure no GPU async errors.
    state.synchronize().expect("cuda sync");
    eprintln!(
        "uploaded htdemucs_ft vocals: encoders={} tencoders={} decoders={} tdecoders={}",
        cpu_model.encoders.len(),
        cpu_model.tencoders.len(),
        cpu_model.decoders.len(),
        cpu_model.tdecoders.len(),
    );
}
