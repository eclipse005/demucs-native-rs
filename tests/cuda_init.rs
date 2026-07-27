//! CUDA initialization test — triggers NVRTC compile of all kernels and
//! confirms they load and run.

#![cfg(feature = "cuda")]

use demucs_core_native::cuda_engine::CudaState;

#[test]
fn cuda_state_init() {
    let state = CudaState::new(0).expect("CudaState::new should succeed");
    // Just hold the state for a moment to confirm it's valid.
    drop(state);
}

#[test]
fn cuda_state_init_with_full_kernel_load() {
    // Triggers NVRTC compile of all 20 kernels + module load.
    let _state = CudaState::new(0).expect("CudaState::new should succeed");
}
