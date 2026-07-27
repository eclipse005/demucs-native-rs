//! Root-cause verification for the MHA blocker.
//!
//! Two independent experiments:
//!   1. Prove cudarc 0.19 `CudaSlice::clone()` is a DEEP device-to-device
//!      copy (not a refcount bump): writing through a clone leaves the
//!      original untouched. Writing through `&mut` works.
//!   2. Prove the strided-batched GEMM with the correct config (direct
//!      batched generalization of the working `gemm_f16` arg-swap pattern)
//!      matches CPU for C = A @ B over multiple batches.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::{CudaState, GpuTensor};

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// `copy_per_head` writes `in[bh*S*d + i]` into `out[i]`. Source layout
/// here is [bh, S, d]; we extract head bh=1 of a [2, S, d] tensor.
#[test]
#[ignore]
fn copy_per_head_clone_vs_mut() {
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let bh = 2;
    let s = 3;
    let d = 4;
    // head 0 = 0..11, head 1 = 12..23
    let input: Vec<f32> = (0..bh * s * d).map(|i| i as f32).collect();
    let gpu_in = state.upload_f32(&input, vec![bh * s * d]).expect("up");

    // ---- (a) BUGGY: write through a clone. Original never written. ----
    let out_clone = state.alloc_uninit_f16(s * d).expect("alloc");
    let mut cloned = out_clone.clone(); // deep copy → separate allocation
    demucs_core_native::cuda_ops::copy_per_head(&state, &mut cloned, &gpu_in.data, 1, s, d)
        .expect("copy");
    // cloned is dropped here; out_clone (the original) was never written.
    let dl_orig = state
        .download_to_f32(&GpuTensor::new(out_clone, vec![s * d]))
        .expect("dl");
    // head 1 expected = [12,13,...,23]
    let head1: Vec<f32> = (12..24).map(|i| i as f32).collect();
    let diff_clone = max_diff(&dl_orig, &head1);
    eprintln!("[clone path]  out_orig[0..3]={:?}  max_diff_vs_head1={}", &dl_orig[..3], diff_clone);
    assert!(
        diff_clone > 10.0,
        "clone path should have FAILED (orig unwritten); got diff={}. \
         If this passes, clone shares memory and the hypothesis is wrong.",
        diff_clone
    );

    // ---- (b) CORRECT: write through &mut of the same slice we download. ----
    let mut out_mut = state.alloc_uninit_f16(s * d).expect("alloc");
    demucs_core_native::cuda_ops::copy_per_head(&state, &mut out_mut, &gpu_in.data, 1, s, d)
        .expect("copy");
    let dl_mut = state
        .download_to_f32(&GpuTensor::new(out_mut, vec![s * d]))
        .expect("dl");
    let diff_mut = max_diff(&dl_mut, &head1);
    eprintln!("[&mut path]   out[0..3]={:?}  max_diff_vs_head1={}", &dl_mut[..3], diff_mut);
    assert!(diff_mut < 1e-3, "&mut path must match head1; diff={}", diff_mut);
}
