//! Verify the strided-batched GEMM matches CPU for C = A @ B over batches.
//! Config is the direct batched generalization of the proven `gemm_f16`
//! arg-swap pattern: m_cublas=n, n_cublas=m, k_cublas=k, lda=n, ldb=k,
//! ldc=n, B passed first, A second, stride_a=k*n, stride_b=m*k.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::CudaState;
use cudarc::cublas::sys;

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
#[ignore]
fn batched_gemm_matches_cpu() {
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let batch = 3;
    let m = 5;
    let k = 4;
    let n = 6;

    // A: [batch, m, k], B: [batch, k, n], row-major.
    let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.01 - 0.3).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.02 - 0.1).collect();

    // CPU reference: C[bi, mi, ni] = sum_ki A[bi,mi,ki] * B[bi,ki,ni].
    let mut cpu = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        for mi in 0..m {
            for ni in 0..n {
                let mut s = 0.0f32;
                for ki in 0..k {
                    s += a[bi * m * k + mi * k + ki] * b[bi * k * n + ki * n + ni];
                }
                cpu[bi * m * n + mi * n + ni] = s;
            }
        }
    }

    let a_gpu = state.upload_f32(&a, vec![batch * m * k]).expect("up a");
    let b_gpu = state.upload_f32(&b, vec![batch * k * n]).expect("up b");

    // C = A @ B: B is first arg (cublasA), A is second (cublasB).
    let out = state
        .gemm_strided_batched_f16(
            &b_gpu.data, // cublasA = B
            &a_gpu.data, // cublasB = A
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_N,
            batch,
            n,   // m_cublas = n
            m,   // n_cublas = m
            k,   // k_cublas = k
            n,   // lda = n  (B row-major cols)
            k,   // ldb = k  (A row-major cols)
            k * n, // stride_a (B per-batch)
            m * k, // stride_b (A per-batch)
        )
        .expect("gemm");

    let dl = state.download_to_f32(&out).expect("dl");
    let diff = max_diff(&cpu, &dl);
    let max_val = cpu.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("batched gemm: max_diff={:.4} max_val={:.2}", diff, max_val);
    eprintln!("cpu[0..3]={:?}", &cpu[..3]);
    eprintln!("gpu[0..3]={:?}", &dl[..3]);
    let tol = (max_val * 0.05).max(5e-2);
    assert!(diff < tol, "max_diff={diff} > tol={tol}");
}
