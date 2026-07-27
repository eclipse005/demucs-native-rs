//! Debug: verify gemm_f16 produces correct results for simple test cases.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::CudaState;

#[test]
#[ignore]
fn cuda_gemm_identity() {
    // C = A @ I = A. A is [2, 3], I is [3, 3] (3x3 identity).
    // Expected C = A.
    let state = Arc::new(CudaState::new(0).expect("cuda init"));
    let a: Vec<f32> = vec![
        1.0, 2.0, 3.0, // row 0
        4.0, 5.0, 6.0, // row 1
    ];
    let i_mat: Vec<f32> = vec![
        1.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, // row 1
        0.0, 0.0, 1.0, // row 2
    ];
    let gpu_a = state.upload_f32(&a, vec![2, 3]).expect("a");
    let gpu_i = state.upload_f32(&i_mat, vec![3, 3]).expect("i");
    let gpu_c = state
        .gemm_f16(&gpu_a, &gpu_i, 2, 3, 3)
        .expect("gemm");
    let c = state.download_to_f32(&gpu_c).expect("dl");
    eprintln!("C (should be [[1,2,3],[4,5,6]]) = {:?}", &c);
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for (i, (&g, &e)) in c.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-2, "idx {i}: gpu={g} cpu={e}");
    }
}

#[test]
#[ignore]
fn cuda_gemm_3x4_4x2() {
    // C = A @ B where A is [3, 4], B is [4, 2]. C is [3, 2].
    let state = Arc::new(CudaState::new(0).expect("cuda init"));
    let a: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
        9.0, 10.0, 11.0, 12.0, // row 2
    ];
    let b: Vec<f32> = vec![
        1.0, 2.0, // row 0
        3.0, 4.0, // row 1
        5.0, 6.0, // row 2
        7.0, 8.0, // row 3
    ];
    // C[0, 0] = 1*1+2*3+3*5+4*7 = 50
    // C[0, 1] = 1*2+2*4+3*6+4*8 = 60
    // C[1, 0] = 5*1+6*3+7*5+8*7 = 130
    // ...
    let expected: Vec<f32> = vec![50.0, 60.0, 114.0, 140.0, 178.0, 220.0];
    let gpu_a = state.upload_f32(&a, vec![3, 4]).expect("a");
    let gpu_b = state.upload_f32(&b, vec![4, 2]).expect("b");
    let gpu_c = state.gemm_f16(&gpu_a, &gpu_b, 3, 2, 4).expect("gemm");
    let c = state.download_to_f32(&gpu_c).expect("dl");
    eprintln!("C = {:?}", &c);
    for (i, (&g, &e)) in c.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1.0, "idx {i}: gpu={g} expected={e}");
    }
}