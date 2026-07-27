//! CUDA ops correctness tests — for each op, run on CPU and GPU, compare
//! outputs to within f16 tolerance (1e-2 absolute, or 5% relative).

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::{CudaState, GpuTensor};
use demucs_core_native::gpu_model::GpuBias;
use demucs_core_native::ops_cpu;

/// Compute max abs diff between two f32 vectors.
fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
#[ignore]
fn cuda_upload_download_roundtrip() {
    let state = Arc::new(CudaState::new(0).expect("cuda init"));
    let n = 1024usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let gpu = state
        .upload_f32(&input, vec![n])
        .expect("upload");
    let downloaded = state.download_to_f32(&gpu).expect("download");
    assert_eq!(downloaded.len(), input.len());
    let diff = max_diff(&input, &downloaded);
    // f16 has ~3 decimal digits of precision; values in [-1, 1] → max_diff ~ 1e-3.
    assert!(
        diff < 1e-2,
        "f16 round-trip max_diff={diff:.6} exceeds 1e-2"
    );
}

#[test]
#[ignore]
fn cuda_gelu_matches_cpu() {
    let state = Arc::new(CudaState::new(0).expect("cuda init"));
    let n = 1024usize;
    // Mix of small and large values to exercise saturation.
    let input: Vec<f32> = (0..n)
        .map(|i| (i as f32 - 512.0) / 100.0) // range [-5.12, 5.11]
        .collect();

    // CPU reference (in-place).
    let mut cpu_in = input.clone();
    ops_cpu::gelu(&mut cpu_in);

    // GPU: upload, run cuda_ops::gelu_inplace, download, compare.
    let gpu_in = state.upload_f32(&input, vec![n]).expect("upload");
    let gpu_out: GpuTensor = demucs_core_native::cuda_ops::gelu_inplace(&state, gpu_in)
        .expect("cuda gelu");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("download");

    let diff = max_diff(&cpu_in, &gpu_dl);
    let max_val = cpu_in.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tol = (max_val * 1e-2).max(5e-2);
    assert!(
        diff < tol,
        "GELU cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})\n  cpu={:?}\n  gpu={:?}",
        &cpu_in[..5],
        &gpu_dl[..5]
    );
}

#[test]
#[ignore]
fn cuda_layer_scale_matches_cpu() {
    use demucs_core_native::model::LayerScale;
    let state = Arc::new(CudaState::new(0).expect("cuda init"));
    let c = 64usize;
    let l = 32usize;
    let input: Vec<f32> = (0..c * l).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let scale: Vec<f32> = (0..c).map(|i| 0.5 + (i as f32) * 0.02).collect();

    // CPU reference.
    let mut cpu_in = input.clone();
    ops_cpu::layer_scale(&mut cpu_in, [1, c, l], &LayerScale { scale: scale.clone() });
    let cpu_out = cpu_in;

    // GPU.
    use demucs_core_native::gpu_model::GpuLayerScale;
    let gpu_scale = GpuLayerScale::from_cpu(&state, &LayerScale { scale: scale.clone() })
        .expect("upload scale");
    let gpu_in = state
        .upload_f32(&input, vec![1, c, l])
        .expect("upload");
    let gpu_out: GpuTensor =
        demucs_core_native::cuda_ops::layer_scale_inplace(&state, gpu_in, &gpu_scale)
            .expect("cuda layer_scale");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("download");

    let diff = max_diff(&cpu_out, &gpu_dl);
    // f16 round-trip at values up to ~50 has precision ~5e-2. Tolerate that.
    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tol = (max_val * 1e-2).max(5e-2);
    assert!(
        diff < tol,
        "layer_scale cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})"
    );
}

#[test]
#[ignore]
fn cuda_add_to_matches_cpu() {
    let state = Arc::new(CudaState::new(0).expect("cuda init"));
    let n = 2048usize;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.0005).collect();

    // CPU reference (computed here directly since ops_cpu doesn't have an
    // add helper; we'll compare against simple a + b).
    let cpu_out: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let gpu_a = state.upload_f32(&a, vec![n]).expect("up a");
    let gpu_b = state.upload_f32(&b, vec![n]).expect("up b");
    let gpu_out: GpuTensor =
        demucs_core_native::cuda_ops::add_to(&state, &gpu_a, &gpu_b).expect("cuda add_to");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("download");

    let diff = max_diff(&cpu_out, &gpu_dl);
    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tol = (max_val * 1e-2).max(5e-2);
    assert!(
        diff < tol,
        "add_to cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})"
    );
}

#[test]
#[ignore]
fn cuda_im2col_8x1_s4p2_matches_cpu() {
    // im2col for HEncLayer/TEncLayer conv: kernel [8,1], stride [4,1], pad [2,0].
    // Input shape: [B=2, C_in=3, H=12, W=5]
    // Output shape: [B*H_out*W_out=12, C_in*8*1=24]
    let b = 2;
    let c_in = 3;
    let h = 12;
    let w = 5;
    let kh = 8;
    let kw = 1;
    let pad_h = 2;
    let pad_w = 0;
    let stride_h = 4;
    let stride_w = 1;
    let h_out = (h + 2 * pad_h - kh) / stride_h + 1; // = 3
    let w_out = (w + 2 * pad_w - kw) / stride_w + 1; // = 5
    let input: Vec<f32> = (0..b * c_in * h * w).map(|i| (i as f32) * 0.1).collect();

    // CPU reference: use ops_cpu::conv2d path indirectly (we'll just compute
    // im2col by hand for this simple case).
    let patch = c_in * kh * kw;
    let n_rows = b * h_out * w_out;
    let mut cpu_out = vec![0.0f32; n_rows * patch];
    for bi in 0..b {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row_idx = (bi * h_out + oh) * w_out + ow;
                let dst = row_idx * patch;
                for ci in 0..c_in {
                    for dkh in 0..kh {
                        for dkw in 0..kw {
                            let ih = oh * stride_h + dkh;
                            let iw = ow * stride_w + dkw;
                            let ih_s = ih as isize - pad_h as isize;
                            let iw_s = iw as isize - pad_w as isize;
                            let val = if ih_s >= 0
                                && iw_s >= 0
                                && (ih_s as usize) < h
                                && (iw_s as usize) < w
                            {
                                input[((bi * c_in + ci) * h + ih_s as usize) * w + iw_s as usize]
                            } else {
                                0.0
                            };
                            cpu_out[dst + (ci * kh + dkh) * kw + dkw] = val;
                        }
                    }
                }
            }
        }
    }

    // GPU im2col.
    let state = Arc::new(CudaState::new(0).expect("cuda init"));
    let gpu_in = state
        .upload_f32(&input, vec![b, c_in, h, w])
        .expect("up");
    let gpu_out: GpuTensor = demucs_core_native::cuda_ops::im2col_8x1_s4p2(
        &state, gpu_in, pad_h, pad_w, stride_h, stride_w,
    )
    .expect("im2col");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    assert_eq!(gpu_dl.len(), cpu_out.len());
    let diff = max_diff(&cpu_out, &gpu_dl);
    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tol = (max_val * 1e-2).max(5e-2);
    assert!(
        diff < tol,
        "im2col cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})"
    );
}

#[test]
#[ignore]
fn cuda_linear_with_bias_matches_cpu() {
    use demucs_core_native::gpu_model::{GpuBias, GpuWeight2D};
    use demucs_core_native::model::{Bias, Weight2D};
    let state = Arc::new(CudaState::new(0).expect("cuda init"));

    // Linear: y = x @ W + b, x [outer=3, in=8], W [in=8, out=4], b [out=4].
    let outer = 3;
    let in_dim = 8;
    let out_dim = 4;
    let input: Vec<f32> = (0..outer * in_dim).map(|i| (i as f32) * 0.1 - 0.5).collect();
    // CPU weight is [out, in] (PyTorch layout). GpuWeight2D::from_cpu
    // transposes internally to [in, out] for gemm_f16.
    let cpu_w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 0.07).collect();
    let cpu_b: Vec<f32> = (0..out_dim).map(|i| 0.1 * i as f32 - 0.2).collect();

    // CPU reference.
    let mut cpu_out = vec![0.0f32; outer * out_dim];
    for o in 0..outer {
        for j in 0..out_dim {
            let mut sum = cpu_b[j];
            for k in 0..in_dim {
                sum += input[o * in_dim + k] * cpu_w[j * in_dim + k];
            }
            cpu_out[o * out_dim + j] = sum;
        }
    }

    // GPU. Pass weight in original PyTorch [out, in] layout — from_cpu
    // transposes internally.
    let gpu_w = GpuWeight2D::from_cpu(
        &state,
        &Weight2D {
            data: cpu_w.clone(),
            rows: out_dim,
            cols: in_dim,
        },
    )
    .expect("upload w");
    let gpu_b = GpuBias::from_cpu(
        &state,
        &Bias {
            data: cpu_b.clone(),
            len: cpu_b.len(),
        },
    )
    .expect("upload b");
    let gpu_x = state
        .upload_f32(&input, vec![outer, in_dim])
        .expect("upload x");
    let gpu_out: GpuTensor =
        demucs_core_native::cuda_ops::linear_with_bias(&state, &gpu_x, &gpu_w, &gpu_b)
            .expect("linear");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");

    let diff = max_diff(&cpu_out, &gpu_dl);
    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("linear cpu[0..5]={:?}", &cpu_out[..5]);
    eprintln!("linear gpu[0..5]={:?}", &gpu_dl[..5]);
    let tol = (max_val * 1e-2).max(5e-2);
    assert!(
        diff < tol,
        "linear cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})"
    );
}

#[test]
#[ignore]
fn cuda_conv1d_k3_dilation_matches_cpu() {
    use demucs_core_native::model::{Bias, Conv1dWeight};
    use demucs_core_native::gpu_model::GpuConv1dWeight;
    let state = Arc::new(CudaState::new(0).expect("cuda init"));

    // Conv1d k=3, pad=dilation, dilation (1 or 2). B=1, C_in=2, L=8, C_out=3.
    for &dilation in &[1usize, 2, 4] {
        let b = 1;
        let c_in = 2;
        let l = 8;
        let c_out = 3;
        let k = 3;
        let pad = dilation;
        let input: Vec<f32> = (0..b * c_in * l).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let w_data: Vec<f32> = (0..c_out * c_in * k).map(|i| (i as f32) * 0.07 - 0.3).collect();
        let b_data: Vec<f32> = (0..c_out).map(|i| 0.1 * i as f32 - 0.15).collect();

        // CPU reference.
        let l_out = (l + 2 * pad - dilation * (k - 1) - 1) / 1 + 1;
        assert_eq!(l_out, l, "test only for same-length output");
        let mut cpu_out = vec![0.0f32; b * c_out * l_out];
        for bi in 0..b {
            for ol in 0..l_out {
                for co in 0..c_out {
                    let mut sum = b_data[co];
                    for ci in 0..c_in {
                        for dk in 0..k {
                            let il = ol as isize + dk as isize * dilation as isize
                                - pad as isize;
                            if il >= 0 && (il as usize) < l {
                                sum += input[(bi * c_in + ci) * l + il as usize]
                                    * w_data[((co * c_in + ci) * k) + dk];
                            }
                        }
                    }
                    cpu_out[(bi * c_out + co) * l_out + ol] = sum;
                }
            }
        }

        // GPU.
        let gpu_w = GpuConv1dWeight::from_cpu(
            &state,
            &Conv1dWeight {
                data: w_data.clone(),
                out_ch: c_out,
                in_ch: c_in,
                k,
            },
        )
        .expect("up w");
        let gpu_b = GpuBias::from_cpu(
            &state,
            &Bias {
                data: b_data.clone(),
                len: b_data.len(),
            },
        )
        .expect("up b");
        let gpu_x = state
            .upload_f32(&input, vec![b, c_in, l])
            .expect("up x");
        let gpu_out: GpuTensor = demucs_core_native::cuda_ops::conv1d_k3_dilation(
            &state,
            &gpu_x,
            &gpu_w,
            &gpu_b,
            dilation,
        )
        .expect("conv1d_k3_dilation");
        let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
        let diff = max_diff(&cpu_out, &gpu_dl);
        let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let tol = (max_val * 1e-2).max(5e-2);
        assert!(
            diff < tol,
            "conv1d_k3_dilation(d={dilation}) cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})"
        );
    }
}

#[test]
#[ignore]
fn cuda_conv1d_k1_matches_cpu() {
    use demucs_core_native::model::{Bias, Conv1dWeight};
    use demucs_core_native::gpu_model::GpuConv1dWeight;
    let state = Arc::new(CudaState::new(0).expect("cuda init"));

    let b = 1;
    let c_in = 2;
    let l = 6;
    let c_out = 3;
    let k = 1;
    let input: Vec<f32> = (0..b * c_in * l).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let w_data: Vec<f32> = (0..c_out * c_in * k).map(|i| (i as f32) * 0.07 - 0.3).collect();
    let b_data: Vec<f32> = (0..c_out).map(|i| 0.1 * i as f32 - 0.15).collect();

    // CPU: y[b,co,l] = sum_ci x[b,ci,l] * w[co,ci,0] + b[co]
    let mut cpu_out = vec![0.0f32; b * c_out * l];
    for bi in 0..b {
        for co in 0..c_out {
            for ol in 0..l {
                let mut sum = b_data[co];
                for ci in 0..c_in {
                    sum += input[(bi * c_in + ci) * l + ol] * w_data[co * c_in + ci];
                }
                cpu_out[(bi * c_out + co) * l + ol] = sum;
            }
        }
    }

    let gpu_w = GpuConv1dWeight::from_cpu(
        &state,
        &Conv1dWeight {
            data: w_data.clone(),
            out_ch: c_out,
            in_ch: c_in,
            k,
        },
    )
    .expect("up w");
    let gpu_b = GpuBias::from_cpu(
        &state,
        &Bias {
            data: b_data.clone(),
            len: b_data.len(),
        },
    )
    .expect("up b");
    let gpu_x = state
        .upload_f32(&input, vec![b, c_in, l])
        .expect("up x");
    let gpu_out: GpuTensor =
        demucs_core_native::cuda_ops::conv1d_k1(&state, &gpu_x, &gpu_w, &gpu_b).expect("conv1d_k1");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    let diff = max_diff(&cpu_out, &gpu_dl);
    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tol = (max_val * 1e-2).max(5e-2);
    assert!(
        diff < tol,
        "conv1d_k1 cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})"
    );
}

#[test]
#[ignore]
fn cuda_conv1d_k8_s4p2_matches_cpu() {
    use demucs_core_native::model::{Bias, Conv1dWeight};
    use demucs_core_native::gpu_model::GpuConv1dWeight;
    let state = Arc::new(CudaState::new(0).expect("cuda init"));

    let b = 1;
    let c_in = 2;
    let l = 12; // -> l_out = (12+4-8)/4 + 1 = 3
    let c_out = 3;
    let k = 8;
    let input: Vec<f32> = (0..b * c_in * l).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let w_data: Vec<f32> = (0..c_out * c_in * k).map(|i| (i as f32) * 0.07 - 0.3).collect();
    let b_data: Vec<f32> = (0..c_out).map(|i| 0.1 * i as f32 - 0.15).collect();

    // CPU reference.
    let pad = 2;
    let stride = 4;
    let l_out = (l + 2 * pad - k) / stride + 1;
    let mut cpu_out = vec![0.0f32; b * c_out * l_out];
    for bi in 0..b {
        for ol in 0..l_out {
            for co in 0..c_out {
                let mut sum = b_data[co];
                for ci in 0..c_in {
                    for dk in 0..k {
                        let il = ol as isize * stride as isize + dk as isize - pad as isize;
                        if il >= 0 && (il as usize) < l {
                            sum += input[(bi * c_in + ci) * l + il as usize]
                                * w_data[((co * c_in + ci) * k) + dk];
                        }
                    }
                }
                cpu_out[(bi * c_out + co) * l_out + ol] = sum;
            }
        }
    }

    let gpu_w = GpuConv1dWeight::from_cpu(
        &state,
        &Conv1dWeight {
            data: w_data.clone(),
            out_ch: c_out,
            in_ch: c_in,
            k,
        },
    )
    .expect("up w");
    let gpu_b = GpuBias::from_cpu(
        &state,
        &Bias {
            data: b_data.clone(),
            len: b_data.len(),
        },
    )
    .expect("up b");
    let gpu_x = state
        .upload_f32(&input, vec![b, c_in, l])
        .expect("up x");
    let gpu_out: GpuTensor =
        demucs_core_native::cuda_ops::conv1d_k8_s4p2(&state, &gpu_x, &gpu_w, &gpu_b)
            .expect("conv1d_k8_s4p2");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    let diff = max_diff(&cpu_out, &gpu_dl);
    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tol = (max_val * 1e-2).max(5e-2);
    assert!(
        diff < tol,
        "conv1d_k8_s4p2 cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})"
    );
}

#[test]
#[ignore]
fn cuda_conv2d_8x1_s4p2_matches_cpu() {
    use demucs_core_native::model::{Bias, Conv2dWeight};
    use demucs_core_native::gpu_model::GpuConv2dWeight;
    let state = Arc::new(CudaState::new(0).expect("cuda init"));

    // B=1, C_in=2, H=8, W=4; kH=8, kW=1, sH=4, sW=1, pH=2, pW=0
    // h_out = (8+4-8)/4+1 = 2, w_out = 4
    let b = 1;
    let c_in = 2;
    let h = 8;
    let w = 4;
    let kh = 8;
    let kw = 1;
    let c_out = 3;
    let pad_h = 2;
    let pad_w = 0;
    let stride_h = 4;
    let stride_w = 1;
    let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w + 2 * pad_w - kw) / stride_w + 1;
    let input: Vec<f32> = (0..b * c_in * h * w).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let w_data: Vec<f32> = (0..c_out * c_in * kh * kw).map(|i| (i as f32) * 0.07 - 0.3).collect();
    let b_data: Vec<f32> = (0..c_out).map(|i| 0.1 * i as f32 - 0.15).collect();

    let mut cpu_out = vec![0.0f32; b * c_out * h_out * w_out];
    for bi in 0..b {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = b_data[co];
                    for ci in 0..c_in {
                        for dkh in 0..kh {
                            for dkw in 0..kw {
                                let ih = oh * stride_h + dkh;
                                let iw = ow * stride_w + dkw;
                                let ih_s = ih as isize - pad_h as isize;
                                let iw_s = iw as isize - pad_w as isize;
                                if ih_s >= 0 && iw_s >= 0 && (ih_s as usize) < h && (iw_s as usize) < w {
                                    sum += input[((bi * c_in + ci) * h + ih_s as usize) * w + iw_s as usize]
                                        * w_data[(((co * c_in + ci) * kh) + dkh) * kw + dkw];
                                }
                            }
                        }
                    }
                    cpu_out[((bi * c_out + co) * h_out + oh) * w_out + ow] = sum;
                }
            }
        }
    }

    let gpu_w = GpuConv2dWeight::from_cpu(
        &state,
        &Conv2dWeight {
            data: w_data.clone(),
            out_ch: c_out,
            in_ch: c_in,
            kh,
            kw,
        },
    )
    .expect("up w");
    let gpu_b = GpuBias::from_cpu(
        &state,
        &Bias {
            data: b_data.clone(),
            len: b_data.len(),
        },
    )
    .expect("up b");
    let gpu_x = state
        .upload_f32(&input, vec![b, c_in, h, w])
        .expect("up x");
    let gpu_out: GpuTensor =
        demucs_core_native::cuda_ops::conv2d_8x1_s4p2(&state, &gpu_x, &gpu_w, &gpu_b)
            .expect("conv2d_8x1_s4p2");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    let diff = max_diff(&cpu_out, &gpu_dl);
    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("conv2d_8x1_s4p2 cpu[0..5]={:?}", &cpu_out[..5]);
    eprintln!("conv2d_8x1_s4p2 gpu[0..5]={:?}", &gpu_dl[..5]);
    let tol = (max_val * 1e-2).max(5e-2);
    assert!(
        diff < tol,
        "conv2d_8x1_s4p2 cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})"
    );
}

#[test]
#[ignore]
fn cuda_groupnorm1_matches_cpu() {
    use demucs_core_native::model::GroupNorm1;
    use demucs_core_native::gpu_model::GpuGroupNorm1;
    let state = Arc::new(CudaState::new(0).expect("cuda init"));

    let b = 2;
    let c = 4;
    let l = 8;
    let mut input: Vec<f32> = (0..b * c * l).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let gamma: Vec<f32> = (0..c).map(|i| 0.5 + 0.1 * i as f32).collect();
    let beta: Vec<f32> = (0..c).map(|i| -0.2 + 0.05 * i as f32).collect();
    // CPU reference.
    let mut cpu = input.clone();
    ops_cpu::groupnorm1(&mut cpu, [b, c, l], &GroupNorm1 { gamma: gamma.clone(), beta: beta.clone(), num_channels: c });
    // GPU.
    let mut gpu = state.upload_f32(&input, vec![b, c, l]).expect("up");
    let gn = GpuGroupNorm1::from_cpu(&state, &GroupNorm1 { gamma, beta, num_channels: c }).expect("up gn");
    demucs_core_native::cuda_ops::groupnorm1_inplace(&state, &mut gpu, &gn).expect("gn");
    let gpu_dl = state.download_to_f32(&gpu).expect("dl");
    let diff = max_diff(&cpu, &gpu_dl);
    let max_val = cpu.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tol = (max_val * 1e-2).max(5e-2);
    assert!(
        diff < tol,
        "groupnorm1 cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})"
    );
    // input was not consumed by the call (still valid).
    let _ = &input;
}

#[test]
#[ignore]
fn cuda_glu_channel_matches_cpu() {
    let state = Arc::new(CudaState::new(0).expect("cuda init"));

    let b = 1;
    let c2 = 4;
    let l = 6;
    let input: Vec<f32> = (0..b * c2 * l).map(|i| (i as f32) * 0.2 - 0.5).collect();
    // CPU reference.
    let (cpu, _) = ops_cpu::glu_channel(&input, [b, c2, l]);

    // GPU.
    let gpu = state.upload_f32(&input, vec![b, c2, l]).expect("up");
    let gpu_out: GpuTensor = demucs_core_native::cuda_ops::glu_channel(&state, &gpu).expect("glu");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    let diff = max_diff(&cpu, &gpu_dl);
    let max_val = cpu.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tol = (max_val * 1e-2).max(5e-2);
    assert!(
        diff < tol,
        "glu_channel cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})"
    );
}

#[test]
#[ignore]
fn cuda_denorm_freq_matches_cpu() {
    let state = Arc::new(CudaState::new(0).expect("cuda init"));

    let b = 2;
    let c = 3;
    let h = 4;
    let w = 5;
    let n = c * h * w;
    let mut input: Vec<f32> = (0..b * n).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let mean: Vec<f32> = (0..b).map(|i| 0.1 * i as f32 - 0.05).collect();
    let std: Vec<f32> = (0..b).map(|i| 0.5 + 0.1 * i as f32).collect();

    // CPU reference.
    let mut cpu = input.clone();
    ops_cpu::denormalize_freq(&mut cpu, [b, c, h, w], &mean, &std);

    // GPU.
    let mut gpu = state.upload_f32(&input, vec![b, c, h, w]).expect("up");
    let gpu_mean = state.upload_f32(&mean, vec![b]).expect("up mean");
    let gpu_std = state.upload_f32(&std, vec![b]).expect("up std");
    demucs_core_native::cuda_ops::denorm_freq_inplace(&state, &mut gpu, &gpu_mean, &gpu_std).expect("denorm");
    let gpu_dl = state.download_to_f32(&gpu).expect("dl");
    let diff = max_diff(&cpu, &gpu_dl);
    let max_val = cpu.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tol = (max_val * 1e-2).max(5e-2);
    assert!(
        diff < tol,
        "denorm_freq cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})"
    );
}
#[test]
#[ignore]
fn cuda_conv_transpose2d_8x1_s4p2_matches_cpu() {
    use demucs_core_native::model::{Bias, Conv2dWeight};
    use demucs_core_native::gpu_model::GpuConvTranspose2dWeight;
    let state = Arc::new(CudaState::new(0).expect("cuda init"));

    // B=1, C_in=2, H=2, W=4; kH=8, kW=1, sH=4, sW=1, pH=2, pW=0
    // h_out = (2-1)*4 + 7 - 4 + 1 = 8, w_out = 4
    let b = 1;
    let c_in = 2;
    let h_in = 2;
    let w_in = 4;
    let kh = 8;
    let kw = 1;
    let c_out = 3;
    let pad_h = 2;
    let pad_w = 0;
    let stride_h = 4;
    let stride_w = 1;
    let h_out = (h_in - 1) * stride_h + (kh - 1) - 2 * pad_h + 1;
    let w_out = (w_in - 1) * stride_w + (kw - 1) - 2 * pad_w + 1;
    let input: Vec<f32> = (0..b * c_in * h_in * w_in).map(|i| (i as f32) * 0.1 - 0.5).collect();
    // PyTorch ConvTranspose2d weight layout [c_in, c_out, kH, kW].
    let mut w_pt: Vec<f32> = Vec::with_capacity(c_in * c_out * kh * kw);
    for ic in 0..c_in {
        for oc in 0..c_out {
            for dkh in 0..kh {
                for dkw in 0..kw {
                    let v = ((ic * c_out + oc) * kh * kw + dkh * kw + dkw) as f32 * 0.07 - 0.3;
                    w_pt.push(v);
                }
            }
        }
    }
    let b_data: Vec<f32> = (0..c_out).map(|i| 0.1 * i as f32 - 0.15).collect();

    // Load-time reorder: patch=ic*kh*kw + dkh*kw + dkw; reordered[patch, oc] = a[ic, oc, dkh, dkw]
    let patch = c_in * kh * kw;
    let mut w_reordered = vec![0.0f32; patch * c_out];
    for ic in 0..c_in {
        for oc in 0..c_out {
            for dkh in 0..kh {
                for dkw in 0..kw {
                    let src = ((ic * c_out + oc) * kh + dkh) * kw + dkw;
                    let p = ic * kh * kw + dkh * kw + dkw;
                    w_reordered[p * c_out + oc] = w_pt[src];
                }
            }
        }
    }

    // CPU reference (matches ops_cpu::conv_transpose2d).
    let mut cpu_out = vec![0.0f32; b * c_out * h_out * w_out];
    for bi in 0..b {
        for oh in 0..h_out {
            for ow in 0..w_out {
                for co in 0..c_out {
                    let mut sum = b_data[co];
                    for ci in 0..c_in {
                        for dkh in 0..kh {
                            for dkw in 0..kw {
                                let oh_p = oh as isize + pad_h as isize - dkh as isize;
                                let ow_p = ow as isize + pad_w as isize - dkw as isize;
                                if oh_p < 0
                                    || ow_p < 0
                                    || oh_p % stride_h as isize != 0
                                    || ow_p % stride_w as isize != 0
                                {
                                    continue;
                                }
                                let ih = (oh_p / stride_h as isize) as usize;
                                let iw = (ow_p / stride_w as isize) as usize;
                                if ih < h_in && iw < w_in {
                                    let x_idx = ((bi * c_in + ci) * h_in + ih) * w_in + iw;
                                    // w[ic, oc, dkh, dkw] in original PyTorch layout.
                                    let w_idx = ((ci * c_out + co) * kh + dkh) * kw + dkw;
                                    sum += input[x_idx] * w_pt[w_idx];
                                }
                            }
                        }
                    }
                    cpu_out[((bi * c_out + co) * h_out + oh) * w_out + ow] = sum;
                }
            }
        }
    }

    // Build the CPU Conv2dWeight (using the reordered data layout).
    let w_cpu = Conv2dWeight {
        data: w_reordered.clone(),
        out_ch: c_out,
        in_ch: c_in,
        kh,
        kw,
    };

    let gpu_w = GpuConvTranspose2dWeight::from_cpu(&state, &w_cpu).expect("up w");
    let gpu_b = GpuBias::from_cpu(
        &state,
        &Bias {
            data: b_data.clone(),
            len: b_data.len(),
        },
    )
    .expect("up b");
    let gpu_x = state
        .upload_f32(&input, vec![b, c_in, h_in, w_in])
        .expect("up x");
    let gpu_out: GpuTensor = demucs_core_native::cuda_ops::conv_transpose2d_8x1_s4p2(
        &state,
        &gpu_x,
        &gpu_w,
        &gpu_b,
    )
    .expect("conv_transpose2d_8x1_s4p2");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");
    let diff = max_diff(&cpu_out, &gpu_dl);
    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("conv_t2d cpu[0..5]={:?}", &cpu_out[..5]);
    eprintln!("conv_t2d gpu[0..5]={:?}", &gpu_dl[..5]);
    let tol = (max_val * 1e-2).max(5e-2);
    assert!(
        diff < tol,
        "conv_transpose2d_8x1_s4p2 cpu vs gpu: max_diff={diff:.6} exceeds tol={tol:.4} (max_val={max_val:.2})"
    );
}
