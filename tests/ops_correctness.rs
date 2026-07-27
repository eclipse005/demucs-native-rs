//! Numerical correctness tests for CPU operators (conv2d, conv1d, gelu, etc).
//!
//! These use small hand-crafted inputs with known outputs to verify the
//! im2col + GEMM implementation matches the mathematical definition exactly.

use demucs_core_native::model::{Bias, Conv1dWeight, Conv2dWeight, LayerNorm1, MhaWeights, Weight2D};
use demucs_core_native::ops_cpu;

#[test]
fn conv2d_identity_kernel() {
    // 1x1 conv with identity weight: output should equal input.
    // weight [1, 1, 1, 1] = [1.0], bias [1] = 0.0
    let w = Conv2dWeight {
        data: vec![1.0],
        out_ch: 1,
        in_ch: 1,
        kh: 1,
        kw: 1,
    };
    let bias = Bias {
        data: vec![0.0],
        len: 1,
    };
    // input [1, 1, 3, 3] = [[1,2,3],[4,5,6],[7,8,9]]
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let (out, shape) = ops_cpu::conv2d(&x, [1, 1, 3, 3], &w, &bias, 0, 0, 1, 1);
    assert_eq!(shape, [1, 1, 3, 3]);
    assert_eq!(out, x, "1x1 identity conv should pass input through");
}

#[test]
fn conv2d_with_bias() {
    // 1x1 conv with weight=2, bias=3: out = 2*in + 3
    let w = Conv2dWeight {
        data: vec![2.0],
        out_ch: 1,
        in_ch: 1,
        kh: 1,
        kw: 1,
    };
    let bias = Bias {
        data: vec![3.0],
        len: 1,
    };
    let x = vec![1.0, 2.0, 3.0, 4.0]; // [1,1,2,2]
    let (out, _) = ops_cpu::conv2d(&x, [1, 1, 2, 2], &w, &bias, 0, 0, 1, 1);
    assert_eq!(out, vec![5.0, 7.0, 9.0, 11.0]); // 2*[1,2,3,4]+3
}

#[test]
fn conv2d_3x3_stride1_pad1_preserves_size() {
    // 3x3 conv, stride 1, pad 1 → output H/W same as input.
    // 2 input channels, 1 output channel, kernel = all ones.
    let out_ch = 1;
    let in_ch = 2;
    let kh = 3;
    let kw = 3;
    // weight [1, 2, 3, 3] = all 1.0 (sum of 2*9=18 input values per output)
    let w = Conv2dWeight {
        data: vec![1.0; out_ch * in_ch * kh * kw],
        out_ch,
        in_ch,
        kh,
        kw,
    };
    let bias = Bias {
        data: vec![0.0],
        len: 1,
    };
    // input [1, 2, 3, 3]: channel 0 = all 1s, channel 1 = all 1s
    let x = vec![1.0; 18]; // 1*2*3*3
    let (out, shape) = ops_cpu::conv2d(&x, [1, 2, 3, 3], &w, &bias, 1, 1, 1, 1);
    assert_eq!(shape, [1, 1, 3, 3]);
    // Center output: full 3x3 receptive field over both channels = 18 * 1.0 = 18.0
    // Corner outputs: 4 of the 9 taps hit padding (0), so 2*4*1.0 (4 valid per channel) = ...
    // Actually with pad=1, a 3x3 kernel on a 3x3 input: center sees all 9 per channel.
    // Corner (0,0): taps at (-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)
    //   padded positions are 0. Valid positions: (0,0),(0,1),(1,0),(1,1) = 4 per channel = 8 total.
    assert!(
        (out[4] - 18.0).abs() < 1e-5,
        "center output should be 18.0, got {}",
        out[4]
    ); // out[0,0,1,1] = center
       // Corner out[0,0,0,0]: 8.0
    assert!(
        (out[0] - 8.0).abs() < 1e-5,
        "corner output should be 8.0, got {}",
        out[0]
    );
}

#[test]
fn conv2d_stride4_downsamples() {
    // This is the HEncLayer pattern: kernel [8,1], stride [4,1], pad [2,0]
    // on a [1, 4, Fr, T] input.
    let out_ch = 1;
    let in_ch = 1;
    let kh = 8;
    let kw = 1;
    let w = Conv2dWeight {
        data: vec![1.0; out_ch * in_ch * kh * kw],
        out_ch,
        in_ch,
        kh,
        kw,
    };
    let bias = Bias {
        data: vec![0.0],
        len: 1,
    };
    // input [1, 1, 16, 1]: values 1..16
    let x: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let (out, shape) = ops_cpu::conv2d(&x, [1, 1, 16, 1], &w, &bias, 2, 0, 4, 1);
    // H_out = (16 + 2*2 - 8) / 4 + 1 = 12/4 + 1 = 4
    assert_eq!(shape, [1, 1, 4, 1]);
    // Output at oh=0: taps at positions [-2..5] (with pad), valid = 0..5
    //   = 1+2+3+4+5+6 = 21 (positions 0-5 in input, 6 values, pad contributes 0 for -2,-1)
    // Wait: kernel positions: ih = oh*4 + dkh for dkh in 0..8
    //   oh=0: ih = 0,1,2,3,4,5,6,7; ih_s = ih-2 = -2,-1,0,1,2,3,4,5
    //   valid: 0,1,2,3,4,5 → x[0..5] = 1,2,3,4,5,6 → sum = 21
    assert!(
        (out[0] - 21.0).abs() < 1e-5,
        "out[0] should be 21.0, got {}",
        out[0]
    );
}

#[test]
fn conv1d_dilation_matches_padded_conv() {
    // conv1d with dilation=2, kernel=3, pad=2 should equal conv1d with
    // effective receptive field [0, 2, 4] (for dilation 2).
    let out_ch = 1;
    let in_ch = 1;
    let k = 3;
    let w = Conv1dWeight {
        data: vec![1.0, 2.0, 3.0], // [1,1,3]
        out_ch,
        in_ch,
        k,
    };
    let bias = Bias {
        data: vec![0.0],
        len: 1,
    };
    // input [1, 1, 5] = [10, 20, 30, 40, 50]
    let x = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let (out, shape) = ops_cpu::conv1d(&x, [1, 1, 5], &w, &bias, 2, 2);
    // L_out = (5 + 2*2 - 2*(3-1) - 1) + 1 = (9 - 5) + 1 = 5
    assert_eq!(shape, [1, 1, 5]);
    // ol=0: taps at il = 0+0*2-2, 0+1*2-2, 0+2*2-2 = -2, 0, 2
    //   valid: 0 (x=10), 2 (x=30) → 1*10 + 2*0(pad) + 3*30 = 10 + 90 = 100
    // Wait, the kernel taps: dk=0 → il=-2 (pad, 0), dk=1 → il=0 (x=10), dk=2 → il=2 (x=30)
    // weight = [1, 2, 3], so out = w[0]*0 + w[1]*10 + w[2]*30 = 0 + 20 + 90 = 110
    assert!(
        (out[0] - 110.0).abs() < 1e-4,
        "out[0] should be 110.0, got {}",
        out[0]
    );
}

#[test]
fn gelu_matches_erf_formula() {
    // GELU(0) = 0, GELU(1) ≈ 0.8412, GELU(-1) ≈ -0.1589
    let mut x = vec![0.0, 1.0, -1.0, 2.0];
    ops_cpu::gelu(&mut x);
    assert!(x[0].abs() < 1e-6, "GELU(0) should be 0, got {}", x[0]);
    assert!(
        (x[1] - 0.8412).abs() < 1e-3,
        "GELU(1) should be ~0.8412, got {}",
        x[1]
    );
    assert!(
        (x[2] - (-0.1589)).abs() < 1e-3,
        "GELU(-1) should be ~-0.1589, got {}",
        x[2]
    );
    assert!(
        (x[3] - 1.9545).abs() < 1e-3,
        "GELU(2) should be ~1.9545, got {}",
        x[3]
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  Transformer building-block unit tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn layernorm_preserves_shape_and_renormalises_last_dim() {
    // Input [1, 2, 4] = a single 2×4 matrix.
    // Row [1, 2, 3, 4]: mean=2.5, var=1.25, std≈1.118, normalized ≈ [-1.34,-0.45,0.45,1.34]
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let ln = LayerNorm1 {
        gamma: vec![1.0; 4],
        beta: vec![0.0; 4],
        dim: 4,
    };
    ops_cpu::layernorm(&mut x, [1, 2, 4], &ln);
    // Shape unchanged.
    assert_eq!(x.len(), 8);
    // Each row should have mean ≈ 0 and unit variance after gamma=1, beta=0.
    let mean0: f32 = x[0..4].iter().sum::<f32>() / 4.0;
    let mean1: f32 = x[4..8].iter().sum::<f32>() / 4.0;
    assert!(mean0.abs() < 1e-4, "row 0 mean should be ~0, got {mean0}");
    assert!(mean1.abs() < 1e-4, "row 1 mean should be ~0, got {mean1}");
    // First element of row 0: (1-2.5) * 1/sqrt(1.25+1e-5) ≈ -1.3416
    assert!(
        (x[0] - (-1.3416)).abs() < 1e-3,
        "x[0] should be ~-1.3416, got {}",
        x[0]
    );
}

#[test]
fn linear_identity_returns_input() {
    // y = x @ I^T + 0 = x (when in_features == out_features and weight = I)
    let w = Weight2D {
        // 3x3 identity, row-major.
        data: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        rows: 3,
        cols: 3,
    };
    let bias = vec![0.0, 0.0, 0.0];
    let x = vec![1.0, 2.0, 3.0];
    let (out, shape) = ops_cpu::linear(&x, [1, 1, 3], &w, &bias);
    assert_eq!(shape, [1, 1, 3]);
    assert_eq!(out, vec![1.0, 2.0, 3.0]);
}

#[test]
fn linear_with_bias_adds_per_output() {
    // y = x @ W^T + b
    // x = [1, 1] (1×2 input), W = [[1, 2], [3, 4]] (2×2), bias = [10, 20]
    // y = [1*1 + 1*2, 1*3 + 1*4] + [10, 20] = [13, 27]
    let w = Weight2D {
        data: vec![1.0, 2.0, 3.0, 4.0],
        rows: 2,
        cols: 2,
    };
    let bias = vec![10.0, 20.0];
    let x = vec![1.0, 1.0];
    let (out, shape) = ops_cpu::linear(&x, [1, 1, 2], &w, &bias);
    assert_eq!(shape, [1, 1, 2]);
    assert_eq!(out, vec![13.0, 27.0]);
}

#[test]
fn softmax_sums_to_one_per_position() {
    // [1, 3, 2] — 3 sequence positions, 2 features.
    // Softmax over axis=1 (sequence dim), so for each (b, d) we softmax 3 values.
    // Layout: x[bi*sd + si*d + di].
    //   (b=0, d=0): positions at indices 0, 2, 4 — softmax(1, 2, 3)
    //   (b=0, d=1): positions at indices 1, 3, 5 — softmax(10, 20, 30)
    let mut x = vec![
        1.0, 10.0, // s=0: (d=0)=1, (d=1)=10
        2.0, 20.0, // s=1: (d=0)=2, (d=1)=20
        3.0, 30.0, // s=2: (d=0)=3, (d=1)=30
    ];
    ops_cpu::softmax(&mut x, [1, 3, 2], 1);
    // d=0 softmax: x[0] + x[2] + x[4]
    let sum0: f32 = x[0] + x[2] + x[4];
    // d=1 softmax: x[1] + x[3] + x[5]
    let sum1: f32 = x[1] + x[3] + x[5];
    assert!((sum0 - 1.0).abs() < 1e-5, "d=0 should sum to 1, got {sum0}");
    assert!((sum1 - 1.0).abs() < 1e-5, "d=1 should sum to 1, got {sum1}");
}

#[test]
fn mha_self_known_shape_and_nonzero_output() {
    // 1 batch, 4 seq, d_model=4, n_heads=2, d_head=2.
    // Random-ish inputs with all weights=0.5, bias=0.
    let d = 4usize;
    let h = 2usize;
    let x: Vec<f32> = (0..d * 4).map(|i| (i as f32 * 0.37 + 0.1).sin()).collect();
    let attn = MhaWeights {
        in_proj_weight: vec![0.5; 3 * d * d],
        in_proj_bias: vec![0.0; 3 * d],
        out_proj_weight: vec![0.5; d * d],
        out_proj_bias: vec![0.0; d],
        d_model: d,
        n_heads: h,
    };
    let (out, shape) = ops_cpu::mha_self(&x, [1, 4, d], &attn);
    assert_eq!(shape, [1, 4, d], "mha output shape");
    // All entries should be non-zero for random non-degenerate input.
    let non_zero = out.iter().filter(|&&v| v.abs() > 1e-6).count();
    assert!(
        non_zero > out.len() / 2,
        "most entries should be non-zero, got {non_zero}/{}",
        out.len()
    );
}

#[test]
fn sin_embed_1d_pos0_is_half_cos_one_half_sin_zero() {
    // For pos=0, every angle is 0 → cos(0)=1, sin(0)=0.
    // The layout is [cos*half, sin*half] along the last dim.
    let data = ops_cpu::sin_embed_1d(2, 8);
    assert_eq!(data.len(), 16);
    // pos=0: first half all 1.0, second half all 0.0.
    for i in 0..4 {
        assert!((data[i] - 1.0).abs() < 1e-6, "pos=0 cos[{i}] should be 1, got {}", data[i]);
    }
    for i in 4..8 {
        assert!(data[i].abs() < 1e-6, "pos=0 sin[{i}] should be 0, got {}", data[i]);
    }
}

#[test]
fn sin_embed_2d_shape_time_major_and_bounded() {
    // d_model=8, height=2, width=3 → seq_len=6, output [6, 8].
    let data = ops_cpu::sin_embed_2d(8, 2, 3);
    assert_eq!(data.len(), 6 * 8);
    // All values should be in [-1, 1] (sin/cos).
    for &v in &data {
        assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6, "value {v} out of [-1, 1]");
    }
    // Time-major: index s = t * height + fr.
    // pos s=0 → (t=0, fr=0): all angles = 0 → first quarter sin=0, cos=1;
    // second quarter sin=0, cos=1.
    // Channel layout: even sin, odd cos, repeated.
    // For d=8, half=4, quarter=2. Channels 0,2 are sin(time*div), 1,3 are cos, 4,6 sin(freq*div), 5,7 cos.
    // At (t=0, fr=0) all angles=0, so all sin=0, all cos=1.
    let s0 = &data[0..8];
    // Channels 0, 2, 4, 6 (sin slots) should be 0.
    for &ch in &[0usize, 2, 4, 6] {
        assert!(s0[ch].abs() < 1e-6, "s0 sin[{ch}] should be 0, got {}", s0[ch]);
    }
    // Channels 1, 3, 5, 7 (cos slots) should be 1.
    for &ch in &[1usize, 3, 5, 7] {
        assert!((s0[ch] - 1.0).abs() < 1e-6, "s0 cos[{ch}] should be 1, got {}", s0[ch]);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  ConvTranspose tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn conv_transpose2d_stride1_kernel2_upsamples_x2() {
    // H_in=2, kH=2, stride=1, pad=0: H_out = (2-1)*1 + 2 = 3.
    // weight all 1.0, input x = [1.0, 0.0]:
    //   reverse index: il * stride + dk - pad = ol → il + dk = ol.
    //   ol=0: il+dk=0 → (il=0,dk=0) → 1*1 = 1
    //   ol=1: il+dk=1 → (il=0,dk=1) + (il=1,dk=0) → 1+0 = 1
    //   ol=2: il+dk=2 → (il=1,dk=1) + (il=0,dk=2 oob) → 0
    // Output = [1, 1, 0].
    let w = demucs_core_native::model::Conv2dWeight {
        data: vec![1.0; 1 * 1 * 2 * 1], // [C_in=1, C_out=1, kH=2, kW=1]
        out_ch: 1,
        in_ch: 1,
        kh: 2,
        kw: 1,
    };
    let bias = Bias { data: vec![0.0], len: 1 };
    let x = vec![1.0, 0.0];
    let (out, shape) = ops_cpu::conv_transpose2d(&x, [1, 1, 2, 1], &w, &bias, 0, 0, 1, 1);
    assert_eq!(shape, [1, 1, 3, 1]);
    assert!((out[0] - 1.0).abs() < 1e-5, "out[0] should be 1.0, got {}", out[0]);
    assert!((out[1] - 1.0).abs() < 1e-5, "out[1] should be 1.0, got {}", out[1]);
    assert!(out[2].abs() < 1e-5, "out[2] should be 0, got {}", out[2]);
}

#[test]
fn conv_transpose2d_htdemucs_pattern_upsamples_x4() {
    // H_in=2, kH=8, stride=4, pad=2: H_out = (2-1)*4 + 8 - 4 = 8
    // Mirror of HEncLayer's [8,1] stride=[4,1] pad=[2,0].
    let w = demucs_core_native::model::Conv2dWeight {
        data: vec![1.0; 1 * 1 * 8 * 1], // [C_in=1, C_out=1, kH=8, kW=1]
        out_ch: 1,
        in_ch: 1,
        kh: 8,
        kw: 1,
    };
    let bias = Bias { data: vec![0.0], len: 1 };
    // X = [1.0, 2.0] at H_in=2.
    let x = vec![1.0, 2.0];
    let (out, shape) = ops_cpu::conv_transpose2d(&x, [1, 1, 2, 1], &w, &bias, 2, 0, 4, 1);
    assert_eq!(shape, [1, 1, 8, 1]);
    // ConvTranspose reverse index: ih * stride + dkh - pad = oh.
    // For stride>1, (ih*stride + dkh - pad) % stride == 0 is required
    // (dkh only takes values where this holds), so each ol can receive
    // multiple contributing taps from different ih.
    //   oh=0: ih*4+dkh=2 → (ih=0,dkh=2) → 1
    //   oh=1: ih*4+dkh=3 → (ih=0,dkh=3) → 1
    //   oh=2: ih*4+dkh=4 → (ih=0,dkh=4)+(ih=1,dkh=0) → 1+2=3
    //   oh=3: ih*4+dkh=5 → (ih=0,dkh=5)+(ih=1,dkh=1) → 1+2=3
    //   oh=4: ih*4+dkh=6 → (ih=0,dkh=6)+(ih=1,dkh=2) → 1+2=3
    //   oh=5: ih*4+dkh=7 → (ih=0,dkh=7)+(ih=1,dkh=3) → 1+2=3
    //   oh=6: ih*4+dkh=8 → (ih=1,dkh=4) → 2
    //   oh=7: ih*4+dkh=9 → (ih=1,dkh=5) → 2
    let expected = [1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0];
    for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-5, "out[{i}] should be {exp}, got {got}");
    }
}

#[test]
fn conv_transpose2d_multi_out_channel_validates_gemm_strides() {
    // Multi-output-channel test. With c_out=1 the GEMM rhs stride pattern is
    // unobservable (row- and column-major interpretations coincide), so this
    // test is the only one that can catch a wrong rhs stride. Here c_in=1,
    // c_out=2, kH=2, kW=1, stride=1, pad=0.
    //
    // PyTorch ConvTranspose2d weight layout: [in=1, out=2, kH=2, kW=1] row-major.
    // Memory = [a[0,0,0,0], a[0,0,1,0], a[0,1,0,0], a[0,1,1,0]] =
    //         [1, 2, 3, 4].
    // Reshape to [patch=in*kh*kw=2, c_out=2] row-major:
    //   W'[0, 0] = memory[0] = a[0,0,0,0] = 1
    //   W'[1, 0] = memory[1] = a[0,0,1,0] = 2
    //   W'[0, 1] = memory[2] = a[0,1,0,0] = 3
    //   W'[1, 1] = memory[3] = a[0,1,1,0] = 4
    // So W' = [[1, 3], [2, 4]].
    //
    // input x = [1.0] (shape [1, 1, 1, 1]). With L_in=1, k=2, stride=1, pad=0:
    //   L_out = (1-1)*1 + (2-1) - 0 + 1 = 2.
    // im2col: for each (oh, ow) of size [1, 1, 2, 1] (1 batch, 1 in_ch,
    // 2 output positions, 2 patches), with ci=0, dkh ∈ {0, 1}, dkw=0:
    //   oh=0: dkh=0 → oh_p=0, ih=0/1=0 → x[0]=1.
    //          dkh=1 → oh_p=-1 (skip).
    //     col row 0 = [1, 0]
    //   oh=1: dkh=0 → oh_p=1, ih=1/1=1 (oob) → 0.
    //          dkh=1 → oh_p=0, ih=0/1=0 → x[0]=1.
    //     col row 1 = [0, 1]
    //
    // out[2, 2] = col[2, 2] @ W'[2, 2]:
    //   out[0, 0] = 1*1 + 0*3 = 1
    //   out[0, 1] = 1*2 + 0*4 = 2
    //   out[1, 0] = 0*1 + 1*3 = 3
    //   out[1, 1] = 0*2 + 1*4 = 4
    let w = demucs_core_native::model::Conv2dWeight {
        // PyTorch [in=1, out=2, kH=2, kW=1] row-major flat: 1, 2, 3, 4.
        data: vec![1.0, 2.0, 3.0, 4.0],
        out_ch: 2,
        in_ch: 1,
        kh: 2,
        kw: 1,
    };
    let bias = Bias {
        data: vec![0.0, 0.0],
        len: 2,
    };
    let x = vec![1.0];
    let (out, shape) =
        ops_cpu::conv_transpose2d(&x, [1, 1, 1, 1], &w, &bias, 0, 0, 1, 1);
    assert_eq!(shape, [1, 2, 2, 1]);
    // `out` is reshaped to [B, C_out, H_out, W_out]. Flat memory order:
    // [c=0,oh=0,ow=0; c=0,oh=1,ow=0; c=1,oh=0,ow=0; c=1,oh=1,ow=0]
    //   = [out[0,0], out[1,0], out[0,1], out[1,1]]
    //   = [1, 3, 2, 4]
    let expected = [1.0, 3.0, 2.0, 4.0];
    for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "out[{i}] should be {exp}, got {got}"
        );
    }
}

#[test]
fn conv_transpose1d_stride1_kernel2_upsamples() {
    // L_in=1, k=2, stride=1, pad=0: L_out = (1-1)*1 + 2 = 2
    // weight all 1.0, input [1.0]: reverse index il*1+dk-0=ol.
    //   ol=0: il+dk=0 → (il=0, dk=0) → x[0] = 1
    //   ol=1: il+dk=1 → (il=0, dk=1) → x[0] = 1
    // So output = [1, 1].
    let w = demucs_core_native::model::Conv1dWeight {
        data: vec![1.0; 1 * 1 * 2],
        out_ch: 1,
        in_ch: 1,
        k: 2,
    };
    let bias = Bias { data: vec![0.0], len: 1 };
    let x = vec![1.0];
    let (out, shape) = ops_cpu::conv_transpose1d(&x, [1, 1, 1], &w, &bias, 0, 1);
    assert_eq!(shape, [1, 1, 2]);
    assert!((out[0] - 1.0).abs() < 1e-5, "out[0] should be 1.0, got {}", out[0]);
    assert!((out[1] - 1.0).abs() < 1e-5, "out[1] should be 1.0, got {}", out[1]);
}

#[test]
fn conv_transpose1d_multi_out_channel_validates_gemm_strides() {
    // Same shape as conv_transpose2d_multi_out_channel but 1D. PyTorch
    // ConvTranspose1d weight [in=1, out=2, k=2] row-major flat: 1, 2, 3, 4.
    // After take_conv_transpose1d's reorder to [patch, c_out] row-major
    // (`reordered[i, oc]` with i = ic*k + dk): for in=1, k=2, c_out=2,
    //   reordered[0, 0] = a[0, 0, 0] = 1, reordered[1, 0] = a[0, 0, 1] = 2
    //   reordered[0, 1] = a[0, 1, 0] = 3, reordered[1, 1] = a[0, 1, 1] = 4
    // so reordered flat = [1, 3, 2, 4].
    let w = demucs_core_native::model::Conv1dWeight {
        data: vec![1.0, 3.0, 2.0, 4.0],
        out_ch: 2,
        in_ch: 1,
        k: 2,
    };
    let bias = Bias { data: vec![0.0, 0.0], len: 2 };
    let x = vec![1.0f32];
    let (out, shape) = ops_cpu::conv_transpose1d(&x, [1, 1, 1], &w, &bias, 0, 1);
    assert_eq!(shape, [1, 2, 2]);
    // L_in=1, k=2, stride=1, pad=0: L_out = (1-1)*1 + 2 - 0 = 2.
    // im2col (with input layout [b=0, c_in=0, l=0]=1):
    //   ol=0: dk=0 → ol_p=0, il=0/1=0 → x[0]=1. dk=1 → ol_p=-1 (skip).
    //     col row 0 = [1, 0]
    //   ol=1: dk=0 → ol_p=1, il=1/1=1 (oob) → 0. dk=1 → ol_p=0 → x[0]=1.
    //     col row 1 = [0, 1]
    // GEMM with W' = [[1, 3], [2, 4]] (row-major, from reordered flat):
    //   out[0, 0] = 1*1 + 0*2 = 1
    //   out[0, 1] = 1*3 + 0*4 = 3
    //   out[1, 0] = 0*1 + 1*2 = 2
    //   out[1, 1] = 0*3 + 1*4 = 4
    // Reshaped to [B, C, L] = [1, 2, 2] flat order:
    //   [c=0,l=0; c=0,l=1; c=1,l=0; c=1,l=1] = [out[0,0], out[1,0], out[0,1], out[1,1]]
    //   = [1, 2, 3, 4]
    let expected = [1.0, 2.0, 3.0, 4.0];
    for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "out[{i}] should be {exp}, got {got}"
        );
    }
}

#[test]
fn normalize_denormalize_freq_roundtrip() {
    // Random-ish 4D input: 1 batch, 4 channels, 8 freq bins, 4 time frames.
    let shape = [1, 4, 8, 4];
    let mut x: Vec<f32> = (0..shape[1] * shape[2] * shape[3])
        .map(|i| (i as f32 * 0.37 - 0.5).sin() * 2.0 + 1.5)
        .collect();
    let x_orig = x.clone();
    let (out, out_shape, mean, mean_shape, std, std_shape) =
        ops_cpu::normalize_freq(&x, shape);
    assert_eq!(out_shape, shape);
    assert_eq!(mean_shape, [1, 1, 1, 1]);
    assert_eq!(std_shape, [1, 1, 1, 1]);
    // Denormalize: should approximately recover x.
    let mut recovered = out;
    ops_cpu::denormalize_freq(&mut recovered, shape, &mean, &std);
    assert_eq!(recovered.len(), x_orig.len());
    let max_diff: f32 = recovered
        .iter()
        .zip(x_orig.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "normalize/denormalize roundtrip error too large: {max_diff}"
    );

    // Verify normalized output has zero mean and unit variance.
    x = x_orig; // keep x for the variance check below
    let (xn, _, _, _, _, _) = ops_cpu::normalize_freq(&x, shape);
    let m: f32 = xn.iter().sum::<f32>() / xn.len() as f32;
    assert!(m.abs() < 1e-5, "normalized mean should be ~0, got {m}");
}
