// GPU kernels for HTDemucs v4 hand-written CUDA engine.
// All arithmetic accumulates in f32 but storage is f16.
// Targets sm_61+ (no requirement for tensor cores or f16 atomics).
//
// Status: foundational kernels. Each kernel has a single (or paired)
// Rust-side launcher in cuda_ops.rs. Vectorized __half2 where possible
// to halve memory traffic.

#include <cuda_fp16.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// Placeholder kernel so NVRTC has something to compile. Removed once real
// kernels are present.
extern "C" __global__ void __launch_bounds__(256)
noop_placeholder(__half* __restrict__ x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // leave x unchanged
    (void)x[idx];
}

// ═══════════════════════════════════════════════════════════════════════
//  HTDemucs v4 element-wise kernels
// ═══════════════════════════════════════════════════════════════════════

// ─── GroupNorm1 (1 group over C channels, per-batch, on [B, C, L]) ───────
// Matches MyGroupNorm(1, C) in demucs: mean/var over (C, L) per batch.
// x [B, C, L] row-major, gamma/beta [C]. eps = 1e-5.
// One block per batch element. Block size = next pow2 >= min(C*L, 1024).
extern "C" __global__ void __launch_bounds__(1024, 4)
groupnorm1_f16(
    __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    int b, int c, int l,
    float eps
) {
    int bi = blockIdx.x;
    if (bi >= b) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int cl = c * l;

    extern __shared__ float sdata[];

    // Sum and sum of squares
    float l_sum = 0.0f, l_sq = 0.0f;
    for (int j = tid; j < cl; j += bs) {
        float v = __half2float(x[bi * cl + j]);
        l_sum += v;
        l_sq  += v * v;
    }
    sdata[tid] = l_sum;
    sdata[tid + bs] = l_sq;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid]     += sdata[tid + s];
            sdata[tid + bs] += sdata[tid + bs + s];
        }
        __syncthreads();
    }
    float mean = sdata[0] / (float)cl;
    float var  = sdata[bs] / (float)cl - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Normalize + per-channel scale/shift. gamma/beta indexed by ci = j / l.
    for (int j = tid; j < cl; j += bs) {
        int ci = j / l;
        float v = (__half2float(x[bi * cl + j]) - mean) * inv_std;
        x[bi * cl + j] = __float2half(v * __half2float(gamma[ci]) + __half2float(beta[ci]));
    }
}

// ─── GLU along channel dim: out = a * sigmoid(b) for [B, 2C, L] ───────
// a, b are the two channel halves. in: [B, 2C, L] row-major, out: [B, C, L].
// Layout: in[(bi*2c + co)*l + li] = a[bi, co, li] for co ∈ [0, c);
//        in[(bi*2c + c + co)*l + li] = b[bi, co, li] for co ∈ [0, c).
// One output element per thread. Scalar __half access (not __half2) so
// odd l doesn't cause CUDA_ERROR_MISALIGNED_ADDRESS — TEnc stride 4 on
// 343980 samples gives l=85995 (odd), which would break a __half2 path
// because base_a = oc*l would be odd when oc is odd.
extern "C" __global__ void __launch_bounds__(1024, 4)
glu_channel_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c, int l
) {
    int total = b * c * l;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int co = tot / l;
    int bi = co / c;
    int oc = co - bi * c;
    int li = tot - co * l;
    int base_a = (bi * 2 * c + oc) * l;
    int base_b = (bi * 2 * c + c + oc) * l;
    float a = __half2float(in[base_a + li]);
    float bv = __half2float(in[base_b + li]);
    float sig = 1.0f / (1.0f + __expf(-bv));
    // Output is only b*c*l (GLU halves channels), so the output index is
    // (bi * c + oc) * l + li — NOT base_a + li (which would walk past
    // the output buffer for bi > 0).
    out[(bi * c + oc) * l + li] = __float2half(a * sig);
}

// ─── LayerScale: x *= scale[c] (per-channel multiplicative) ────────────
// x [B, C, L] row-major, scale [C]. Per-channel scale, broadcast over B and L.
extern "C" __global__ void __launch_bounds__(1024, 4)
layer_scale_f16(
    __half* __restrict__ x,
    const __half* __restrict__ scale,
    int b, int c, int l
) {
    int total = b * c * l;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int ci = (idx / l) % c;
    x[idx] = __float2half(__half2float(x[idx]) * __half2float(scale[ci]));
}

// ─── LayerScale-last: x[outer, last] *= scale[last] (transformer γ₁/γ₂) ─
extern "C" __global__ void __launch_bounds__(1024, 4)
layer_scale_last_f16(
    __half* __restrict__ x,
    const __half* __restrict__ scale,
    int total, int last
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int j = idx % last;
    x[idx] = __float2half(__half2float(x[idx]) * __half2float(scale[j]));
}

// ─── Add bias broadcast: x[outer, last] += bias[last] ──────────────────
extern "C" __global__ void __launch_bounds__(1024, 4)
add_bias_inplace_f16(
    __half* __restrict__ x,
    const __half* __restrict__ bias,
    int outer, int last
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= outer * last) return;
    int j = tot % last;
    x[tot] = __float2half(__half2float(x[tot]) + __half2float(bias[j]));
}

// ─── Element-wise add: out = a + b (alloc-free, requires pre-allocated out) ─
extern "C" __global__ void __launch_bounds__(1024, 4)
add_to_f16(
    __half* __restrict__ out,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

// ─── In-place add: a += b ─────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(1024, 4)
add_inplace_f16(
    __half* __restrict__ a,
    const __half* __restrict__ b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

// ─── Zero-pad right: out[b, c, t_padded] = [x[b, c, :t], zeros] ──────
// Used by TEncLayer to right-pad the input to a multiple of STRIDE=4 so
// that the conv1d output length matches what the model was trained with.
extern "C" __global__ void __launch_bounds__(1024, 4)
zero_pad_right_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c, int t, int t_padded
) {
    int total = b * c * t_padded;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int ci = (tot / t_padded) % c;
    int ti = tot % t_padded;
    int bi = tot / (t_padded * c);
    if (ti < t) {
        out[tot] = in[(bi * c + ci) * t + ti];
    } else {
        out[tot] = __float2half(0.0f);
    }
}

// ─── Add positional embed: x[b, t, d] += pe[t, d] (broadcast over b) ──
// x [b*t, d] row-major (b contiguous, then t, then d), pe [t, d].
// b contiguous convention: row index = b * t + ti.
extern "C" __global__ void __launch_bounds__(512, 4)
add_pe_f16(
    __half* __restrict__ x,
    const __half* __restrict__ pe,
    int d, int t, int bt
) {
    int row = blockIdx.x;
    if (row >= bt) return;
    int ti = row % t;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    for (int j = tid; j < d; j += bs) {
        float v = __half2float(x[row * d + j]) + __half2float(pe[ti * d + j]);
        x[row * d + j] = __float2half(v);
    }
}

// ─── Add freq embed: x[b, c, fr, t] += emb[fr, c] * scale ─────────────
// x [b, c, fr, t] (PyTorch row-major), emb [fr, c] (also row-major).
// Broadcast over b and t; multiply by scale.
extern "C" __global__ void __launch_bounds__(256, 4)
add_freq_emb_f16(
    __half* __restrict__ x,
    const __half* __restrict__ emb,
    int b, int c, int fr, int t,
    float scale
) {
    int bi = blockIdx.z;
    if (bi >= b) return;
    int fi = blockIdx.y;
    if (fi >= fr) return;
    int ci = blockIdx.x;
    if (ci >= c) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    float emb_val = __half2float(emb[fi * c + ci]) * scale;
    for (int ti = tid; ti < t; ti += bs) {
        int idx = ((bi * c + ci) * fr + fi) * t + ti;
        x[idx] = __float2half(__half2float(x[idx]) + emb_val);
    }
}

// ─── Softmax with scale: x [B, S, N] → out ─────────────────────────
// scale = 1/sqrt(d_head). For each (b, s) row, computes softmax over N cols.
//
// Each warp handles one row independently; each block does 8 rows (256 threads).
// Warp-shuffle reduction avoids shared memory and __syncthreads() barriers.
// vs the old one-block-per-row kernel: 8× fewer blocks, no sync overhead.
extern "C" __global__ void __launch_bounds__(256, 4)
softmax_scaled_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int s, int n,
    float scale
) {
    constexpr int WARP = 32;
    int total_rows = b * s;
    int warp_id = threadIdx.x / WARP;
    int lane = threadIdx.x % WARP;
    int rows_per_block = blockDim.x / WARP;
    int row = blockIdx.x * rows_per_block + warp_id;
    if (row >= total_rows) return;

    const __half* src = in + row * n;
    __half* dst = out + row * n;

    // ── Pass 1: per-row max via warp shuffle ──
    float max_val = -INFINITY;
    for (int j = lane; j < n; j += WARP) {
        float v = __half2float(src[j]) * scale;
        if (v > max_val) max_val = v;
    }
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, max_val, off);
        if (other > max_val) max_val = other;
    }

    // ── Pass 2: sum of exp(x - max) via warp shuffle ──
    float sum_val = 0.0f;
    for (int j = lane; j < n; j += WARP) {
        sum_val += __expf(__half2float(src[j]) * scale - max_val);
    }
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) {
        sum_val += __shfl_xor_sync(0xffffffff, sum_val, off);
    }
    float inv_sum = 1.0f / sum_val;

    // ── Pass 3: write softmax ──
    for (int j = lane; j < n; j += WARP) {
        dst[j] = __float2half(__expf(__half2float(src[j]) * scale - max_val) * inv_sum);
    }
}

// ─── Denormalize: x = x * std + mean (broadcast, [B, 1, ...] shape) ──
// For [B, C, H, W]: x[b,c,h,w] = x[b,c,h,w] * std[b] + mean[b]
extern "C" __global__ void __launch_bounds__(1024, 4)
denorm_freq_f16(
    __half* __restrict__ x,
    const __half* __restrict__ mean,
    const __half* __restrict__ std,
    int b, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= b * n) return;
    int bi = idx / n;
    float v = __half2float(x[idx]) * __half2float(std[bi]) + __half2float(mean[bi]);
    x[idx] = __float2half(v);
}

// ─── Normalize: x = (x - mean) / (std + eps) ─────────────────────────
// For [B, C, H, W]: x[b,c,h,w] = (x[b,c,h,w] - mean[b]) / (std[b] + eps)
extern "C" __global__ void __launch_bounds__(1024, 4)
norm_freq_f16(
    __half* __restrict__ x,
    const __half* __restrict__ mean,
    const __half* __restrict__ std,
    int b, int n,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= b * n) return;
    int bi = idx / n;
    float v = (__half2float(x[idx]) - __half2float(mean[bi])) / (__half2float(std[bi]) + eps);
    x[idx] = __float2half(v);
}

// ─── LayerNorm: x [outer, last] * weight[last] + bias[last] (f32 acc) ────
// mean/var per row. eps = 1e-5. Used for transformer norm1/norm2/norm3.
extern "C" __global__ void __launch_bounds__(1024, 4)
layer_norm_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ bias,
    int last, int outer,
    float eps
) {
    int row = blockIdx.x;
    if (row >= outer) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    extern __shared__ float sdata[];

    float l_sum = 0.0f, l_sq = 0.0f;
    for (int j = tid; j < last; j += bs) {
        float v = __half2float(x[row * last + j]);
        l_sum += v;
        l_sq  += v * v;
    }
    sdata[tid] = l_sum;
    sdata[tid + bs] = l_sq;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid]     += sdata[tid + s];
            sdata[tid + bs] += sdata[tid + bs + s];
        }
        __syncthreads();
    }
    float mean = sdata[0] / (float)last;
    float var  = sdata[bs] / (float)last - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int j = tid; j < last; j += bs) {
        float v = (__half2float(x[row * last + j]) - mean) * inv_std;
        out[row * last + j] = __float2half(v * __half2float(w[j]) + __half2float(bias[j]));
    }
}

// ─── Swap dims 1 and 2 of a 3D tensor [d0, d1, d2] → [d0, d2, d1] ────
extern "C" __global__ void __launch_bounds__(1024, 4)
swap_dims_12_3d_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int d0, int d1, int d2
) {
    // Output is [d0, d2, d1] (dims 1 and 2 of the [d0,d1,d2] source swapped).
    // tot indexes the OUTPUT row-major: fastest = d1, then d2, then d0.
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2;
    if (tot >= total) return;
    int a = tot % d1;                 // output dim-2 index (ranges over d1)
    int b = (tot / d1) % d2;          // output dim-1 index (ranges over d2)
    int i0 = tot / (d1 * d2);         // output dim-0 index (ranges over d0)
    // out[i0, b, a] = src[i0, a, b]  (swap dims 1 and 2)
    int src_idx = (i0 * d1 + a) * d2 + b;
    dst[tot] = src[src_idx];
}

// ─── Swap dims 1 and 2 of a 4D tensor [d0, d1, d2, d3] → [d0, d2, d1, d3] ─
// For MyGroupNorm transpose on [B, S, D] → [B, D, S].
extern "C" __global__ void __launch_bounds__(1024, 4)
swap_dims_12_4d_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int d0, int d1, int d2, int d3
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2 * d3;
    if (tot >= total) return;
    int i3 = tot % d3;
    int i2 = (tot / d3) % d2;
    int i1 = (tot / (d3 * d2)) % d1;
    int i0 = tot / (d3 * d2 * d1);
    int src_idx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
    dst[tot] = src[src_idx];
}

// ─── Permute [b, c, f, t] → [b, t, c, f] (used in cross-attn reshape) ─
extern "C" __global__ void __launch_bounds__(512, 4)
permute_bcft_to_btcf_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int b, int c, int f, int t
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * c * f * t;
    if (tot >= total) return;
    int it = tot % t;
    int rem = tot / t;
    int if_ = rem % f;
    rem /= f;
    int ic = rem % c;
    int ib = rem / c;
    int dst_idx = ((ib * t + it) * c + ic) * f + if_;
    dst[dst_idx] = src[tot];
}

// ─── Trim + reshape: input [B, C, Fr, W] → output [B, C, Fr_target, W] ──
// Used by HDec to drop the extra freq slots from ConvTranspose2d (which can
// produce a few more than needed for 4× upsampling at certain sizes).
extern "C" __global__ void __launch_bounds__(1024, 4)
trim_h2_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c, int fr, int fr_target, int w
) {
    int total = b * c * fr_target * w;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int iw = tot % w;
    int fr_i = (tot / w) % fr_target;
    int co = (tot / (w * fr_target)) % c;
    int bi = tot / (w * fr_target * c);
    int src = ((bi * c + co) * fr + fr_i) * w + iw;
    out[tot] = in[src];
}

// ─── Trim + reshape: input [B, C, L] → output [B, C, L_target] ──────────
// Used by TDec after ConvTranspose1d.
extern "C" __global__ void __launch_bounds__(1024, 4)
trim_l_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c, int l, int l_target
) {
    int total = b * c * l_target;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int li = tot % l_target;
    int co = (tot / l_target) % c;
    int bi = tot / (l_target * c);
    int src = (bi * c + co) * l + li;
    out[tot] = in[src];
}

// ─── Transpose [B, C, H, W] → [B, H, W, C] (NHWC, for 1x1 conv via GEMM) ─
// out[(b*H + h)*W*C + w*C + c] = in[((b*C + c)*H + h)*W + w]
extern "C" __global__ void __launch_bounds__(1024, 4)
transpose_bchw_to_bhwc_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c, int h, int w
) {
    int total = b * c * h * w;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int iw = tot % w;
    int ih = (tot / w) % h;
    int co = (tot / (w * h)) % c;
    int bi = tot / (w * h * c);
    int dst = ((bi * h + ih) * w + iw) * c + co;
    out[dst] = in[tot];
}

// ─── Permute [B, S, D] → [B, h, S, d_head] for multi-head attention ────
// in[(b*S + s) * D + d] → out[((b*h + h_) * S + s) * d_head + dh]
// where d = h_ * d_head + dh, h_ = d / d_head, dh = d % d_head.
// After this permute, the tensor is logically [B*h, S, d_head] row-major.
extern "C" __global__ void __launch_bounds__(1024, 4)
permute_bsd_to_bhsd_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int s, int d, int h, int d_head
) {
    int total = b * s * d;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int d_ = tot % d;
    int si = (tot / d) % s;
    int bi = tot / (d * s);
    int h_ = d_ / d_head;
    int dh = d_ - h_ * d_head;
    int dst = ((bi * h + h_) * s + si) * d_head + dh;
    out[dst] = in[tot];
}

// ─── Permute [B*h, S, d_head] → [B*h, d_head, S] (per-head transpose) ──
// Used for K^T in attention: in[(bh*S + s)*d_head + dh] → out[(bh*d_head + dh)*S + s]
extern "C" __global__ void __launch_bounds__(1024, 4)
permute_bhsd_to_bhds_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int bh, int s, int d_head
) {
    int total = bh * s * d_head;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int dh = tot % d_head;
    int si = (tot / d_head) % s;
    int bh_ = tot / (d_head * s);
    int dst = (bh_ * d_head + dh) * s + si;
    out[dst] = in[tot];
}
// in[((b*h + h_) * S + s) * d_head + dh] → out[(b*S + s) * D + h_*d_head + dh]
extern "C" __global__ void __launch_bounds__(1024, 4)
permute_bhsd_to_bsd_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int s, int h, int d_head
) {
    int total = b * h * s * d_head;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int dh = tot % d_head;
    int si = (tot / d_head) % s;
    int h_ = (tot / (d_head * s)) % h;
    int bi = tot / (d_head * s * h);
    int dst = (bi * s + si) * (h * d_head) + h_ * d_head + dh;
    out[dst] = in[tot];
}

// ─── Copy a per-head subview of [B*h, S, d_head] to [S, d_head] ──────
// Extracts the (bh)-th head from a `[B*h, S, d_head]` source into a
// contiguous `[S, d_head]` buffer. Used for per-head GEMM.
extern "C" __global__ void __launch_bounds__(1024, 4)
copy_per_head_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int bh, int s, int d_head
) {
    int total = s * d_head;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    out[tot] = in[bh * s * d_head + tot];
}

// ─── Inverse of copy_per_head: scatter [S, d_head] into [B*h, S, d_head] at slot bh
extern "C" __global__ void
scatter_per_head_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int bh, int s, int d_head
) {
    int total = s * d_head;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    out[bh * s * d_head + tot] = in[tot];
}
extern "C" __global__ void __launch_bounds__(1024, 4)
transpose_bhwc_to_bchw_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c, int h, int w
) {
    int total = b * c * h * w;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int iw = tot % w;
    int ih = (tot / w) % h;
    int co = (tot / (w * h)) % c;
    int bi = tot / (w * h * c);
    int src = ((bi * h + ih) * w + iw) * c + co;
    out[tot] = in[src];
}

// ─── Reshape [B, C, F, T] → [B*F, C, T] (used by henc to feed dconv) ──
// Output strides: ((bf*F + fr)*C + c)*T + t
// Input  strides: ((b*C + c)*F + fr)*T + t
extern "C" __global__ void __launch_bounds__(1024, 4)
reshape_bcft_to_bfct_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c, int fr, int t
) {
    int total = b * c * fr * t;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int ti = tot % t;
    int fr_i = (tot / t) % fr;
    int co = (tot / (t * fr)) % c;
    int bi = tot / (t * fr * c);
    int dst = ((bi * fr + fr_i) * c + co) * t + ti;
    out[dst] = in[tot];
}

// ─── Reshape [B*F, C, T] → [B, C, F, T] (inverse, hdec path) ──────────
extern "C" __global__ void __launch_bounds__(1024, 4)
reshape_bfct_to_bcft_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c, int fr, int t
) {
    int total = b * c * fr * t;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int ti = tot % t;
    int fr_i = (tot / t) % fr;
    int co = (tot / (t * fr)) % c;
    int bi = tot / (t * fr * c);
    int src = ((bi * fr + fr_i) * c + co) * t + ti;
    out[tot] = in[src];
}

// One thread per (r, c) pair. Generic square-or-rectangular transpose.
extern "C" __global__ void __launch_bounds__(32, 8)
transpose_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int rows, int cols
) {
    __shared__ __half tile[32][33];  // +1 to avoid bank conflicts
    int bx = blockIdx.x * 32;
    int by = blockIdx.y * 32;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = bx + tx;
    int y = by + ty;
    if (x < cols && y < rows) {
        tile[ty][tx] = src[y * cols + x];
    }
    __syncthreads();
    int x2 = by + tx;
    int y2 = bx + ty;
    if (x2 < rows && y2 < cols) {
        dst[y2 * rows + x2] = tile[tx][ty];
    }
}

// ─── Fused GELU + bias add (for linear/conv1d postprocess) ────────────
// x [outer, last] += bias[last]; y = gelu(x). In-place if out == in.
extern "C" __global__ void __launch_bounds__(1024, 4)
gelu_bias_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    const __half* __restrict__ bias,
    int outer, int last
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * last;
    if (tot >= total) return;
    int j = tot % last;
    float v = __half2float(in[tot]) + __half2float(bias[j]);
    float g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    out[tot] = __float2half(g);
}

// ─── Pure GELU (no bias) ──────────────────────────────────────────────
// x [n] in-place. Matches ops_cpu::gelu.
extern "C" __global__ void __launch_bounds__(1024, 4)
gelu_f16(
    __half* __restrict__ x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    float g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    x[i] = __float2half(g);
}

// ─── im2col for conv2d k=8 s=4 p=2 (HEncLayer/TEncLayer) ─────────────
// in [b, c_in, h, w]; out [b*h_out*w_out, c_in*8*1] (kW=1).
// Block per spatial position; threads loop over (ci, ky) pairs via stride.
// Eliminates per-element integer division (was 6 divs/element → 2 divs/block).
extern "C" __global__ void __launch_bounds__(256, 4)
im2col_8x1_s4p2_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c_in, int h, int w,
    int spatial_per_batch, int w_out
) {
    int spatial = blockIdx.x;
    int bi = spatial / spatial_per_batch;
    int rem = spatial - bi * spatial_per_batch;
    int oh = rem / w_out;
    int ow = rem - oh * w_out;

    int n_ci_k = c_in * 8;
    int out_base = spatial * n_ci_k;
    for (int ci_k = threadIdx.x; ci_k < n_ci_k; ci_k += blockDim.x) {
        int kk = ci_k & 7;         // ci_k % 8
        int ci = ci_k >> 3;        // ci_k / 8
        int in_y = oh * 4 + kk - 2;
        __half v;
        if (in_y < 0 || in_y >= h) {
            v = __float2half(0.0f);
        } else {
            v = in[((bi * c_in + ci) * h + in_y) * w + ow];
        }
        out[out_base + ci_k] = v;
    }
}

// ─── im2col for conv1d k=8 s=4 p=2 (TEncLayer/TDecLayer) ─────────────
// in [b, c_in, l]; out [b*l_out, c_in*8].
// Block per spatial position; threads loop over (ci, kk) pairs.
extern "C" __global__ void __launch_bounds__(256, 4)
im2col_8_s4p2_1d_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c_in, int l, int l_out
) {
    int spatial = blockIdx.x;
    int bi = spatial / l_out;
    int ol = spatial - bi * l_out;

    int n_ci_k = c_in * 8;
    int out_base = spatial * n_ci_k;
    for (int ci_k = threadIdx.x; ci_k < n_ci_k; ci_k += blockDim.x) {
        int kk = ci_k & 7;
        int ci = ci_k >> 3;
        int in_l = ol * 4 + kk - 2;
        __half v;
        if (in_l < 0 || in_l >= l) {
            v = __float2half(0.0f);
        } else {
            v = in[(bi * c_in + ci) * l + in_l];
        }
        out[out_base + ci_k] = v;
    }
}

// ─── im2col for conv1d k=3 stride=1 pad=dilation, dilation param (DConv) ─
// in [b, c_in, l]; out [b*l_out, c_in*3]. l_out = l (same length, given pad).
// Block per spatial position; threads loop over (ci, kk) pairs.
extern "C" __global__ void __launch_bounds__(256, 4)
im2col_1d_k3_dilation_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c_in, int l, int l_out,
    int dilation
) {
    int spatial = blockIdx.x;
    int bi = spatial / l_out;
    int ol = spatial - bi * l_out;

    int n_ci_k = c_in * 3;
    int out_base = spatial * n_ci_k;
    for (int ci_k = threadIdx.x; ci_k < n_ci_k; ci_k += blockDim.x) {
        // kk = ci_k % 3, ci = ci_k / 3 — use precomputed via loop below
        int kk = ci_k - (ci_k / 3) * 3;  // ci_k % 3 (3 is small, compiler optimizes)
        int ci = ci_k / 3;
        int in_l = ol + (kk - 1) * dilation;
        __half v;
        if (in_l < 0 || in_l >= l) {
            v = __float2half(0.0f);
        } else {
            v = in[(bi * c_in + ci) * l + in_l];
        }
        out[out_base + ci_k] = v;
    }
}

// ─── im2col for conv1d k=1 stride=1 pad=0 (DConv inner conv2) ─────────
// in [b, c_in, l]; out [b*l, c_in] (identity reshape).
// Block per spatial position; threads loop over channels.
extern "C" __global__ void __launch_bounds__(256, 4)
im2col_1d_k1_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c_in, int l
) {
    int spatial = blockIdx.x;
    int bi = spatial / l;
    int ol = spatial - bi * l;

    int out_base = spatial * c_in;
    int in_base = (bi * c_in) * l + ol;
    for (int ci = threadIdx.x; ci < c_in; ci += blockDim.x) {
        out[out_base + ci] = in[in_base + ci * l];
    }
}

// ─── im2col for conv_transpose2d k=8 s=4 p=2 (HDecLayer conv_tr) ───
// in [b, c_in, h_in, w_in]; out [b*h_out*w_out, c_in*8*1] (kW=1).
// For each output (oh, ow) and kernel offset dkh:
//   ih = (oh + pad_h - dkh) / stride_h   (only valid if (oh+pad-dkh) % stride == 0)
//   iw = ow                              (kW=1, no kw offset)
// If invalid or out-of-range, write 0.
// Block per spatial position; threads loop over (ci, kk) pairs.
// Eliminates per-element integer division (was 6 divs/element → 2 divs/block).
extern "C" __global__ void __launch_bounds__(256, 4)
im2col_conv_transpose_8x1_s4p2_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c_in, int h_in, int w_in,
    int h_out, int w_out,
    int pad_h, int stride_h
) {
    int spatial = blockIdx.x;
    int spatial_per_batch = h_out * w_out;
    int bi = spatial / spatial_per_batch;
    int rem = spatial - bi * spatial_per_batch;
    int oh = rem / w_out;
    int ow = rem - oh * w_out;

    int n_ci_k = c_in * 8;
    int out_base = spatial * n_ci_k;
    for (int ci_k = threadIdx.x; ci_k < n_ci_k; ci_k += blockDim.x) {
        int kk = ci_k & 7;
        int ci = ci_k >> 3;
        int oh_p = oh + pad_h - kk;
        __half v;
        if (oh_p < 0 || (oh_p & (stride_h - 1)) != 0) {
            v = __float2half(0.0f);
        } else {
            int ih = oh_p / stride_h;
            if (ih < 0 || ih >= h_in) {
                v = __float2half(0.0f);
            } else {
                v = in[((bi * c_in + ci) * h_in + ih) * w_in + ow];
            }
        }
        out[out_base + ci_k] = v;
    }
}

// ─── im2col for conv_transpose1d k=8 s=4 p=2 (TDecLayer conv_tr) ───
// in [b, c_in, l_in]; out [b*l_out, c_in*8].
// Block per spatial position; threads loop over (ci, kk) pairs.
extern "C" __global__ void __launch_bounds__(256, 4)
im2col_conv_transpose_8_s4p2_1d_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c_in, int l_in, int l_out,
    int pad, int stride
) {
    int spatial = blockIdx.x;
    int bi = spatial / l_out;
    int ol = spatial - bi * l_out;

    int n_ci_k = c_in * 8;
    int out_base = spatial * n_ci_k;
    for (int ci_k = threadIdx.x; ci_k < n_ci_k; ci_k += blockDim.x) {
        int kk = ci_k & 7;
        int ci = ci_k >> 3;
        int ol_p = ol + pad - kk;
        __half v;
        if (ol_p < 0 || (ol_p & (stride - 1)) != 0) {
            v = __float2half(0.0f);
        } else {
            int il = ol_p / stride;
            if (il < 0 || il >= l_in) {
                v = __float2half(0.0f);
            } else {
                v = in[(bi * c_in + ci) * l_in + il];
            }
        }
        out[out_base + ci_k] = v;
    }
}

// ─── im2col for conv2d k=3 s=1 p=1 (HDecLayer rewrite) ──────────────
// in [b, c_in, h, w]; out [b*h_out*w_out, c_in*3*3]. h_out = h, w_out = w.
// Block per spatial position; threads loop over ci, unrolled inner loop over 3×3.
extern "C" __global__ void __launch_bounds__(256, 4)
im2col_3x3_s1p1_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c_in, int h, int w, int h_out, int w_out
) {
    int spatial = blockIdx.x;
    int spatial_per_batch = h_out * w_out;
    int bi = spatial / spatial_per_batch;
    int rem = spatial - bi * spatial_per_batch;
    int oh = rem / w_out;
    int ow = rem - oh * w_out;

    int out_base = spatial * (c_in * 9);
    for (int ci = threadIdx.x; ci < c_in; ci += blockDim.x) {
        int ci_base = out_base + ci * 9;
        // Unrolled 3×3 kernel loop
        #pragma unroll
        for (int ky = 0; ky < 3; ky++) {
            int in_y = oh + ky - 1;
            bool y_valid = (in_y >= 0 && in_y < h);
            #pragma unroll
            for (int kx = 0; kx < 3; kx++) {
                int in_x = ow + kx - 1;
                __half v;
                if (y_valid && in_x >= 0 && in_x < w) {
                    v = in[((bi * c_in + ci) * h + in_y) * w + in_x];
                } else {
                    v = __float2half(0.0f);
                }
                out[ci_base + ky * 3 + kx] = v;
            }
        }
    }
}

// ─── Conv2d postprocess: GEMM output [B, h_out*w_out, c_out] row-major ─
// (from im2col × weight) → add bias → optionally GELU → reshape to [B, c_out, h_out, w_out].
// gemm_out layout: rows = b*h_out*w_out (in that order), cols = c_out.
// in: gemm_out, bias, b, c_out, h_out, w_out.
// Optional gelu: set gelu=1 to apply.
extern "C" __global__ void __launch_bounds__(1024, 4)
conv2d_postprocess_f16(
    __half* __restrict__ out,
    const __half* __restrict__ gemm_out,
    const __half* __restrict__ bias,
    int b, int c_out, int h_out, int w_out,
    int apply_gelu
) {
    int total = b * c_out * h_out * w_out;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int iw = tot % w_out;
    int ih = (tot / w_out) % h_out;
    int co = (tot / (w_out * h_out)) % c_out;
    int ib = tot / (w_out * h_out * c_out);
    int gemm_idx = ((ib * h_out + ih) * w_out + iw) * c_out + co;
    float v = __half2float(gemm_out[gemm_idx]) + __half2float(bias[co]);
    if (apply_gelu) {
        v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    }
    out[tot] = __float2half(v);
}

// ─── Flatten [B, C, F, T] → [B, T*F, C] (time-major) for cross-domain TX ──
// out[((bi*T*F + ti*F + fri)*C + ci)] = in[((bi*C + ci)*F + fri)*T + ti]
extern "C" __global__ void __launch_bounds__(1024, 4)
flatten_bcft_to_btfc_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c, int f, int t
) {
    int total = b * c * f * t;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int ti = tot % t;
    int rem = tot / t;
    int fri = rem % f;
    rem /= f;
    int ci = rem % c;
    int bi = rem / c;
    int dst = ((bi * t * f + ti * f + fri) * c + ci);
    out[dst] = in[tot];
}

// ─── Inverse: [B, T*F, C] → [B, C, F, T] ──────────────────────────────
extern "C" __global__ void __launch_bounds__(1024, 4)
unflatten_btfc_to_bcft_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c, int f, int t
) {
    int total = b * c * f * t;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int ti = tot % t;
    int rem = tot / t;
    int fri = rem % f;
    rem /= f;
    int ci = rem % c;
    int bi = rem / c;
    int src = ((bi * t * f + ti * f + fri) * c + ci);
    out[tot] = in[src];
}

// ─── f16 → f32 element-wise: avoid 700ms CPU conversion on D2H path ────
extern "C" __global__ void __launch_bounds__(1024, 4)
convert_f16_to_f32_f32(
    float* __restrict__ out,
    const __half* __restrict__ in,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __half2float(in[i]);
}
