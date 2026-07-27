//! GPU-side ops for HTDemucs v4.
//!
//! Each op corresponds to a CPU op in `ops_cpu.rs`. Implementations use
//! cuBLAS for matmul-backed ops (linear / conv / MHA) and NVRTC kernels
//! for element-wise ops. All storage is f16; arithmetic uses f32
//! accumulation in the kernels.
//!
//! Convention: ops that are conceptually in-place take `GpuTensor` by
//! value (the underlying `CudaSlice<f16>` is `!Clone` so passing by value
//! is cheap). Ops that return a new tensor allocate a fresh `CudaSlice`.

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use cudarc::cublas::sys;
use cudarc::driver::{safe::CudaEvent, safe::PushKernelArg, CudaSlice, CudaView, CudaViewMut, LaunchConfig};
use half::f16;

use crate::cuda_engine::{CudaState, GpuTensor};
use crate::gpu_model::{
    GpuBias, GpuConv1dWeight, GpuConv2dWeight, GpuGroupNorm1, GpuLayerNorm1, GpuLayerScale,
    GpuWeight2D,
};

// ─── phase timer (gated by env var DEMUCS_CUDA_PROFILE=1) ─────────────

/// Per-phase GPU timing accumulator. Each phase gets a pair of CudaEvents
/// recorded on the stream; `elapsed_ms` is summed into the named bucket.
/// Activated only when `DEMUCS_CUDA_PROFILE=1` is set.
pub(crate) struct CudaPhaseTimer {
    enabled: bool,
    map: std::collections::HashMap<String, f64>,
    starts: std::collections::HashMap<String, CudaEvent>,
    ends: std::collections::HashMap<String, CudaEvent>,
    print_at_end: bool,
}

impl CudaPhaseTimer {
    pub fn new(enabled: bool) -> Self {
        Self { enabled, map: Default::default(), starts: Default::default(), ends: Default::default(), print_at_end: false }
    }
    /// Construct a timer that only prints aggregated totals at process end
    /// (via the global accumulator) instead of per-call. Use this for noisy
    /// per-op timings to avoid 30+ printouts per run.
    pub fn new_aggregate(enabled: bool) -> Self {
        Self { enabled, map: Default::default(), starts: Default::default(), ends: Default::default(), print_at_end: true }
    }
    pub fn is_enabled(&self) -> bool { self.enabled }
    pub fn start(&mut self, state: &Arc<CudaState>, name: &str) {
        if !self.enabled { return; }
        let ev = state.ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).expect("cuda event start");
        ev.record(&state.stream).expect("record start");
        self.starts.insert(name.to_string(), ev);
    }
    pub fn end(&mut self, state: &Arc<CudaState>, name: &str) {
        if !self.enabled { return; }
        let ev = state.ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).expect("cuda event end");
        ev.record(&state.stream).expect("record end");
        self.ends.insert(name.to_string(), ev);
    }
    pub fn resolve(&mut self) {
        if !self.enabled { return; }
        for (name, end_ev) in self.ends.drain() {
            if let Some(start_ev) = self.starts.remove(&name) {
                start_ev.synchronize().ok();
                end_ev.synchronize().ok();
                let ms = start_ev.elapsed_ms(&end_ev).unwrap_or(0.0) as f64;
                *self.map.entry(name).or_insert(0.0) += ms;
            }
        }
    }
    pub fn print(&self, label: &str) {
        if !self.enabled { return; }
        if self.print_at_end {
            let mut g = GLOBAL_AGG.lock().unwrap();
            let map = g.get_or_insert_with(Default::default);
            for (k, v) in &self.map { *map.entry(k.clone()).or_insert(0.0) += v; }
            return;
        }
        let mut entries: Vec<_> = self.map.iter().collect();
        entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let total: f64 = entries.iter().map(|(_, v)| **v).sum();
        eprintln!("\n── CudaPhaseTimer[{}] (total {:.1}ms) ──", label, total);
        for (k, v) in entries {
            let pct = if total > 0.0 { *v / total * 100.0 } else { 0.0 };
            eprintln!("  {:40} {:8.1}ms  {:5.1}%", k, v, pct);
        }
    }
}

/// Global aggregator for cross-thread profiling (3-thread pipeline safe).
static GLOBAL_AGG: Mutex<Option<std::collections::HashMap<String, f64>>> = Mutex::new(None);

/// Drain and print the global aggregate. Call once at process end.
pub fn print_global_agg() {
    let mut guard = GLOBAL_AGG.lock().unwrap();
    let g = guard.take();
    if let Some(g) = g {
        if g.is_empty() { return; }
        let mut entries: Vec<_> = g.iter().collect();
        entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let total: f64 = entries.iter().map(|(_, v)| **v).sum();
        eprintln!("\n═══ CudaGlobalAgg (total {:.1}ms) ═══", total);
        for (k, v) in entries {
            let pct = if total > 0.0 { *v / total * 100.0 } else { 0.0 };
            eprintln!("  {:40} {:8.1}ms  {:5.1}%", k, v, pct);
        }
    }
}

// ─── launch helpers ──────────────────────────────────────────────────

/// Compute (grid, block) for a 1-D element-wise op over `n` elements.
fn launch_1d(n: usize) -> LaunchConfig {
    let block = 256u32;
    let grid = ((n as u32) + block - 1) / block;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Element-wise ops
// ═══════════════════════════════════════════════════════════════════════

/// In-place GELU: `x[i] = x[i] * 0.5 * (1 + erf(x[i] / √2))`.
/// Matches `ops_cpu::gelu`.
pub fn gelu_inplace(state: &Arc<CudaState>, mut x: GpuTensor) -> Result<GpuTensor> {
    let n = x.data.len();
    if n == 0 {
        return Ok(x);
    }
    let cfg = launch_1d(n);
    let n_i = n as i32;
    let mut bb = state.stream.launch_builder(&state.k.gelu);
    bb.arg(&mut x.data);
    bb.arg(&n_i);
    unsafe { bb.launch(cfg) }?;
    Ok(x)
}

/// In-place layer scale: `x[b, c, l] *= scale[c]`.
/// Matches `ops_cpu::layer_scale` for [B, C, L] layout.
pub fn layer_scale_inplace(
    state: &Arc<CudaState>,
    mut x: GpuTensor,
    scale: &GpuLayerScale,
) -> Result<GpuTensor> {
    let [b, c, l] = [x.shape[0], x.shape[1], x.shape[2]];
    let n = b * c * l;
    if n == 0 {
        return Ok(x);
    }
    let cfg = launch_1d(n);
    let b_i = b as i32;
    let c_i = c as i32;
    let l_i = l as i32;
    let mut bb = state.stream.launch_builder(&state.k.layer_scale);
    bb.arg(&mut x.data);
    bb.arg(&scale.scale);
    bb.arg(&b_i);
    bb.arg(&c_i);
    bb.arg(&l_i);
    unsafe { bb.launch(cfg) }?;
    Ok(x)
}

/// LayerScale on the last dim: `x[..., d] *= scale[d]`. For transformer
/// γ₁/γ₂ applied to `[B, S, D]` (D = last). In-place; x may be any shape,
/// the kernel just needs total element count and the last-dim size.
pub fn layer_scale_last_inplace(
    state: &Arc<CudaState>,
    mut x: GpuTensor,
    scale: &GpuLayerScale,
) -> Result<GpuTensor> {
    let total = x.numel();
    if total == 0 {
        return Ok(x);
    }
    let last = *x.shape.last().unwrap();
    let cfg = launch_1d(total);
    let total_i = total as i32;
    let last_i = last as i32;
    let mut bb = state
        .stream
        .launch_builder(&state.k.layer_scale_last);
    bb.arg(&mut x.data);
    bb.arg(&scale.scale);
    bb.arg(&total_i);
    bb.arg(&last_i);
    unsafe { bb.launch(cfg) }?;
    Ok(x)
}

/// Element-wise add: out = a + b (allocates new buffer).
pub fn add_to(state: &Arc<CudaState>, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
    if a.numel() != b.numel() {
        return Err(anyhow!(
            "add_to: shape mismatch (a.numel={} != b.numel={}) a.shape={:?} b.shape={:?}",
            a.numel(),
            b.numel(),
            a.shape,
            b.shape
        ));
    }
    let n = a.numel();
    let mut out_data = state.alloc_uninit_f16(n)?;
    let cfg = launch_1d(n);
    let n_i = n as i32;
    let mut bb = state.stream.launch_builder(&state.k.add_to);
    bb.arg(&mut out_data);
    bb.arg(&a.data);
    bb.arg(&b.data);
    bb.arg(&n_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out_data,
        shape: a.shape.clone(),
    })
}

// ═══════════════════════════════════════════════════════════════════════
//  Normalisation kernels (groupnorm1 / layer_norm / freq-norm)
// ═══════════════════════════════════════════════════════════════════════

/// GroupNorm1 (1 group, per-batch) on `[B, C, L]`. Matches `ops_cpu::groupnorm1`.
pub fn groupnorm1_inplace(
    state: &Arc<CudaState>,
    x: &mut GpuTensor,
    gn: &GpuGroupNorm1,
) -> Result<()> {
    let [b, c, l] = [x.shape[0], x.shape[1], x.shape[2]];
    let cl = c * l;
    if cl == 0 || b == 0 {
        return Ok(());
    }
    // Block size: next pow2 of cl, capped at 1024.
    let mut bs = 1usize;
    while bs < cl.min(1024) {
        bs <<= 1;
    }
    let cfg = LaunchConfig {
        grid_dim: (b as u32, 1, 1),
        block_dim: (bs as u32, 1, 1),
        shared_mem_bytes: (2 * bs * std::mem::size_of::<f32>()) as u32,
    };
    let eps = 1e-5f32;
    let b_i = b as i32;
    let c_i = c as i32;
    let l_i = l as i32;
    let mut bb = state.stream.launch_builder(&state.k.groupnorm1);
    bb.arg(&mut x.data);
    bb.arg(&gn.gamma);
    bb.arg(&gn.beta);
    bb.arg(&b_i);
    bb.arg(&c_i);
    bb.arg(&l_i);
    bb.arg(&eps);
    unsafe { bb.launch(cfg) }?;
    Ok(())
}

/// LayerNorm on `[outer, last]`. Matches `ops_cpu::layernorm`.
pub fn layer_norm(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    ln: &GpuLayerNorm1,
    outer: usize,
    last: usize,
) -> Result<GpuTensor> {
    let mut out = state.alloc_uninit_f16(outer * last)?;
    let cfg = LaunchConfig {
        grid_dim: (outer as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (2 * 256 * std::mem::size_of::<f32>()) as u32,
    };
    let eps = 1e-5f32;
    let last_i = last as i32;
    let outer_i = outer as i32;
    let mut bb = state.stream.launch_builder(&state.k.layer_norm);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&ln.gamma);
    bb.arg(&ln.beta);
    bb.arg(&last_i);
    bb.arg(&outer_i);
    bb.arg(&eps);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![outer, last],
    })
}

/// GLU along channel dim: input `[B, 2C, L]` → output `[B, C, L]`.
/// Allocates a fresh buffer (size = b * c * l).
pub fn glu_channel(state: &Arc<CudaState>, x: &GpuTensor) -> Result<GpuTensor> {
    let [b, c2, l] = [x.shape[0], x.shape[1], x.shape[2]];
    if c2 % 2 != 0 {
        return Err(anyhow!("glu_channel: c2={} not even", c2));
    }
    let c = c2 / 2;
    let mut out = state.alloc_uninit_f16(b * c * l)?;
    let cfg = LaunchConfig {
        grid_dim: (((b * c * l) as u32 + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_i = c as i32;
    let l_i = l as i32;
    let mut bb = state.stream.launch_builder(&state.k.glu_channel);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_i);
    bb.arg(&l_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c, l],
    })
}

// ═══════════════════════════════════════════════════════════════════════
//  Element-wise extras
// ═══════════════════════════════════════════════════════════════════════

/// In-place add: a += b. Same shape.
pub fn add_inplace(state: &Arc<CudaState>, a: &mut GpuTensor, b: &GpuTensor) -> Result<()> {
    if a.numel() != b.numel() {
        return Err(anyhow!(
            "add_inplace: a.numel={} != b.numel={}",
            a.numel(),
            b.numel()
        ));
    }
    let n = a.numel();
    if n == 0 {
        return Ok(());
    }
    let cfg = launch_1d(n);
    let n_i = n as i32;
    let mut bb = state.stream.launch_builder(&state.k.add_inplace);
    bb.arg(&mut a.data);
    bb.arg(&b.data);
    bb.arg(&n_i);
    unsafe { bb.launch(cfg) }?;
    Ok(())
}

/// Add freq-embed: x[b, c, fr, t] += emb[fr, c] * scale (in-place).
pub fn add_freq_emb_inplace(
    state: &Arc<CudaState>,
    x: &mut GpuTensor,
    emb: &crate::gpu_model::GpuFreqEmb,
    scale: f32,
) -> Result<()> {
    let [b, c, fr, t] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    if b == 0 || c == 0 || fr == 0 || t == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig {
        grid_dim: (c as u32, fr as u32, b as u32),
        block_dim: (256.min(t as u32), 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_i = c as i32;
    let fr_i = fr as i32;
    let t_i = t as i32;
    let mut bb = state.stream.launch_builder(&state.k.add_freq_emb);
    bb.arg(&mut x.data);
    bb.arg(&emb.data);
    bb.arg(&b_i);
    bb.arg(&c_i);
    bb.arg(&fr_i);
    bb.arg(&t_i);
    bb.arg(&scale);
    unsafe { bb.launch(cfg) }?;
    Ok(())
}

/// Denorm (per-batch): x[b, ...] = x[b, ...] * std[b] + mean[b].
/// Shape `[B, n]` where n = product of remaining dims.
pub fn denorm_freq_inplace(
    state: &Arc<CudaState>,
    x: &mut GpuTensor,
    mean: &GpuTensor,
    std: &GpuTensor,
) -> Result<()> {
    let b = x.shape[0];
    let n: usize = x.shape[1..].iter().product();
    let total = b * n;
    let cfg = launch_1d(total);
    let b_i = b as i32;
    let n_i = n as i32;
    let mut bb = state.stream.launch_builder(&state.k.denorm_freq);
    bb.arg(&mut x.data);
    bb.arg(&mean.data);
    bb.arg(&std.data);
    bb.arg(&b_i);
    bb.arg(&n_i);
    unsafe { bb.launch(cfg) }?;
    Ok(())
}

/// Norm (per-batch): x[b, ...] = (x[b, ...] - mean[b]) / (std[b] + eps).
pub fn norm_freq_inplace(
    state: &Arc<CudaState>,
    x: &mut GpuTensor,
    mean: &GpuTensor,
    std: &GpuTensor,
    eps: f32,
) -> Result<()> {
    let b = x.shape[0];
    let n: usize = x.shape[1..].iter().product();
    let total = b * n;
    let cfg = launch_1d(total);
    let b_i = b as i32;
    let n_i = n as i32;
    let mut bb = state.stream.launch_builder(&state.k.norm_freq);
    bb.arg(&mut x.data);
    bb.arg(&mean.data);
    bb.arg(&std.data);
    bb.arg(&b_i);
    bb.arg(&n_i);
    bb.arg(&eps);
    unsafe { bb.launch(cfg) }?;
    Ok(())
}

/// Swap dims 1 and 2 of `[d0, d1, d2]`.
pub fn swap_dims_12_3d(
    state: &Arc<CudaState>,
    x: &GpuTensor,
) -> Result<GpuTensor> {
    let [d0, d1, d2] = [x.shape[0], x.shape[1], x.shape[2]];
    let total = d0 * d1 * d2;
    let mut out = state.alloc_uninit_f16(total)?;
    let cfg = launch_1d(total);
    let d0_i = d0 as i32;
    let d1_i = d1 as i32;
    let d2_i = d2 as i32;
    let mut bb = state.stream.launch_builder(&state.k.swap_dims_12_3d);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&d0_i);
    bb.arg(&d1_i);
    bb.arg(&d2_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![d0, d2, d1],
    })
}

/// Permute `[b, c, f, t]` → `[b, t, c, f]`.
pub fn permute_bcft_to_btcf(
    state: &Arc<CudaState>,
    x: &GpuTensor,
) -> Result<GpuTensor> {
    let [b, c, f, t] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    let total = b * c * f * t;
    let mut out = state.alloc_uninit_f16(total)?;
    let cfg = LaunchConfig {
        grid_dim: (((total as u32) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_i = c as i32;
    let f_i = f as i32;
    let t_i = t as i32;
    let mut bb = state.stream.launch_builder(&state.k.permute_bcft_to_btcf);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_i);
    bb.arg(&f_i);
    bb.arg(&t_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, t, c, f],
    })
}

// ═══════════════════════════════════════════════════════════════════════
//  im2col (for conv2d / conv1d)
// ═══════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════
//  Conv ops (im2col + GEMM + bias + postprocess)
// ═══════════════════════════════════════════════════════════════════════

/// Conv1d for DConv inner conv2: kernel=1, stride=1, pad=0, dilation=1.
/// Input: `[B, C_in, L]` row-major. Output: `[B, C_out, L]` row-major.
pub fn conv1d_k1(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    w: &GpuConv1dWeight,
    bias: &GpuBias,
) -> Result<GpuTensor> {
    let [b, c_in, l] = [x.shape[0], x.shape[1], x.shape[2]];
    if c_in != w.in_ch {
        return Err(anyhow!("conv1d_k1: c_in={} != w.in_ch={}", c_in, w.in_ch));
    }
    let c_out = w.out_ch;
    let patch = w.k; // = c_in * 1
    // im2col for k=1 is identity reshape.
    let n_spatial = b * l;
    let mut col = state.alloc_uninit_f16(n_spatial * c_in)?;
    let cfg = LaunchConfig {
        grid_dim: (n_spatial as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_in_i = c_in as i32;
    let l_i = l as i32;
    let mut bb = state.stream.launch_builder(&state.k.im2col_1d_k1);
    bb.arg(&mut col);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_in_i);
    bb.arg(&l_i);
    unsafe { bb.launch(cfg) }?;
    // GEMM: out[b*l, c_out] = col[b*l, c_in] @ w[c_in, c_out]
    let gemm_out = state.gemm_f16(
        &GpuTensor {
            data: col,
            shape: vec![b * l, c_in],
        },
        &GpuTensor {
            data: w.data.clone(),
            shape: vec![c_in, c_out],
        },
        b * l,
        c_out,
        c_in,
    )?;
    // Bias add and reshape [B*L, C_out] → [B, C_out, L] via a manual layout switch:
    //   out[b, c, l] = gemm_out[b*L + l, c]
    // We use conv2d_postprocess with b, c_out, h_out=L, w_out=1 to do bias + layout.
    let mut out = state.alloc_uninit_f16(b * c_out * l)?;
    let cfg2 = launch_1d(b * c_out * l);
    let mut bb2 = state.stream.launch_builder(&state.k.conv2d_postprocess);
    let mut gemm_data = gemm_out.data;
    let c_out_i = c_out as i32;
    bb2.arg(&mut out);
    bb2.arg(&gemm_data);
    bb2.arg(&bias.data);
    bb2.arg(&b_i);
    bb2.arg(&c_out_i);
    bb2.arg(&l_i);
    bb2.arg(&1i32);
    bb2.arg(&0i32); // apply_gelu=0
    unsafe { bb2.launch(cfg2) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c_out, l],
    })
}

/// Conv1d with kernel=3, stride=1, pad=dilation, dilation=dilation (DConv conv1).
/// Output length == input length.
pub fn conv1d_k3_dilation(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    w: &GpuConv1dWeight,
    bias: &GpuBias,
    dilation: usize,
) -> Result<GpuTensor> {
    let [b, c_in, l] = [x.shape[0], x.shape[1], x.shape[2]];
    if c_in != w.in_ch {
        return Err(anyhow!(
            "conv1d_k3_dilation: c_in={} != w.in_ch={}",
            c_in,
            w.in_ch
        ));
    }
    if w.k != 3 {
        return Err(anyhow!("conv1d_k3_dilation: w.k={} != 3", w.k));
    }
    let c_out = w.out_ch;
    let l_out = l; // pad=dilation preserves length
    let patch = c_in * 3;
    let n_spatial = b * l_out;
    let mut col = state.alloc_uninit_f16(n_spatial * patch)?;
    let cfg = LaunchConfig {
        grid_dim: (n_spatial as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_in_i = c_in as i32;
    let l_i = l as i32;
    let l_out_i = l_out as i32;
    let dilation_i = dilation as i32;
    let mut bb = state.stream.launch_builder(&state.k.im2col_1d_k3_dilation);
    bb.arg(&mut col);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_in_i);
    bb.arg(&l_i);
    bb.arg(&l_out_i);
    bb.arg(&dilation_i);
    unsafe { bb.launch(cfg) }?;
    let gemm_out = state.gemm_f16(
        &GpuTensor {
            data: col,
            shape: vec![b * l_out, patch],
        },
        &GpuTensor {
            data: w.data.clone(),
            shape: vec![patch, c_out],
        },
        b * l_out,
        c_out,
        patch,
    )?;
    // Postprocess: add bias + reshape [B*L, C_out] → [B, C_out, L].
    let mut out = state.alloc_uninit_f16(b * c_out * l_out)?;
    let cfg2 = launch_1d(b * c_out * l_out);
    let c_out_i = c_out as i32;
    let mut bb2 = state.stream.launch_builder(&state.k.conv2d_postprocess);
    bb2.arg(&mut out);
    bb2.arg(&gemm_out.data);
    bb2.arg(&bias.data);
    bb2.arg(&b_i);
    bb2.arg(&c_out_i);
    bb2.arg(&l_out_i);
    bb2.arg(&1i32);
    bb2.arg(&0i32);
    unsafe { bb2.launch(cfg2) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c_out, l_out],
    })
}

/// Conv1d with kernel=8, stride=4, pad=2 (TEncLayer/TDecLayer conv).
pub fn conv1d_k8_s4p2(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    w: &GpuConv1dWeight,
    bias: &GpuBias,
) -> Result<GpuTensor> {
    let [b, c_in, l] = [x.shape[0], x.shape[1], x.shape[2]];
    if c_in != w.in_ch {
        return Err(anyhow!(
            "conv1d_k8_s4p2: c_in={} != w.in_ch={}",
            c_in,
            w.in_ch
        ));
    }
    if w.k != 8 {
        return Err(anyhow!("conv1d_k8_s4p2: w.k={} != 8", w.k));
    }
    let c_out = w.out_ch;
    let l_out = (l + 4 - 8) / 4 + 1;
    let patch = c_in * 8;
    let n_spatial = b * l_out;
    let mut col = state.alloc_uninit_f16(n_spatial * patch)?;
    let cfg = LaunchConfig {
        grid_dim: (n_spatial as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_in_i = c_in as i32;
    let l_i = l as i32;
    let l_out_i = l_out as i32;
    let mut bb = state.stream.launch_builder(&state.k.im2col_8_s4p2_1d);
    bb.arg(&mut col);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_in_i);
    bb.arg(&l_i);
    bb.arg(&l_out_i);
    unsafe { bb.launch(cfg) }?;
    let gemm_out = state.gemm_f16(
        &GpuTensor {
            data: col,
            shape: vec![b * l_out, patch],
        },
        &GpuTensor {
            data: w.data.clone(),
            shape: vec![patch, c_out],
        },
        b * l_out,
        c_out,
        patch,
    )?;
    let mut out = state.alloc_uninit_f16(b * c_out * l_out)?;
    let cfg2 = launch_1d(b * c_out * l_out);
    let c_out_i = c_out as i32;
    let mut bb2 = state.stream.launch_builder(&state.k.conv2d_postprocess);
    bb2.arg(&mut out);
    bb2.arg(&gemm_out.data);
    bb2.arg(&bias.data);
    bb2.arg(&b_i);
    bb2.arg(&c_out_i);
    bb2.arg(&l_out_i);
    bb2.arg(&1i32);
    bb2.arg(&0i32);
    unsafe { bb2.launch(cfg2) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c_out, l_out],
    })
}

/// Conv2d with kernel=[8,1], stride=[4,1], pad=[2,0] (HEncLayer/TEncLayer conv).
/// Input: `[B, C_in, H, W]`. Output: `[B, C_out, H_out, W_out]`.
pub fn conv2d_8x1_s4p2(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    w: &GpuConv2dWeight,
    bias: &GpuBias,
) -> Result<GpuTensor> {
    let [b, c_in, h, width] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    if c_in != w.in_ch || w.kh != 8 || w.kw != 1 {
        return Err(anyhow!(
            "conv2d_8x1_s4p2: shape mismatch (c_in={} w.in={}, kh={} kw={})",
            c_in,
            w.in_ch,
            w.kh,
            w.kw
        ));
    }
    let c_out = w.out_ch;
    let pad_h = 2usize;
    let pad_w = 0usize;
    let stride_h = 4usize;
    let stride_w = 1usize;
    let h_out = (h + 2 * pad_h - 8) / stride_h + 1;
    let w_out = (width + 2 * pad_w - 1) / stride_w + 1;
    let patch = c_in * 8; // = c_in * kh * kw
    let n_spatial = b * h_out * w_out;
    let mut col = state.alloc_uninit_f16(n_spatial * patch)?;
    let cfg = LaunchConfig {
        grid_dim: (n_spatial as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_in_i = c_in as i32;
    let h_i = h as i32;
    let width_i = width as i32;
    let spatial_per_batch_i = (h_out * w_out) as i32;
    let w_out_i = w_out as i32;
    let h_out_i = h_out as i32;
    let mut bb = state.stream.launch_builder(&state.k.im2col_8x1_s4p2);
    bb.arg(&mut col);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_in_i);
    bb.arg(&h_i);
    bb.arg(&width_i);
    bb.arg(&spatial_per_batch_i);
    bb.arg(&w_out_i);
    unsafe { bb.launch(cfg) }?;
    // GEMM
    let gemm_out = state.gemm_f16(
        &GpuTensor {
            data: col,
            shape: vec![b * h_out * w_out, patch],
        },
        &GpuTensor {
            data: w.data.clone(),
            shape: vec![patch, c_out],
        },
        b * h_out * w_out,
        c_out,
        patch,
    )?;
    // Bias + reshape [B*H_out*W_out, C_out] → [B, C_out, H_out, W_out]
    let mut out = state.alloc_uninit_f16(b * c_out * h_out * w_out)?;
    let cfg2 = launch_1d(b * c_out * h_out * w_out);
    let c_out_i = c_out as i32;
    let mut bb2 = state.stream.launch_builder(&state.k.conv2d_postprocess);
    bb2.arg(&mut out);
    bb2.arg(&gemm_out.data);
    bb2.arg(&bias.data);
    bb2.arg(&b_i);
    bb2.arg(&c_out_i);
    bb2.arg(&h_out_i);
    bb2.arg(&w_out_i);
    bb2.arg(&0i32); // apply_gelu=0
    unsafe { bb2.launch(cfg2) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c_out, h_out, w_out],
    })
}

/// Conv2d with kernel=[1,1] (HEnc rewrite, HDec rewrite).
pub fn conv2d_1x1(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    w: &GpuConv2dWeight,
    bias: &GpuBias,
) -> Result<GpuTensor> {
    let [b, c_in, h, width] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    if c_in != w.in_ch || w.kh != 1 || w.kw != 1 {
        return Err(anyhow!("conv2d_1x1: shape mismatch"));
    }
    let c_out = w.out_ch;
    // 1x1 conv = GEMM over channels. Transpose [B,C,H,W] → [B,H,W,C] so the
    // channel dim is contiguous (innermost), then GEMM. The gemm_out is
    // [B*H*W, C_out] row-major — we reshape it to [B, C_out, H, W] and add
    // bias via conv2d_postprocess (which does the correct NCHW reshape +
    // per-channel bias; the previous add_bias_inplace path indexed bias
    // incorrectly assuming NHWC layout, corrupting all but element 0).
    let x_nhwc = transpose_bchw_to_bhwc(state, x)?;
    let n_rows = b * h * width;
    let gemm_out = state.gemm_f16(
        &GpuTensor {
            data: x_nhwc.data.clone(),
            shape: vec![n_rows, c_in],
        },
        &GpuTensor {
            data: w.data.clone(),
            shape: vec![c_in, c_out],
        },
        n_rows,
        c_out,
        c_in,
    )?;
    let mut out = state.alloc_uninit_f16(b * c_out * h * width)?;
    let cfg2 = launch_1d(b * c_out * h * width);
    let b_i = b as i32;
    let c_out_i = c_out as i32;
    let h_i = h as i32;
    let w_i = width as i32;
    let mut bb2 = state.stream.launch_builder(&state.k.conv2d_postprocess);
    bb2.arg(&mut out);
    bb2.arg(&gemm_out.data);
    bb2.arg(&bias.data);
    bb2.arg(&b_i);
    bb2.arg(&c_out_i);
    bb2.arg(&h_i);
    bb2.arg(&w_i);
    bb2.arg(&0i32); // apply_gelu=0
    unsafe { bb2.launch(cfg2) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c_out, h, width],
    })
}

// ═══════════════════════════════════════════════════════════════════════
//  ConvTranspose ops (im2col + GEMM + bias + postprocess)
// ═══════════════════════════════════════════════════════════════════════

/// ConvTranspose2d with kernel=[8,1], stride=[4,1], pad=[2,0] (HDecLayer conv_tr).
/// Input: `[B, C_in, H_in, W_in]`. Output: `[B, C_out, H_out, W_out]`.
pub fn conv_transpose2d_8x1_s4p2(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    w: &crate::gpu_model::GpuConvTranspose2dWeight,
    bias: &GpuBias,
) -> Result<GpuTensor> {
    let [b, c_in, h_in, w_in] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    if c_in != w.in_ch || w.kh != 8 || w.kw != 1 {
        return Err(anyhow!(
            "conv_transpose2d_8x1_s4p2: shape mismatch (c_in={} w.in={}, kh={} kw={})",
            c_in,
            w.in_ch,
            w.kh,
            w.kw
        ));
    }
    let c_out = w.out_ch;
    let pad_h = 2i32;
    let pad_w = 0i32;
    let stride_h = 4i32;
    let stride_w = 1i32;
    let h_out = ((h_in as i32 - 1) * stride_h + 8 - 1 - 2 * pad_h + 1) as usize;
    let w_out = ((w_in as i32 - 1) * stride_w + 1 - 1 - 2 * pad_w + 1) as usize;
    let patch = c_in * 8; // = c_in * kh * kw
    let n_spatial = b * h_out * w_out;
    let mut col = state.alloc_uninit_f16(n_spatial * patch)?;
    let cfg = LaunchConfig {
        grid_dim: (n_spatial as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_in_i = c_in as i32;
    let h_in_i = h_in as i32;
    let w_in_i = w_in as i32;
    let h_out_i = h_out as i32;
    let w_out_i = w_out as i32;
    let mut bb = state.stream.launch_builder(&state.k.im2col_conv_transpose_8x1_s4p2);
    bb.arg(&mut col);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_in_i);
    bb.arg(&h_in_i);
    bb.arg(&w_in_i);
    bb.arg(&h_out_i);
    bb.arg(&w_out_i);
    bb.arg(&pad_h);
    bb.arg(&stride_h);
    unsafe { bb.launch(cfg) }?;
    // GEMM: out[b*h_out*w_out, c_out] = col[b*h_out*w_out, patch] @ w[patch, c_out]
    let gemm_out = state.gemm_f16(
        &GpuTensor {
            data: col,
            shape: vec![b * h_out * w_out, patch],
        },
        &GpuTensor {
            data: w.data.clone(),
            shape: vec![patch, c_out],
        },
        b * h_out * w_out,
        c_out,
        patch,
    )?;
    let mut out = state.alloc_uninit_f16(b * c_out * h_out * w_out)?;
    let cfg2 = launch_1d(b * c_out * h_out * w_out);
    let c_out_i = c_out as i32;
    let mut bb2 = state.stream.launch_builder(&state.k.conv2d_postprocess);
    bb2.arg(&mut out);
    bb2.arg(&gemm_out.data);
    bb2.arg(&bias.data);
    bb2.arg(&b_i);
    bb2.arg(&c_out_i);
    bb2.arg(&h_out_i);
    bb2.arg(&w_out_i);
    bb2.arg(&0i32); // apply_gelu=0
    unsafe { bb2.launch(cfg2) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c_out, h_out, w_out],
    })
}

/// ConvTranspose1d with kernel=8, stride=4, pad=2 (TDecLayer conv_tr).
pub fn conv_transpose1d_8_s4p2(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    w: &crate::gpu_model::GpuConvTranspose1dWeight,
    bias: &GpuBias,
) -> Result<GpuTensor> {
    let [b, c_in, l_in] = [x.shape[0], x.shape[1], x.shape[2]];
    if c_in != w.in_ch || w.k != 8 {
        return Err(anyhow!(
            "conv_transpose1d_8_s4p2: shape mismatch (c_in={} w.in={}, k={})",
            c_in,
            w.in_ch,
            w.k
        ));
    }
    let c_out = w.out_ch;
    let pad = 2i32;
    let stride = 4i32;
    let l_out = ((l_in as i32 - 1) * stride + 8 - 1 - 2 * pad + 1) as usize;
    let patch = c_in * 8;
    let n_spatial = b * l_out;
    let mut col = state.alloc_uninit_f16(n_spatial * patch)?;
    let cfg = LaunchConfig {
        grid_dim: (n_spatial as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_in_i = c_in as i32;
    let l_in_i = l_in as i32;
    let l_out_i = l_out as i32;
    let mut bb = state.stream.launch_builder(&state.k.im2col_conv_transpose_8_s4p2_1d);
    bb.arg(&mut col);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_in_i);
    bb.arg(&l_in_i);
    bb.arg(&l_out_i);
    bb.arg(&pad);
    bb.arg(&stride);
    unsafe { bb.launch(cfg) }?;
    let gemm_out = state.gemm_f16(
        &GpuTensor {
            data: col,
            shape: vec![b * l_out, patch],
        },
        &GpuTensor {
            data: w.data.clone(),
            shape: vec![patch, c_out],
        },
        b * l_out,
        c_out,
        patch,
    )?;
    let mut out = state.alloc_uninit_f16(b * c_out * l_out)?;
    let cfg2 = launch_1d(b * c_out * l_out);
    let c_out_i = c_out as i32;
    let mut bb2 = state.stream.launch_builder(&state.k.conv2d_postprocess);
    bb2.arg(&mut out);
    bb2.arg(&gemm_out.data);
    bb2.arg(&bias.data);
    bb2.arg(&b_i);
    bb2.arg(&c_out_i);
    bb2.arg(&l_out_i);
    bb2.arg(&1i32);
    bb2.arg(&0i32);
    unsafe { bb2.launch(cfg2) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c_out, l_out],
    })
}

/// Trim H2 (frequency dim): `x[b, c, fr, w] → out[b, c, fr_target, w]`.
/// Used by HDecLayer to drop the extra freq slots from ConvTranspose2d.
pub fn trim_h2(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    fr_target: usize,
) -> Result<GpuTensor> {
    let [b, c, fr, w] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    if fr_target > fr {
        return Err(anyhow!(
            "trim_h2: fr_target={} > fr={}",
            fr_target,
            fr
        ));
    }
    let mut out = state.alloc_uninit_f16(b * c * fr_target * w)?;
    let cfg = launch_1d(b * c * fr_target * w);
    let b_i = b as i32;
    let c_i = c as i32;
    let fr_i = fr as i32;
    let fr_target_i = fr_target as i32;
    let w_i = w as i32;
    let mut bb = state.stream.launch_builder(&state.k.trim_h2);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_i);
    bb.arg(&fr_i);
    bb.arg(&fr_target_i);
    bb.arg(&w_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c, fr_target, w],
    })
}

/// Trim L (time dim): `x[b, c, l] → out[b, c, l_target]`.
pub fn trim_l(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    l_target: usize,
) -> Result<GpuTensor> {
    let [b, c, l] = [x.shape[0], x.shape[1], x.shape[2]];
    if l_target > l {
        return Err(anyhow!("trim_l: l_target={} > l={}", l_target, l));
    }
    let mut out = state.alloc_uninit_f16(b * c * l_target)?;
    let cfg = launch_1d(b * c * l_target);
    let b_i = b as i32;
    let c_i = c as i32;
    let l_i = l as i32;
    let l_target_i = l_target as i32;
    let mut bb = state.stream.launch_builder(&state.k.trim_l);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_i);
    bb.arg(&l_i);
    bb.arg(&l_target_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c, l_target],
    })
}

/// Conv1d k=1 (linear, no spatial extent) — used by TEnc/TDec rewrite.
pub fn conv1d_1x1(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    w: &GpuConv1dWeight,
    bias: &GpuBias,
) -> Result<GpuTensor> {
    let [b, c_in, l] = [x.shape[0], x.shape[1], x.shape[2]];
    if c_in != w.in_ch || w.k != 1 {
        return Err(anyhow!("conv1d_1x1: shape mismatch (k={} c_in={})", w.k, c_in));
    }
    let c_out = w.out_ch;
    // 1x1 conv1d = per-position linear. [B,C,L] is NOT flat-[B*L, C] (C is the
    // middle dim), so transpose to [B,L,C] (channels-contiguous) first, GEMM,
    // transpose back. Mirrors conv2d_1x1's NHWC approach.
    let x_blc = swap_dims_12_3d(state, x)?; // [B, L, C_in]
    let n_rows = b * l;
    let gemm_out = state.gemm_f16(
        &GpuTensor { data: x_blc.data.clone(), shape: vec![n_rows, c_in] },
        &GpuTensor { data: w.data.clone(), shape: vec![c_in, c_out] },
        n_rows, c_out, c_in,
    )?;
    // gemm_out flat = [B, L, C_out] row-major → add bias (last dim) → swap back.
    let mut biased = gemm_out.data;
    let cfg = launch_1d(b * l * c_out);
    let outer = b * l;
    let outer_i = outer as i32;
    let c_out_i = c_out as i32;
    let mut bb = state.stream.launch_builder(&state.k.add_bias_inplace);
    bb.arg(&mut biased);
    bb.arg(&bias.data);
    bb.arg(&outer_i);
    bb.arg(&c_out_i);
    unsafe { bb.launch(cfg) }?;
    let blc = GpuTensor::new(biased, vec![b, l, c_out]);
    swap_dims_12_3d(state, &blc) // → [B, C_out, L]
}
pub fn conv2d_3x3_s1p1(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    w: &GpuConv2dWeight,
    bias: &GpuBias,
) -> Result<GpuTensor> {
    let [b, c_in, h, w_dim] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    if c_in != w.in_ch || w.kh != 3 || w.kw != 3 {
        return Err(anyhow!("conv2d_3x3_s1p1: shape mismatch"));
    }
    let c_out = w.out_ch;
    let h_out = h;
    let w_out = w_dim;
    let patch = c_in * 9;
    let n_spatial = b * h_out * w_out;
    let mut col = state.alloc_uninit_f16(n_spatial * patch)?;
    let cfg = LaunchConfig {
        grid_dim: (n_spatial as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let c_in_i = c_in as i32;
    let h_i = h as i32;
    let w_i = w_dim as i32;
    let h_out_i = h_out as i32;
    let w_out_i = w_out as i32;
    let mut bb = state.stream.launch_builder(&state.k.im2col_3x3_s1p1);
    bb.arg(&mut col);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_in_i);
    bb.arg(&h_i);
    bb.arg(&w_i);
    bb.arg(&h_out_i);
    bb.arg(&w_out_i);
    unsafe { bb.launch(cfg) }?;
    let gemm_out = state.gemm_f16(
        &GpuTensor {
            data: col,
            shape: vec![b * h_out * w_out, patch],
        },
        &GpuTensor {
            data: w.data.clone(),
            shape: vec![patch, c_out],
        },
        b * h_out * w_out,
        c_out,
        patch,
    )?;
    let mut out = state.alloc_uninit_f16(b * c_out * h_out * w_out)?;
    let cfg2 = launch_1d(b * c_out * h_out * w_out);
    let c_out_i = c_out as i32;
    let mut bb2 = state.stream.launch_builder(&state.k.conv2d_postprocess);
    bb2.arg(&mut out);
    bb2.arg(&gemm_out.data);
    bb2.arg(&bias.data);
    bb2.arg(&b_i);
    bb2.arg(&c_out_i);
    bb2.arg(&h_out_i);
    bb2.arg(&w_out_i);
    bb2.arg(&0i32);
    unsafe { bb2.launch(cfg2) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c_out, h_out, w_out],
    })
}

/// Transpose [B, C, H, W] → [B, H, W, C] (NHWC layout).
pub fn transpose_bchw_to_bhwc(
    state: &Arc<CudaState>,
    x: &GpuTensor,
) -> Result<GpuTensor> {
    let [b, c, h, w] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    let mut out = state.alloc_uninit_f16(b * c * h * w)?;
    let cfg = launch_1d(b * c * h * w);
    let b_i = b as i32;
    let c_i = c as i32;
    let h_i = h as i32;
    let w_i = w as i32;
    let mut bb = state.stream.launch_builder(&state.k.transpose_bchw_to_bhwc);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_i);
    bb.arg(&h_i);
    bb.arg(&w_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, h, w, c],
    })
}

/// Transpose [B, H, W, C] → [B, C, H, W].
pub fn transpose_bhwc_to_bchw(
    state: &Arc<CudaState>,
    x: &GpuTensor,
) -> Result<GpuTensor> {
    let [b, h, w, c] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    let mut out = state.alloc_uninit_f16(b * c * h * w)?;
    let cfg = launch_1d(b * c * h * w);
    let b_i = b as i32;
    let c_i = c as i32;
    let h_i = h as i32;
    let w_i = w as i32;
    let mut bb = state.stream.launch_builder(&state.k.transpose_bhwc_to_bchw);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_i);
    bb.arg(&h_i);
    bb.arg(&w_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c, h, w],
    })
}

/// Permute [B, S, D] → [B, h, S, d_head] for multi-head attention.
/// Memory layout becomes `[B*h, S, d_head]` row-major.
pub fn permute_bsd_to_bhsd(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    h: usize,
    d_head: usize,
) -> Result<GpuTensor> {
    let [b, s, d] = [x.shape[0], x.shape[1], x.shape[2]];
    if d != h * d_head {
        return Err(anyhow!(
            "permute_bsd_to_bhsd: d={} != h={}*d_head={}",
            d, h, d_head
        ));
    }
    let mut out = state.alloc_uninit_f16(b * s * d)?;
    let cfg = launch_1d(b * s * d);
    let b_i = b as i32;
    let s_i = s as i32;
    let d_i = d as i32;
    let h_i = h as i32;
    let d_head_i = d_head as i32;
    let mut bb = state.stream.launch_builder(&state.k.permute_bsd_to_bhsd);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&s_i);
    bb.arg(&d_i);
    bb.arg(&h_i);
    bb.arg(&d_head_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b * h, s, d_head],
    })
}

/// Permute [B*h, S, d_head] → [B*h, d_head, S] (per-head transpose).
/// Used to get K^T before the attention GEMM.
pub fn permute_bhsd_to_bhds(
    state: &Arc<CudaState>,
    x: &GpuTensor,
) -> Result<GpuTensor> {
    let [bh, s, d_head] = [x.shape[0], x.shape[1], x.shape[2]];
    let mut out = state.alloc_uninit_f16(bh * d_head * s)?;
    let cfg = launch_1d(bh * s * d_head);
    let bh_i = bh as i32;
    let s_i = s as i32;
    let d_head_i = d_head as i32;
    let mut bb = state.stream.launch_builder(&state.k.permute_bhsd_to_bhds);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&bh_i);
    bb.arg(&s_i);
    bb.arg(&d_head_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![bh, d_head, s],
    })
}

/// Copy a per-head subview of [B*h, S, d_head] into a separate [S, d_head] buffer.
/// `in_data` is the source buffer; `bh` is the head index to extract.
/// The result is written IN-PLACE to `out_data` (must be pre-allocated with
/// size ≥ S*d_head). NOTE: pass a `&mut CudaSlice` — do NOT pass a clone.
/// In cudarc 0.19 `CudaSlice::clone()` is a deep device-to-device copy, so a
/// clone would be a separate allocation that gets dropped (freed) when this
/// function returns, leaving the caller's original buffer untouched.
pub fn copy_per_head(
    state: &Arc<CudaState>,
    out_data: &mut CudaSlice<half::f16>,
    in_data: &CudaSlice<half::f16>,
    bh: usize,
    s: usize,
    d_head: usize,
) -> Result<()> {
    let total = s * d_head;
    if total == 0 {
        return Ok(());
    }
    let cfg = launch_1d(total);
    let bh_i = bh as i32;
    let s_i = s as i32;
    let d_head_i = d_head as i32;
    let mut bb = state.stream.launch_builder(&state.k.copy_per_head);
    bb.arg(out_data);
    bb.arg(in_data);
    bb.arg(&bh_i);
    bb.arg(&s_i);
    bb.arg(&d_head_i);
    unsafe { bb.launch(cfg) }?;
    Ok(())
}

/// Scatter a per-head [S, d_head] into the (bh)-th slot of [B*h, S, d_head].
pub fn scatter_per_head(
    state: &Arc<CudaState>,
    mut out_data: cudarc::driver::CudaViewMut<'_, half::f16>,
    in_data: CudaSlice<half::f16>,
    bh: usize,
    s: usize,
    d_head: usize,
) -> Result<()> {
    let total = s * d_head;
    if total == 0 {
        return Ok(());
    }
    let cfg = launch_1d(total);
    let bh_i = bh as i32;
    let s_i = s as i32;
    let d_head_i = d_head as i32;
    let mut bb = state.stream.launch_builder(&state.k.scatter_per_head);
    bb.arg(&mut out_data);
    bb.arg(&in_data);
    bb.arg(&bh_i);
    bb.arg(&s_i);
    bb.arg(&d_head_i);
    unsafe { bb.launch(cfg) }?;
    Ok(())
}
pub fn permute_bhsd_to_bsd(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    b: usize,
) -> Result<GpuTensor> {
    let [bh, s, d_head] = [x.shape[0], x.shape[1], x.shape[2]];
    if bh % b != 0 {
        return Err(anyhow!("permute_bhsd_to_bsd: bh={} not divisible by b={}", bh, b));
    }
    let h = bh / b;
    let d = h * d_head;
    let mut out = state.alloc_uninit_f16(b * s * d)?;
    let cfg = launch_1d(b * h * s * d_head);
    let b_i = b as i32;
    let s_i = s as i32;
    let h_i = h as i32;
    let d_head_i = d_head as i32;
    let mut bb = state.stream.launch_builder(&state.k.permute_bhsd_to_bsd);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&s_i);
    bb.arg(&h_i);
    bb.arg(&d_head_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, s, d],
    })
}

pub fn reshape_bcft_to_bfct(
    state: &Arc<CudaState>,
    x: &GpuTensor,
) -> Result<GpuTensor> {
    let [b, c, fr, t] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    let mut out = state.alloc_uninit_f16(b * c * fr * t)?;
    let cfg = launch_1d(b * c * fr * t);
    let b_i = b as i32;
    let c_i = c as i32;
    let fr_i = fr as i32;
    let t_i = t as i32;
    let mut bb = state.stream.launch_builder(&state.k.reshape_bcft_to_bfct);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_i);
    bb.arg(&fr_i);
    bb.arg(&t_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b * fr, c, t],
    })
}

/// Reshape [B*F, C, T] → [B, C, F, T] (hdec, post-dconv).
pub fn reshape_bfct_to_bcft(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    b: usize,
) -> Result<GpuTensor> {
    let [_bf, c, t] = [x.shape[0], x.shape[1], x.shape[2]];
    let fr = x.shape[0] / b;
    let mut out = state.alloc_uninit_f16(b * c * fr * t)?;
    let cfg = launch_1d(b * c * fr * t);
    let b_i = b as i32;
    let c_i = c as i32;
    let fr_i = fr as i32;
    let t_i = t as i32;
    let mut bb = state.stream.launch_builder(&state.k.reshape_bfct_to_bcft);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&c_i);
    bb.arg(&fr_i);
    bb.arg(&t_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: vec![b, c, fr, t],
    })
}

// ═══════════════════════════════════════════════════════════════════════
//  Multi-Head Attention (MHA)
// ═══════════════════════════════════════════════════════════════════════

use crate::gpu_model::GpuMhaWeights;

/// MHA forward. `q_in` `[B, S_q, D]`, `kv_in` `[B, S_k, D]` → `[B, S_q, D]`.
/// `kv_in == q_in` for self-attention; differ for cross-attention.
///
/// Pipeline (all primitives verified against CPU):
///   Q=linear(q), K=linear(kv), V=linear(kv)            [B, S, D]
///   permute_bsd_to_bhsd → [B*h, S, d_head]
///   K_t = permute_bhsd_to_bhds(K) → [B*h, d_head, S_k]
///   scores = batched GEMM Q @ K_t  → [B*h, S_q, S_k]
///   softmax(scores * scale) over S_k
///   attn = batched GEMM scores @ V → [B*h, S_q, d_head]
///   permute_bhsd_to_bsd → [B, S_q, D]
///   out = linear(attn)
pub fn mha(
    state: &Arc<CudaState>,
    q_in: &GpuTensor,
    kv_in: &GpuTensor,
    attn: &GpuMhaWeights,
) -> Result<GpuTensor> {
    let d = attn.d_model;
    let h = attn.n_heads;
    let d_head = d / h;
    assert!(d % h == 0, "mha: d={} not divisible by h={}", d, h);
    let [b, s_q, _qd] = [q_in.shape[0], q_in.shape[1], q_in.shape[2]];
    let [_kb, s_k, _kd] = [kv_in.shape[0], kv_in.shape[1], kv_in.shape[2]];
    let scale = 1.0f32 / (d_head as f32).sqrt();
    let bh = b * h;
    let enabled = std::env::var("DEMUCS_CUDA_PROFILE").map(|v| v == "1").unwrap_or(false);
    let mut pt = CudaPhaseTimer::new_aggregate(enabled);

    // 1. Project Q, K, V (linear_with_bias treats input as 2D [outer, in]).
    pt.start(state, "mha_qkv_proj");
    let q_proj = linear_with_bias(state, q_in, &attn.q_w, &attn.q_b)?;
    let k_proj = linear_with_bias(state, kv_in, &attn.k_w, &attn.k_b)?;
    let v_proj = linear_with_bias(state, kv_in, &attn.v_w, &attn.v_b)?;
    pt.end(state, "mha_qkv_proj");
    // Re-wrap 2D [b*s, d] as 3D [b, s, d] (same memory).
    let q3 = GpuTensor { data: q_proj.data, shape: vec![b, s_q, d] };
    let k3 = GpuTensor { data: k_proj.data, shape: vec![b, s_k, d] };
    let v3 = GpuTensor { data: v_proj.data, shape: vec![b, s_k, d] };

    // 2. Permute to per-head [B*h, S, d_head]; transpose K to [B*h, d_head, S_k].
    pt.start(state, "mha_permute");
    let q_h = permute_bsd_to_bhsd(state, &q3, h, d_head)?;     // [B*h, S_q, d_head]
    let k_h = permute_bsd_to_bhsd(state, &k3, h, d_head)?;     // [B*h, S_k, d_head]
    let v_h = permute_bsd_to_bhsd(state, &v3, h, d_head)?;     // [B*h, S_k, d_head]
    let k_t = permute_bhsd_to_bhds(state, &k_h)?;              // [B*h, d_head, S_k]
    pt.end(state, "mha_permute");

    // 3. scores = Q @ K_t : [B*h, S_q, d_head] @ [B*h, d_head, S_k] = [B*h, S_q, S_k].
    //    arg-swap: pass K_t (B-side) first, Q (A-side) second.
    pt.start(state, "mha_qkT_gemm");
    let scores = state.gemm_strided_batched_f16(
        &k_t.data, &q_h.data,
        sys::cublasOperation_t::CUBLAS_OP_N,
        sys::cublasOperation_t::CUBLAS_OP_N,
        bh,
        s_k,           // n  (K_t cols)
        s_q,           // m  (Q rows)
        d_head,        // k
        s_k,           // lda = n (K_t row-major cols)
        d_head,        // ldb = k (Q row-major cols)
        d_head * s_k,  // stride_a (K_t per batch)
        s_q * d_head,  // stride_b (Q per batch)
    )?;
    // Re-wrap as [B*h, S_q, S_k] (memory is row-major per batch).
    let scores = GpuTensor { data: scores.data, shape: vec![bh, s_q, s_k] };

    // 4. Softmax over S_k (the key dim) with 1/sqrt(d_head) scale.
    pt.start(state, "mha_softmax");
    let attn_w = softmax_scaled(state, &scores, scale)?;       // [B*h, S_q, S_k]
    pt.end(state, "mha_softmax");

    // 5. attn = attn_w @ V : [B*h, S_q, S_k] @ [B*h, S_k, d_head] = [B*h, S_q, d_head].
    //    arg-swap: pass V (B-side) first, attn_w (A-side) second.
    pt.start(state, "mha_attnV_gemm");
    let out_h = state.gemm_strided_batched_f16(
        &v_h.data, &attn_w.data,
        sys::cublasOperation_t::CUBLAS_OP_N,
        sys::cublasOperation_t::CUBLAS_OP_N,
        bh,
        d_head,        // n  (V cols)
        s_q,           // m  (attn rows)
        s_k,           // k
        d_head,        // lda = n (V row-major cols)
        s_k,           // ldb = k (attn row-major cols)
        s_k * d_head,  // stride_a (V per batch)
        s_q * s_k,     // stride_b (attn per batch)
    )?;
    pt.end(state, "mha_attnV_gemm");
    let out_h = GpuTensor { data: out_h.data, shape: vec![bh, s_q, d_head] };

    // 6. Permute back [B*h, S_q, d_head] → [B, S_q, D], then output projection.
    pt.start(state, "mha_out_perm+proj");
    let attn3 = permute_bhsd_to_bsd(state, &out_h, b)?;
    let out = linear_with_bias(state, &attn3, &attn.out_proj_w, &attn.out_proj_b)?;
    pt.end(state, "mha_out_perm+proj");
    pt.resolve();
    pt.print("mha");
    Ok(GpuTensor { data: out.data, shape: vec![b, s_q, d] })
}

// ═══════════════════════════════════════════════════════════════════════
//  Transformer layers (self + cross attention)
// ═══════════════════════════════════════════════════════════════════════

use crate::gpu_model::{GpuCrossAttnLayer, GpuSelfAttnLayer};

/// Self-attention transformer layer. Input/output `[B, S, D]`.
///
///   x_n = LN(x, norm1)
///   x = x + γ₁ · MHA_self(x_n)
///   x_n2 = LN(x, norm2)
///   x = x + γ₂ · Linear2(GELU(Linear1(x_n2)))
///   MyGroupNorm over (S, D): swap→[B,D,S]→GN1→swap back
pub fn self_attn_layer(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    layer: &GpuSelfAttnLayer,
) -> Result<GpuTensor> {
    let [b, s, d] = [x.shape[0], x.shape[1], x.shape[2]];
    let outer = b * s;
    let enabled = std::env::var("DEMUCS_CUDA_PROFILE").map(|v| v == "1").unwrap_or(false);
    let mut pt = CudaPhaseTimer::new_aggregate(enabled);

    // Block 1: self-attention. layer_norm returns 2D [b*s, d]; wrap as 3D for mha.
    pt.start(state, "sa_ln1");
    let x_n = layer_norm(state, x, &layer.norm1, outer, d)?;
    pt.end(state, "sa_ln1");
    let x_n3 = GpuTensor::new(x_n.data, vec![b, s, d]);
    pt.start(state, "sa_mha");
    let attn_out = mha(state, &x_n3, &x_n3, &layer.attn)?;
    pt.end(state, "sa_mha");
    pt.start(state, "sa_layer_scale+add1");
    let attn_scaled = layer_scale_last_inplace(state, attn_out, &layer.gamma_1)?;
    let mut x_after = add_to(state, x, &attn_scaled)?;
    pt.end(state, "sa_layer_scale+add1");

    // Block 2: FFN. linear_with_bias treats input as flat 2D.
    pt.start(state, "sa_ln2");
    let x_n2 = layer_norm(state, &x_after, &layer.norm2, outer, d)?;
    pt.end(state, "sa_ln2");
    pt.start(state, "sa_ffn");
    let h1 = linear_with_bias(state, &x_n2, &layer.linear1, &layer.linear1_bias)?;
    let h1g = gelu_inplace(state, h1)?;
    let h2 = linear_with_bias(state, &h1g, &layer.linear2, &layer.linear2_bias)?;
    pt.end(state, "sa_ffn");
    pt.start(state, "sa_layer_scale+add2");
    let h2_scaled = layer_scale_last_inplace(state, h2, &layer.gamma_2)?;
    x_after = add_to(state, &x_after, &h2_scaled)?;
    pt.end(state, "sa_layer_scale+add2");

    // Block 3: MyGroupNorm over (S, D) per batch.
    // [B,S,D] → swap dims 1,2 → [B,D,S] → groupnorm1 → swap back → [B,S,D].
    pt.start(state, "sa_groupnorm");
    let mut swapped = swap_dims_12_3d(state, &x_after)?; // [B,D,S]
    groupnorm1_inplace(state, &mut swapped, &layer.norm_out)?;
    let out = swap_dims_12_3d(state, &swapped)?; // back to [B,S,D]
    pt.end(state, "sa_groupnorm");
    pt.resolve();
    pt.print("self_attn");
    Ok(out)
}

/// Cross-attention transformer layer. `query` `[B, Sq, D]`, `cross` `[B, Sk, D]`.
///
///   q_n = LN(query, norm1); kv_n = LN(cross, norm2)
///   x = query + γ₁ · MHA(q_n, kv_n)
///   x_n2 = LN(x, norm3)
///   x = x + γ₂ · Linear2(GELU(Linear1(x_n2)))
///   MyGroupNorm over (S, D).
pub fn cross_attn_layer(
    state: &Arc<CudaState>,
    query: &GpuTensor,
    cross: &GpuTensor,
    layer: &GpuCrossAttnLayer,
) -> Result<GpuTensor> {
    let [b, sq, d] = [query.shape[0], query.shape[1], query.shape[2]];
    let outer = b * sq;
    let sk = cross.shape[1];
    let outer_k = cross.numel() / d;
    let enabled = std::env::var("DEMUCS_CUDA_PROFILE").map(|v| v == "1").unwrap_or(false);
    let mut pt = CudaPhaseTimer::new_aggregate(enabled);

    // Block 1: cross-attention. Wrap layer_norm 2D outputs as 3D for mha.
    pt.start(state, "ca_ln_q");
    let q_n = layer_norm(state, query, &layer.norm1, outer, d)?;
    pt.end(state, "ca_ln_q");
    let q_n3 = GpuTensor::new(q_n.data, vec![b, sq, d]);
    pt.start(state, "ca_ln_kv");
    let kv_n = layer_norm(state, cross, &layer.norm2, outer_k, d)?;
    pt.end(state, "ca_ln_kv");
    let kv_n3 = GpuTensor::new(kv_n.data, vec![b, sk, d]);
    pt.start(state, "ca_mha");
    let attn_out = mha(state, &q_n3, &kv_n3, &layer.attn)?;
    pt.end(state, "ca_mha");
    pt.start(state, "ca_layer_scale+add1");
    let attn_scaled = layer_scale_last_inplace(state, attn_out, &layer.gamma_1)?;
    let mut x_after = add_to(state, query, &attn_scaled)?;
    pt.end(state, "ca_layer_scale+add1");

    // Block 2: FFN (uses norm3).
    pt.start(state, "ca_ln3");
    let x_n2 = layer_norm(state, &x_after, &layer.norm3, outer, d)?;
    pt.end(state, "ca_ln3");
    pt.start(state, "ca_ffn");
    let h1 = linear_with_bias(state, &x_n2, &layer.linear1, &layer.linear1_bias)?;
    let h1g = gelu_inplace(state, h1)?;
    let h2 = linear_with_bias(state, &h1g, &layer.linear2, &layer.linear2_bias)?;
    pt.end(state, "ca_ffn");
    pt.start(state, "ca_layer_scale+add2");
    let h2_scaled = layer_scale_last_inplace(state, h2, &layer.gamma_2)?;
    x_after = add_to(state, &x_after, &h2_scaled)?;
    pt.end(state, "ca_layer_scale+add2");

    // Block 3: MyGroupNorm over (S, D).
    pt.start(state, "ca_groupnorm");
    let mut swapped = swap_dims_12_3d(state, &x_after)?;
    groupnorm1_inplace(state, &mut swapped, &layer.norm_out)?;
    let out = swap_dims_12_3d(state, &swapped)?;
    pt.end(state, "ca_groupnorm");
    pt.resolve();
    pt.print("cross_attn");
    Ok(out)
}

/// Wraps the `softmax_scaled_f16` kernel. Input is 3D `[B, S, N]`
/// (we use `B*h` as the batch dim for MHA scores). Allocates a fresh output
/// (the kernel can't safely run in-place because it reads from `in` to compute
/// the row max before writing to `out`).
///
/// Kernel uses 8 rows/block (256 threads, 32-thread warps) with warp-shuffle
/// reduction — no shared memory, no __syncthreads barriers.
pub fn softmax_scaled(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    scale: f32,
) -> Result<GpuTensor> {
    let [b, s, n] = [x.shape[0], x.shape[1], x.shape[2]];
    let total = b * s * n;
    if total == 0 {
        return Ok(GpuTensor {
            data: x.data.clone(),
            shape: x.shape.clone(),
        });
    }
    let total_rows = b * s;
    const ROWS_PER_BLOCK: u32 = 8;
    let grid = ((total_rows as u32) + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    let mut out = state.alloc_uninit_f16(total)?;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1), // 8 warps × 32 threads
        shared_mem_bytes: 0,
    };
    let b_i = b as i32;
    let s_i = s as i32;
    let n_i = n as i32;
    let mut bb = state.stream.launch_builder(&state.k.softmax_scaled);
    bb.arg(&mut out);
    bb.arg(&x.data);
    bb.arg(&b_i);
    bb.arg(&s_i);
    bb.arg(&n_i);
    bb.arg(&scale);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out,
        shape: x.shape.clone(),
    })
}

/// Linear: `out = x @ w + bias` for `x` `[outer, in]` and `w` `[in, out]`
/// row-major. Note: `GpuWeight2D.data` is already in this `[in, out]`
/// layout (transposed from CPU's `[out, in]`). Bias is broadcast across the
/// outer dim. Output shape: `[outer, out]`.
pub fn linear_with_bias(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    w: &GpuWeight2D,
    bias: &GpuBias,
) -> Result<GpuTensor> {
    let in_dim = w.rows;
    let out_dim = w.cols;
    if x.numel() % in_dim != 0 {
        return Err(anyhow!("linear: x.numel()={} not divisible by in={}", x.numel(), in_dim));
    }
    let outer = x.numel() / in_dim;
    // data is [in_dim, out_dim] row-major; gemm_f16 expects B[k, n].
    let gemm_out = state.gemm_f16(x, &GpuTensor::new(w.data.clone(), vec![in_dim, out_dim]), outer, out_dim, in_dim)?;
    // Add bias in-place (kernel add_bias_inplace_f16).
    let mut out_data = gemm_out.data;
    let cfg = LaunchConfig {
        grid_dim: (((outer * out_dim) as u32 + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let outer_i = outer as i32;
    let out_dim_i = out_dim as i32;
    let mut bb = state.stream.launch_builder(&state.k.add_bias_inplace);
    bb.arg(&mut out_data);
    bb.arg(&bias.data);
    bb.arg(&outer_i);
    bb.arg(&out_dim_i);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor {
        data: out_data,
        shape: vec![outer, out_dim],
    })
}

// ═══════════════════════════════════════════════════════════════════════
//  Layer-level forward functions
// ═══════════════════════════════════════════════════════════════════════

use crate::gpu_model::{
    GpuDConv, GpuDConvLayer, GpuHDecLayer, GpuHEncLayer, GpuTDecLayer, GpuTEncLayer,
};

/// One DConvLayer forward (matches `ops_cpu::dconv_layer_forward`).
///
/// Input/output: `[B, C, L]` row-major.
///   residual = x
///   x = conv1d_k3_dilation(x, conv1, conv1_bias, dilation)  → [B, compress, L]
///   x = groupnorm1(x, norm1)
///   x = gelu(x)
///   x = conv1d_k1(x, conv2, conv2_bias)                       → [B, 2*C, L]
///   x = groupnorm1(x, norm2)
///   x = glu_channel(x)                                       → [B, C, L]
///   x = layer_scale(x, scale)
///   return x + residual
pub fn dconv_layer(
    state: &Arc<CudaState>,
    x: GpuTensor,
    layer: &GpuDConvLayer,
    dilation: usize,
) -> Result<GpuTensor> {
    let [b, c, l] = [x.shape[0], x.shape[1], x.shape[2]];
    let residual = x;

    // conv1
    let mut h = conv1d_k3_dilation(state, &residual, &layer.conv1, &layer.conv1_bias, dilation)?;
    // groupnorm1
    groupnorm1_inplace(state, &mut h, &layer.norm1)?;
    // gelu
    h = gelu_inplace(state, h)?;
    // conv2 (k=1)
    let mut h2 = conv1d_k1(state, &h, &layer.conv2, &layer.conv2_bias)?;
    // groupnorm1
    groupnorm1_inplace(state, &mut h2, &layer.norm2)?;
    // glu
    let mut h3 = glu_channel(state, &h2)?;
    // layer_scale + residual add.
    // h3 = h3 * scale + residual
    let h3 = layer_scale_inplace(state, h3, &layer.scale)?;
    // add_to allocates a new buffer for the result.
    let out = add_to(state, &h3, &residual)?;
    Ok(out)
}

/// 2-layer DConv forward (dilation 1 then 2).
pub fn dconv(state: &Arc<CudaState>, x: GpuTensor, dconv: &GpuDConv) -> Result<GpuTensor> {
    let mut data = x;
    for (j, layer) in dconv.layers.iter().enumerate() {
        let dilation = 1 << j;
        data = dconv_layer(state, data, layer, dilation)?;
    }
    Ok(data)
}

/// HEncLayer forward.
///
/// Input:  x [B, C_in, Fr, T]
/// Output: [B, C_out, Fr/4, T]
///
/// Forward:
///   x = conv2d_8x1_s4p2(x, conv, conv_bias) → [B, C_out, Fr/4, T]
///   x = gelu(x)
///   reshape → [B*Fr_out, C_out, T]
///   x = dconv(x)
///   reshape back → [B, C_out, Fr_out, T]
///   x = conv2d_1x1(x, rewrite, rewrite_bias)  → [B, 2*C_out, Fr_out, T]
///   x = glu_channel(x)                         → [B, C_out, Fr_out, T]
pub fn henc_layer(
    state: &Arc<CudaState>,
    x: GpuTensor,
    layer: &GpuHEncLayer,
) -> Result<GpuTensor> {
    let enabled = std::env::var("DEMUCS_CUDA_PROFILE").map(|v| v == "1").unwrap_or(false);
    let mut pt = CudaPhaseTimer::new_aggregate(enabled);
    let [b, _c_in, _fr, _t] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    // 1. Conv2d [8,1] stride [4,1] pad [2,0]
    pt.start(state, "henc_conv2d_8x1");
    let h = conv2d_8x1_s4p2(state, &x, &layer.conv, &layer.conv_bias)?;
    pt.end(state, "henc_conv2d_8x1");
    // 2. GELU
    pt.start(state, "henc_gelu");
    let mut h = gelu_inplace(state, h)?;
    pt.end(state, "henc_gelu");
    let [_, c_out, fr_out, t] = [h.shape[0], h.shape[1], h.shape[2], h.shape[3]];
    // 3. Reshape [B, C_out, Fr_out, T] → [B*Fr_out, C_out, T] (true transpose).
    pt.start(state, "henc_reshape+dconv");
    h = reshape_bcft_to_bfct(state, &h)?;
    // 4. DConv — operates on [B*Fr_out, C_out, T].
    let dconv_out = dconv(state, h, &layer.dconv)?;
    let [n, c2, t2] = [dconv_out.shape[0], dconv_out.shape[1], dconv_out.shape[2]];
    assert_eq!(n, b * fr_out, "henc dconv out n mismatch");
    assert_eq!(c2, c_out, "henc dconv out c mismatch");
    // 5. Reshape [B*Fr_out, C_out, T] → [B, C_out, Fr_out, T] for the 1x1 conv.
    let unflat = reshape_bfct_to_bcft(state, &dconv_out, b)?;
    pt.end(state, "henc_reshape+dconv");
    // 6. Conv2d [1,1] → [B, 2*C_out, Fr_out, T]
    pt.start(state, "henc_conv2d_1x1");
    let rewritten = conv2d_1x1(state, &unflat, &layer.rewrite, &layer.rewrite_bias)?;
    pt.end(state, "henc_conv2d_1x1");
    // 7. GLU on dim=1: rewritten is 4D [B, 2C, Fr, T]. We need to treat it as
    //    [B, 2C, Fr*T] (channel-flat) for glu_channel since the kernel expects
    //    [B, 2C, L]. The row-major memory layout is the same so we can just
    //    wrap with a different shape.
    pt.start(state, "henc_glu");
    let rewritten_flat = GpuTensor {
        data: rewritten.data,
        shape: vec![rewritten.shape[0], rewritten.shape[1], rewritten.shape[2] * rewritten.shape[3]],
    };
    let out_flat = glu_channel(state, &rewritten_flat)?;
    pt.end(state, "henc_glu");
    pt.resolve();
    pt.print("henc");
    // Reshape back to 4D for downstream.
    Ok(GpuTensor {
        data: out_flat.data,
        shape: vec![out_flat.shape[0], out_flat.shape[1], fr_out, t],
    })
}

/// HDecLayer forward.
///
/// Inputs:
///   - x [B, C_in, Fr, T]
///   - skip [B, C_in, Fr, T]
///   - freq_target: trim output freq dim to this
///
/// Output: [B, C_in, freq_target, T].
pub fn hdec_layer(
    state: &Arc<CudaState>,
    x: GpuTensor,
    skip: &GpuTensor,
    freq_target: usize,
    layer: &GpuHDecLayer,
) -> Result<GpuTensor> {
    let enabled = std::env::var("DEMUCS_CUDA_PROFILE").map(|v| v == "1").unwrap_or(false);
    let mut pt = CudaPhaseTimer::new_aggregate(enabled);
    let [b, chin, fr, t] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    // 1. Residual: h = x + skip
    pt.start(state, "hdec_add_skip");
    let mut h = add_to(state, &x, skip)?;
    pt.end(state, "hdec_add_skip");
    // 2. Conv2d(3,3) pad=(1,1)
    pt.start(state, "hdec_conv2d_3x3");
    let h2 = conv2d_3x3_s1p1(state, &h, &layer.rewrite, &layer.rewrite_bias)?;
    pt.end(state, "hdec_conv2d_3x3");
    // 3. GLU on dim=1: 2*chin → chin. The `glu_channel` kernel only accepts a
    //    3D [B, 2C, L] view, so flatten the trailing (Fr, T) dims and rewrap
    //    the result as 4D. Memory layout is row-major so the kernel sees the
    //    same data.
    pt.start(state, "hdec_glu");
    let h2_flat = GpuTensor {
        data: h2.data,
        shape: vec![h2.shape[0], h2.shape[1], h2.shape[2] * h2.shape[3]],
    };
    let h3_flat = glu_channel(state, &h2_flat)?;
    let c3 = h2.shape[1] / 2;
    let h3 = GpuTensor {
        data: h3_flat.data,
        shape: vec![b, c3, fr, t],
    };
    pt.end(state, "hdec_glu");
    let [_b3, c3, fr3, t3] = [h3.shape[0], h3.shape[1], h3.shape[2], h3.shape[3]];
    // 4. DConv (per-frequency flatten). The dconv kernels are 3D-only
    //    ([B*Fr, C, T]), so reshape the 4D input down, then unflatten the
    //    output back to 4D for the conv-transpose. Mirrors the henc_layer
    //    pattern at lines 1772-1779.
    pt.start(state, "hdec_reshape+dconv");
    let h3_flat_for_dconv = reshape_bcft_to_bfct(state, &h3)?;
    let dconv_out = dconv(state, h3_flat_for_dconv, &layer.dconv)?;
    // dconv_out shape is [B*Fr, C, T] — pass directly to reshape_bfct_to_bcft
    // which uses shape[0]/b to recover Fr. Don't rewrap with a 4D shape —
    // that misleads reshape into thinking Fr is small and collapses T into Fr.
    let unflat = reshape_bfct_to_bcft(state, &dconv_out, b)?;
    pt.end(state, "hdec_reshape+dconv");
    // 5. ConvTranspose2d([8,1], stride=[4,1], pad=[2,0])
    pt.start(state, "hdec_convtr_8x1");
    let h5 = conv_transpose2d_8x1_s4p2(state, &unflat, &layer.conv_tr, &layer.conv_tr_bias)?;
    pt.end(state, "hdec_convtr_8x1");
    // 6. Trim freq dim if > freq_target
    let [_, _, h5_fr, _] = [h5.shape[0], h5.shape[1], h5.shape[2], h5.shape[3]];
    let h5_trimmed = if h5_fr > freq_target {
        pt.start(state, "hdec_trim");
        let r = trim_h2(state, &h5, freq_target)?;
        pt.end(state, "hdec_trim");
        r
    } else {
        h5
    };
    // 7. Optional GELU
    if !layer.last {
        pt.start(state, "hdec_gelu");
        let r = gelu_inplace(state, h5_trimmed)?;
        pt.end(state, "hdec_gelu");
        pt.resolve();
        pt.print("hdec");
        Ok(r)
    } else {
        pt.resolve();
        pt.print("hdec");
        Ok(h5_trimmed)
    }
}

/// TEncLayer forward.
///
/// Input:  x [B, C_in, T]
/// Output: [B, C_out, T/4]
pub fn tenc_layer(
    state: &Arc<CudaState>,
    x: GpuTensor,
    layer: &GpuTEncLayer,
) -> Result<GpuTensor> {
    let enabled = std::env::var("DEMUCS_CUDA_PROFILE").map(|v| v == "1").unwrap_or(false);
    let mut pt = CudaPhaseTimer::new_aggregate(enabled);
    let [b, c_in, t] = [x.shape[0], x.shape[1], x.shape[2]];
    let stride = 4;
    // 1. Right-pad so length is divisible by STRIDE. CPU does this
    //    (ops_cpu.rs tenc_layer_forward line ~1843); without it the chain
    //    misses samples at every stride-4 boundary (85995 → 21499 expected
    //    becomes 21498, etc.) and the final TDec time_out comes up short
    //    of padded_len, which trips the extract_stems assertion.
    let pad_right = if t % stride == 0 { 0 } else { stride - (t % stride) };
    let x_padded = if pad_right > 0 {
        // Build a [B, C_in, t + pad_right] tensor with zeros on the right.
        let t_padded = t + pad_right;
        let mut padded = state.alloc_uninit_f16(b * c_in * t_padded)?;
        // Copy x into the left part.
        let cfg = launch_1d(b * c_in * t_padded);
        let b_i = b as i32;
        let c_in_i = c_in as i32;
        let t_i = t as i32;
        let t_padded_i = t_padded as i32;
        let pad_right_i = pad_right as i32;
        let mut bb = state.stream.launch_builder(&state.k.zero_pad_right);
        bb.arg(&mut padded);
        bb.arg(&x.data);
        bb.arg(&b_i);
        bb.arg(&c_in_i);
        bb.arg(&t_i);
        bb.arg(&t_padded_i);
        bb.arg(&pad_right_i);
        unsafe { bb.launch(cfg) }?;
        GpuTensor {
            data: padded,
            shape: vec![b, c_in, t_padded],
        }
    } else {
        x
    };
    // 2. Conv1d [8] s=4 p=2
    pt.start(state, "tenc_conv1d_8s4");
    let mut h = conv1d_k8_s4p2(state, &x_padded, &layer.conv, &layer.conv_bias)?;
    pt.end(state, "tenc_conv1d_8s4");
    // 3. GELU
    pt.start(state, "tenc_gelu");
    h = gelu_inplace(state, h)?;
    pt.end(state, "tenc_gelu");
    // 4. DConv (no reshape needed; conv1d preserves [B, C, T])
    pt.start(state, "tenc_dconv");
    let dconv_out = dconv(state, h, &layer.dconv)?;
    pt.end(state, "tenc_dconv");
    // 5. Conv1d [1] (rewrite 1x1) — treats input as [b*t, c_in] for the gemm.
    pt.start(state, "tenc_conv1d_1x1");
    let rewritten = conv1d_1x1(state, &dconv_out, &layer.rewrite, &layer.rewrite_bias)?;
    pt.end(state, "tenc_conv1d_1x1");
    // 6. GLU on dim=1
    pt.start(state, "tenc_glu");
    let out = glu_channel(state, &rewritten)?;
    pt.end(state, "tenc_glu");
    pt.resolve();
    pt.print("tenc");
    Ok(out)
}

/// TDecLayer forward.
///
/// Inputs:
///   - x [B, C_in, T]
///   - skip [B, C_in, T]
///   - time_target: trim output time dim to this
///
/// Output: [B, C_in, time_target].
pub fn tdec_layer(
    state: &Arc<CudaState>,
    x: GpuTensor,
    skip: &GpuTensor,
    time_target: usize,
    layer: &GpuTDecLayer,
) -> Result<GpuTensor> {
    let enabled = std::env::var("DEMUCS_CUDA_PROFILE").map(|v| v == "1").unwrap_or(false);
    let mut pt = CudaPhaseTimer::new_aggregate(enabled);
    let [b, chin, t] = [x.shape[0], x.shape[1], x.shape[2]];
    let skip_t = skip.shape[2];
    // 1. Residual: h = x + skip, but trim skip to min(skip_t, t). The CPU
    //    tdec_layer_forward (ops_cpu.rs:1750) does this implicitly; without
    //    it, the natural conv-transpose output length can be 1-3 shorter
    //    than the corresponding encoder output (e.g. with odd time dim from
    //    stride-4 convs) and add_to would panic.
    let skip_trimmed = if skip_t > t {
        pt.start(state, "tdec_trim_skip");
        let r = trim_l(state, skip, t)?;
        pt.end(state, "tdec_trim_skip");
        r
    } else {
        skip.clone_shallow()
    };
    pt.start(state, "tdec_add_skip");
    let h = add_to(state, &x, &skip_trimmed)?;
    pt.end(state, "tdec_add_skip");
    // 2. TDec rewrite is Conv1d k=3 pad=1 (length-preserving). Must use
    //    conv1d_k3_dilation (dilation=1) — not conv1d_k1. The k=1 path
    //    silently changes a 3-tap conv into a 1-tap conv, which produces
    //    wildly different values (and the comment claiming "k=1" was
    //    wrong; see model.rs:201 `rewrite: [2*chin, chin, 3]`).
    pt.start(state, "tdec_conv1d_k3");
    let h2 = conv1d_k3_dilation(state, &h, &layer.rewrite, &layer.rewrite_bias, 1)?;
    pt.end(state, "tdec_conv1d_k3");
    // 3. GLU on dim=1: 2*chin → chin
    pt.start(state, "tdec_glu");
    let h3 = glu_channel(state, &h2)?;
    pt.end(state, "tdec_glu");
    // 4. DConv
    pt.start(state, "tdec_dconv");
    let dconv_out = dconv(state, h3, &layer.dconv)?;
    pt.end(state, "tdec_dconv");
    // 5. ConvTranspose1d k=8 s=4 p=2
    pt.start(state, "tdec_convtr_8s4");
    let h5 = conv_transpose1d_8_s4p2(state, &dconv_out, &layer.conv_tr, &layer.conv_tr_bias)?;
    pt.end(state, "tdec_convtr_8s4");
    // 6. Trim time dim if > time_target
    let h5_trimmed = if h5.shape[2] > time_target {
        pt.start(state, "tdec_trim");
        let r = trim_l(state, &h5, time_target)?;
        pt.end(state, "tdec_trim");
        r
    } else {
        h5
    };
    // 7. Optional GELU
    if !layer.last {
        pt.start(state, "tdec_gelu");
        let r = gelu_inplace(state, h5_trimmed)?;
        pt.end(state, "tdec_gelu");
        pt.resolve();
        pt.print("tdec");
        Ok(r)
    } else {
        pt.resolve();
        pt.print("tdec");
        Ok(h5_trimmed)
    }
}
// ═══════════════════════════════════════════════════════════════════════
//  Cross-domain transformer
// ═══════════════════════════════════════════════════════════════════════

use crate::gpu_model::GpuCrossDomainTransformer;

/// Flatten [B, C, F, T] → [B, T*F, C] (time-major).
pub fn flatten_bcft_to_btfc(state: &Arc<CudaState>, x: &GpuTensor) -> Result<GpuTensor> {
    let [b, c, f, t] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    let mut out = state.alloc_uninit_f16(b * c * f * t)?;
    let cfg = launch_1d(b * c * f * t);
    let (bi, ci, fi, ti) = (b as i32, c as i32, f as i32, t as i32);
    let mut bb = state.stream.launch_builder(&state.k.flatten_bcft_to_btfc);
    bb.arg(&mut out); bb.arg(&x.data);
    bb.arg(&bi); bb.arg(&ci); bb.arg(&fi); bb.arg(&ti);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor { data: out, shape: vec![b, t * f, c] })
}

/// Inverse: [B, T*F, C] → [B, C, F, T].
pub fn unflatten_btfc_to_bcft(state: &Arc<CudaState>, x: &GpuTensor, f: usize, t: usize) -> Result<GpuTensor> {
    let [b, _tf, c] = [x.shape[0], x.shape[1], x.shape[2]];
    let mut out = state.alloc_uninit_f16(b * c * f * t)?;
    let cfg = launch_1d(b * c * f * t);
    let (bi, ci, fi, ti) = (b as i32, c as i32, f as i32, t as i32);
    let mut bb = state.stream.launch_builder(&state.k.unflatten_btfc_to_bcft);
    bb.arg(&mut out); bb.arg(&x.data);
    bb.arg(&bi); bb.arg(&ci); bb.arg(&fi); bb.arg(&ti);
    unsafe { bb.launch(cfg) }?;
    Ok(GpuTensor { data: out, shape: vec![b, c, f, t] })
}

/// Channel up/down sample 4D: [B, Cin, Fr, T] → [B, Cout, Fr, T] via a
/// Conv1d k=1 applied per (b, fr). Reshape to [B*Fr, Cin, T], conv1d_1x1,
/// reshape back. `w` is a [Cout, Cin, 1] Conv1dWeight.
pub fn channel_resample_4d(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    w: &crate::gpu_model::GpuConv1dWeight,
    bias: &GpuBias,
) -> Result<GpuTensor> {
    let [b, cin, fr, t] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    // [B, Cin, Fr, T] → [B*Fr, Cin, T]
    let flat = reshape_bcft_to_bfct(state, x)?;
    let up = conv1d_1x1(state, &flat, w, bias)?; // [B*Fr, Cout, T]
    // [B*Fr, Cout, T] → [B, Cout, Fr, T]
    reshape_bfct_to_bcft(state, &up, b)
}

/// Channel resample 3D: [B, Cin, T] → [B, Cout, T] (just conv1d_1x1).
pub fn channel_resample_3d(
    state: &Arc<CudaState>,
    x: &GpuTensor,
    w: &crate::gpu_model::GpuConv1dWeight,
    bias: &GpuBias,
) -> Result<GpuTensor> {
    conv1d_1x1(state, x, w, bias)
}

/// Cross-domain transformer forward.
///
/// `freq` [1, ch, Fr, T], `time` [1, ch, T2] → ([1, ch, Fr, T], [1, ch, T2]).
/// `freq_pe`/`time_pe` are precomputed sinusoidal embeds uploaded by the caller
/// (freq_pe [T*Fr, d_model], time_pe [T2, d_model]).
pub fn cross_domain_transformer(
    state: &Arc<CudaState>,
    freq: &GpuTensor,
    time: &GpuTensor,
    ct: &GpuCrossDomainTransformer,
    freq_pe: &GpuTensor,
    time_pe: &GpuTensor,
) -> Result<(GpuTensor, GpuTensor)> {
    let d_model = ct.norm_in.dim;
    let [_, _ch, fr, t] = [freq.shape[0], freq.shape[1], freq.shape[2], freq.shape[3]];
    let [_tb, _tc, t2] = [time.shape[0], time.shape[1], time.shape[2]];

    // 1. Channel upsample (4D freq, 3D time).
    let freq_d = match (&ct.channel_upsampler, &ct.channel_upsampler_bias) {
        (Some(w), Some(b)) => channel_resample_4d(state, freq, w, b)?,
        _ => freq.clone_shallow(),
    };
    let time_d = match (&ct.channel_upsampler_t, &ct.channel_upsampler_t_bias) {
        (Some(w), Some(b)) => channel_resample_3d(state, time, w, b)?,
        _ => time.clone_shallow(),
    };

    // 2. Flatten freq → [1, T*Fr, d_model]; permute time → [1, T2, d_model].
    let freq_seq = flatten_bcft_to_btfc(state, &freq_d)?;
    let time_seq = swap_dims_12_3d(state, &time_d)?;

    // 3. Input norms.
    let freq_n = layer_norm(state, &freq_seq, &ct.norm_in, t * fr, d_model)?;
    let time_n = layer_norm(state, &time_seq, &ct.norm_in_t, t2, d_model)?;

    // 4. Add positional embeds (in-place). freq_pe is [T*Fr, d], freq_n is [1, T*Fr, d].
    let mut freq_n = GpuTensor::new(freq_n.data, vec![1, t * fr, d_model]);
    add_inplace(state, &mut freq_n, freq_pe)?;
    let mut time_n = GpuTensor::new(time_n.data, vec![1, t2, d_model]);
    add_inplace(state, &mut time_n, time_pe)?;
    let mut freq_seq = freq_n;
    let mut time_seq = time_n;

    // 5. Transformer layers. CPU uses ct.layers[i] for freq and ct.layers_t[i]
    //    for time — separate weights, same type per index. Iterate in parallel.
    for (fl, tl) in ct.layers.iter().zip(ct.layers_t.iter()) {
        match (fl, tl) {
            (crate::gpu_model::GpuTransformerLayerWeights::SelfAttn(fl), crate::gpu_model::GpuTransformerLayerWeights::SelfAttn(tl)) => {
                freq_seq = self_attn_layer(state, &freq_seq, fl)?;
                time_seq = self_attn_layer(state, &time_seq, tl)?;
            }
            (crate::gpu_model::GpuTransformerLayerWeights::CrossAttn(fl), crate::gpu_model::GpuTransformerLayerWeights::CrossAttn(tl)) => {
                let f = cross_attn_layer(state, &freq_seq, &time_seq, fl)?;
                let ti = cross_attn_layer(state, &time_seq, &freq_seq, tl)?;
                freq_seq = f;
                time_seq = ti;
            }
            _ => return Err(anyhow!("cross_domain_transformer: freq/time layer type mismatch")),
        }
    }

    // 6. Unflatten freq → [1, d_model, Fr, T]; permute time → [1, d_model, T2].
    let freq_unflat = unflatten_btfc_to_bcft(state, &freq_seq, fr, t)?;
    let time_unflat = swap_dims_12_3d(state, &time_seq)?;

    // 7. Channel downsample.
    let freq_out = match (&ct.channel_downsampler, &ct.channel_downsampler_bias) {
        (Some(w), Some(b)) => channel_resample_4d(state, &freq_unflat, w, b)?,
        _ => freq_unflat,
    };
    let time_out = match (&ct.channel_downsampler_t, &ct.channel_downsampler_t_bias) {
        (Some(w), Some(b)) => channel_resample_3d(state, &time_unflat, w, b)?,
        _ => time_unflat,
    };
    Ok((freq_out, time_out))
}

impl GpuTensor {
    /// Re-wrap with the same data, different shape (same numel). Used when an
    /// op returns 2D but the caller knows the logical 3D shape.
    fn clone_shallow(&self) -> GpuTensor {
        // CudaSlice::clone is a deep D2D copy in cudarc 0.19; we accept the
        // copy here since channel samplers are optional and rare.
        GpuTensor { data: self.data.clone(), shape: self.shape.clone() }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  HTDemucs top-level forward (encoders → freq_emb → transformer → decoders)
// ═══════════════════════════════════════════════════════════════════════

use crate::gpu_model::GpuHTDemucs;

/// Download a tensor and log max abs. Diagnostic helper — re-enable the
/// `log_mag("label", state, &t)?` calls inside henc_layer/hdec_layer when
/// investigating the freq-path scale drift (see ROADMAP §11.4).
#[allow(dead_code)]
fn log_mag(label: &str, state: &Arc<CudaState>, t: &GpuTensor) -> Result<()> {
    let host = state.download_to_f32(t)?;
    let max = host.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("GMAG {label}: shape={:?} max={:.4}", t.shape(), max);
    Ok(())
}

/// HTDemucs forward on GPU. Inputs are ALREADY normalized (caller does
/// normalize_freq/normalize_time on CPU and uploads mean/std for denorm).
/// Returns the (still-normalized-space) freq/time outputs; caller denormalizes.
///
/// `freq` [1, 4, Fr, T], `time` [1, 2, T]. freq_mean/std [1], time_mean/std [1].
pub fn htdemucs_forward(
    state: &Arc<CudaState>,
    freq: &GpuTensor,
    time: &GpuTensor,
    model: &GpuHTDemucs,
) -> Result<(GpuTensor, GpuTensor)> {
    let depth = model.encoders.len();
    // freq_emb scale = 2.0 (ScaledEmbedding ×10 baked into weight, ×0.2 in
    // HTDemucs forward → net ×2.0; see ROADMAP §11.4 / commit 415b722).
    const FREQ_EMB_SCALE: f32 = 2.0;

    let mut pt = CudaPhaseTimer::new(
        std::env::var("DEMUCS_CUDA_PROFILE").map(|v| v == "1").unwrap_or(false)
    );

    // ─── 1. Freq encoder chain ───────────────────────────────────────
    pt.start(state, "01_freq_encoder");
    let mut freq_skips: Vec<GpuTensor> = Vec::with_capacity(depth);
    let mut h = henc_layer(state, freq.clone_shallow(), &model.encoders[0])?;
    // freq_emb after encoder[0].
    add_freq_emb_inplace(state, &mut h, &model.freq_emb, FREQ_EMB_SCALE)?;
    freq_skips.push(h.clone_shallow());
    for i in 1..depth {
        h = henc_layer(state, h, &model.encoders[i])?;
        freq_skips.push(h.clone_shallow());
    }
    let mut freq = h;
    pt.end(state, "01_freq_encoder");
    let fr_bot = freq.shape[2];
    let t_bot = freq.shape[3];

    // ─── 2. Time encoder chain ───────────────────────────────────────
    pt.start(state, "02_time_encoder");
    let mut time_skips: Vec<GpuTensor> = Vec::with_capacity(depth);
    let mut time_lengths: Vec<usize> = Vec::with_capacity(depth);
    let mut time = time.clone_shallow();
    for i in 0..depth {
        time_lengths.push(time.shape[2]);
        time = tenc_layer(state, time, &model.tencoders[i])?;
        time_skips.push(time.clone_shallow());
    }
    let t2_bot = time.shape[2];
    pt.end(state, "02_time_encoder");

    // ─── 3. Cross-domain transformer ────────────────────────────────
    pt.start(state, "03_transformer_setup");
    let d_model = model.crosstransformer.norm_in.dim;
    let freq_pe = crate::ops_cpu::sin_embed_2d(d_model, fr_bot, t_bot);
    let time_pe = crate::ops_cpu::sin_embed_1d(t2_bot, d_model);
    let gfpe = state.upload_f32_as_f16(&freq_pe, vec![1, t_bot * fr_bot, d_model])?;
    let gtpe = state.upload_f32_as_f16(&time_pe, vec![1, t2_bot, d_model])?;
    pt.end(state, "03_transformer_setup");

    pt.start(state, "04_transformer_layers");
    let (freq_t, time_t) = cross_domain_transformer(
        state, &freq, &time, &model.crosstransformer, &gfpe, &gtpe,
    )?;
    pt.end(state, "04_transformer_layers");
    freq = freq_t;
    time = time_t;

    // ─── 4. Freq decoder chain (reverse, with skips) ────────────────
    pt.start(state, "05_freq_decoder");
    let freq_dims: Vec<usize> = freq_skips.iter().map(|s| s.shape[2]).collect();
    for i in 0..depth {
        let skip = freq_skips.pop().expect("freq skip");
        let target = if i + 1 < freq_dims.len() {
            freq_dims[freq_dims.len() - 2 - i]
        } else {
            crate::N_FFT / 2
        };
        freq = hdec_layer(state, freq, &skip, target, &model.decoders[i])?;
    }
    pt.end(state, "05_freq_decoder");

    // ─── 5. Time decoder chain (reverse, with skips) ────────────────
    pt.start(state, "06_time_decoder");
    for i in 0..depth {
        let skip = time_skips.pop().expect("time skip");
        let target = time_lengths[time_lengths.len() - 1 - i];
        time = tdec_layer(state, time, &skip, target, &model.tdecoders[i])?;
    }
    pt.end(state, "06_time_decoder");

    pt.resolve();
    pt.print("htdemucs_forward");
    Ok((freq, time))
}
