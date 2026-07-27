//! GPU-resident inference engine for HTDemucs v4.
//!
//! cuBLAS handles every GEMM (linear layers, im2col convolutions); hand-written
//! NVRTC kernels handle element-wise ops (norms, activations, CaC reshaping).
//! Weights stay on the device after load — no CPU↔GPU round-trips in the
//! steady-state forward pass.
//!
//! Status: skeleton — CudaState (context/stream/cuBLAS/NVRTC) is functional;
//! the conv/transformer/decoder operators are filled in incrementally.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::sys;
use cudarc::driver::{
    safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, PushKernelArg},
    DevicePtr, DriverError, LaunchConfig,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use half::f16;

use crate::metadata::ModelInfo;
use crate::weights::WeightStore;
use crate::{LoadOptions, Stem, StemSelection};

const KERNEL_SRC: &str = include_str!("kernels/kernels.cu");

// ═══════════════════════════════════════════════════════════════════════
//  GpuTensor — owned f16 tensor on the GPU
// ═══════════════════════════════════════════════════════════════════════

/// A contiguous f16 tensor living in GPU memory.
pub struct GpuTensor {
    pub data: CudaSlice<f16>,
    pub(crate) shape: Vec<usize>,
}

impl GpuTensor {
    pub fn new(data: CudaSlice<f16>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "GpuTensor data len mismatch (shape {:?})",
            shape
        );
        Self { data, shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// A 2D weight matrix on the GPU: [out_features, in_features] row-major f16.
pub(crate) struct GpuWeightF16 {
    pub(crate) data: CudaSlice<f16>,
    pub(crate) rows: usize, // out_features
    pub(crate) cols: usize, // in_features
}

// ═══════════════════════════════════════════════════════════════════════
//  CudaState — context, stream, cuBLAS handle, kernel registry
// ═══════════════════════════════════════════════════════════════════════

/// Registry of compiled NVRTC kernels, looked up by name.
#[allow(non_snake_case)]
pub(crate) struct CudaKernels {
    pub noop: CudaFunction,
    // ─── HTDemucs element-wise ──────────────────────────────────────
    pub groupnorm1: CudaFunction,
    pub glu_channel: CudaFunction,
    pub layer_scale: CudaFunction,
    pub layer_scale_last: CudaFunction,
    pub add_bias_inplace: CudaFunction,
    pub add_to: CudaFunction,
    pub add_inplace: CudaFunction,
    pub add_pe: CudaFunction,
    pub add_freq_emb: CudaFunction,
    pub softmax_scaled: CudaFunction,
    pub denorm_freq: CudaFunction,
    pub norm_freq: CudaFunction,
    pub layer_norm: CudaFunction,
    pub swap_dims_12_3d: CudaFunction,
    pub swap_dims_12_4d: CudaFunction,
    pub permute_bcft_to_btcf: CudaFunction,
    pub transpose: CudaFunction,
    pub gelu: CudaFunction,
    pub gelu_bias: CudaFunction,
    // ─── im2col ────────────────────────────────────────────────────
    pub im2col_8x1_s4p2: CudaFunction,
    pub im2col_8_s4p2_1d: CudaFunction,
    pub im2col_3x3_s1p1: CudaFunction,
    pub im2col_1d_k3_dilation: CudaFunction,
    pub im2col_1d_k1: CudaFunction,
    pub im2col_conv_transpose_8x1_s4p2: CudaFunction,
    pub im2col_conv_transpose_8_s4p2_1d: CudaFunction,
    pub conv2d_postprocess: CudaFunction,
    pub trim_h2: CudaFunction,
    pub trim_l: CudaFunction,
    pub zero_pad_right: CudaFunction,
    pub reshape_bcft_to_bfct: CudaFunction,
    pub reshape_bfct_to_bcft: CudaFunction,
    pub transpose_bchw_to_bhwc: CudaFunction,
    pub transpose_bhwc_to_bchw: CudaFunction,
    pub permute_bsd_to_bhsd: CudaFunction,
    pub permute_bhsd_to_bsd: CudaFunction,
    pub permute_bhsd_to_bhds: CudaFunction,
    pub copy_per_head: CudaFunction,
    pub scatter_per_head: CudaFunction,
    pub flatten_bcft_to_btfc: CudaFunction,
    pub unflatten_btfc_to_bcft: CudaFunction,
    pub convert_f16_to_f32: CudaFunction,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
}

pub struct CudaState {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) blas: CudaBlas,
    pub(crate) k: CudaKernels,
}

unsafe impl Send for CudaState {}
unsafe impl Sync for CudaState {}

impl CudaState {
    /// Initialise CUDA on device `ordinal` (0 = first GPU).
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        Self::new_with_ctx(&ctx)
    }

    pub(crate) fn new_with_ctx(ctx: &Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;

        // Enable Tensor Core math mode — no-op on Pascal, ensures TC usage on Ampere+.
        unsafe {
            sys::cublasSetMathMode(*blas.handle(), sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH);
        }

        // NVRTC: target native arch for better codegen.
        let cuda_include = std::env::var("CUDA_PATH")
            .map(|p| format!("{}/include", p))
            .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
        let arch: Option<&'static str> = ctx.compute_capability().ok().map(|(major, minor)| {
            &*Box::leak(format!("sm_{}{}", major, minor).into_boxed_str())
        });
        let opts = CompileOptions {
            arch,
            include_paths: vec![cuda_include],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SRC, opts)
            .map_err(|e| anyhow!("kernel compile failed: {:?}", e))?;

        // Load PTX as a module, then resolve kernel functions from it.
        let module = ctx
            .load_module(ptx)
            .map_err(|e| anyhow!("load_module failed: {:?}", e))?;

        // Helper closure: load_function with cleaner error.
        let load = |name: &str| -> Result<CudaFunction> {
            module
                .load_function(name)
                .map_err(|e| anyhow!("load kernel {name} failed: {e:?}"))
        };

        let noop = load("noop_placeholder")?;
        // HTDemucs element-wise kernels
        let groupnorm1 = load("groupnorm1_f16")?;
        let glu_channel = load("glu_channel_f16")?;
        let layer_scale = load("layer_scale_f16")?;
        let layer_scale_last = load("layer_scale_last_f16")?;
        let add_bias_inplace = load("add_bias_inplace_f16")?;
        let add_to = load("add_to_f16")?;
        let add_inplace = load("add_inplace_f16")?;
        let add_pe = load("add_pe_f16")?;
        let add_freq_emb = load("add_freq_emb_f16")?;
        let softmax_scaled = load("softmax_scaled_f16")?;
        let denorm_freq = load("denorm_freq_f16")?;
        let norm_freq = load("norm_freq_f16")?;
        let layer_norm = load("layer_norm_f16")?;
        let swap_dims_12_3d = load("swap_dims_12_3d_f16")?;
        let swap_dims_12_4d = load("swap_dims_12_4d_f16")?;
        let permute_bcft_to_btcf = load("permute_bcft_to_btcf_f16")?;
        let transpose = load("transpose_f16")?;
        let gelu = load("gelu_f16")?;
        let gelu_bias = load("gelu_bias_f16")?;
        // im2col kernels
        let im2col_8x1_s4p2 = load("im2col_8x1_s4p2_f16")?;
        let im2col_8_s4p2_1d = load("im2col_8_s4p2_1d_f16")?;
        let im2col_3x3_s1p1 = load("im2col_3x3_s1p1_f16")?;
        let im2col_1d_k3_dilation = load("im2col_1d_k3_dilation_f16")?;
        let im2col_1d_k1 = load("im2col_1d_k1_f16")?;
        let im2col_conv_transpose_8x1_s4p2 =
            load("im2col_conv_transpose_8x1_s4p2_f16")?;
        let im2col_conv_transpose_8_s4p2_1d =
            load("im2col_conv_transpose_8_s4p2_1d_f16")?;
        let conv2d_postprocess = load("conv2d_postprocess_f16")?;
        let trim_h2 = load("trim_h2_f16")?;
        let trim_l = load("trim_l_f16")?;
        let zero_pad_right = load("zero_pad_right_f16")?;
        let reshape_bcft_to_bfct = load("reshape_bcft_to_bfct_f16")?;
        let reshape_bfct_to_bcft = load("reshape_bfct_to_bcft_f16")?;
        let transpose_bchw_to_bhwc = load("transpose_bchw_to_bhwc_f16")?;
        let transpose_bhwc_to_bchw = load("transpose_bhwc_to_bchw_f16")?;
        let permute_bsd_to_bhsd = load("permute_bsd_to_bhsd_f16")?;
        let permute_bhsd_to_bsd = load("permute_bhsd_to_bsd_f16")?;
        let permute_bhsd_to_bhds = load("permute_bhsd_to_bhds_f16")?;
        let copy_per_head = load("copy_per_head_f16")?;
        let scatter_per_head = load("scatter_per_head_f16")?;
        let flatten_bcft_to_btfc = load("flatten_bcft_to_btfc_f16")?;
        let unflatten_btfc_to_bcft = load("unflatten_btfc_to_bcft_f16")?;
        let convert_f16_to_f32 = load("convert_f16_to_f32_f32")?;

        Ok(Self {
            ctx: ctx.clone(),
            stream,
            blas,
            k: CudaKernels {
                noop,
                groupnorm1,
                glu_channel,
                layer_scale,
                layer_scale_last,
                add_bias_inplace,
                add_to,
                add_inplace,
                add_pe,
                add_freq_emb,
                softmax_scaled,
                denorm_freq,
                norm_freq,
                layer_norm,
                swap_dims_12_3d,
                swap_dims_12_4d,
                permute_bcft_to_btcf,
                transpose,
                gelu,
                gelu_bias,
                im2col_8x1_s4p2,
                im2col_8_s4p2_1d,
                im2col_3x3_s1p1,
                im2col_1d_k3_dilation,
                im2col_1d_k1,
                im2col_conv_transpose_8x1_s4p2,
                im2col_conv_transpose_8_s4p2_1d,
                conv2d_postprocess,
                trim_h2,
                trim_l,
                zero_pad_right,
                reshape_bcft_to_bfct,
                reshape_bfct_to_bcft,
                transpose_bchw_to_bhwc,
                transpose_bhwc_to_bchw,
                permute_bsd_to_bhsd,
                permute_bhsd_to_bsd,
                permute_bhsd_to_bhds,
                copy_per_head,
                scatter_per_head,
                flatten_bcft_to_btfc,
                unflatten_btfc_to_bcft,
                convert_f16_to_f32,
                module,
            },
        })
    }

    // ─── Memory helpers (all on stream) ──────────────────────────────────

    /// Upload a host f16 slice to the device as a contiguous tensor.
    pub fn upload_f16(&self, data: &[f16], shape: Vec<usize>) -> Result<GpuTensor> {
        let slice = self
            .stream
            .clone_htod(data)
            .map_err(|e| anyhow!("upload_f16 H2D failed: {:?}", e))?;
        Ok(GpuTensor::new(slice, shape))
    }

    /// Upload a host f32 slice, converting to f16 on the host first.
    pub fn upload_f32_as_f16(
        &self,
        data: &[f32],
        shape: Vec<usize>,
    ) -> Result<GpuTensor> {
        let f16_data: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
        self.upload_f16(&f16_data, shape)
    }

    /// Download an f16 GPU tensor to a host Vec<f32>.
    pub fn download_to_f32(&self, t: &GpuTensor) -> Result<Vec<f32>> {
        let host: Vec<f16> = self
            .stream
            .clone_dtoh(&t.data)
            .map_err(|e| anyhow!("download_to_f32 D2H failed: {:?}", e))?;
        Ok(host.iter().map(|v| v.to_f32()).collect())
    }

    /// Fast download: convert f16→f32 on GPU first, then D2H as f32.
    /// Avoids the ~700ms CPU f16→f32 conversion on 11M-element tensors.
    pub fn download_to_f32_fast(&self, t: &GpuTensor) -> Result<Vec<f32>> {
        let n = t.data.len();
        let mut f32_dev = unsafe { self.stream.alloc::<f32>(n) }
            .map_err(|e| anyhow!("alloc f32 failed: {:?}", e))?;
        let grid = ((n as u32) + 1023) / 1024;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.convert_f16_to_f32);
        bb.arg(&mut f32_dev);
        bb.arg(&t.data);
        bb.arg(&n_i32);
        unsafe { bb.launch(cfg) }
            .map_err(|e| anyhow!("convert_f16_to_f32 launch failed: {:?}", e))?;
        let host: Vec<f32> = self
            .stream
            .clone_dtoh(&f32_dev)
            .map_err(|e| anyhow!("D2H f32 failed: {:?}", e))?;
        Ok(host)
    }

    /// Allocate an uninitialised f16 buffer (no memset). Faster than
    /// alloc_zeros when the caller will fully overwrite the buffer.
    pub fn alloc_uninit_f16(&self, n: usize) -> Result<CudaSlice<f16>> {
        // SAFETY: callers overwrite every element before reading.
        unsafe { self.stream.alloc::<f16>(n) }
            .map_err(|e| anyhow!("alloc_uninit_f16 failed: {:?}", e))
    }

    /// Block until all queued work on the stream is done.
    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("synchronize failed: {:?}", e))
    }

    // ─── cuBLAS GEMM ─────────────────────────────────────────────────────

    /// f16 GEMM: `C[m, n] = A[m, k] @ B[k, n]`, all row-major. (beta = 0, fresh output.)
    ///
    /// cuBLAS is column-major. We pass `b` as the first arg and `a` as the
    /// second (swapped from the natural order), with both `OP_N`. cuBLAS
    /// then computes `op(B)[N, K] @ op(A)[K, M] = C[N, M]` col-major, which
    /// is `(A @ B)^T` col-major = `A @ B` row-major. Both `lda` and `ldb`
    /// match the inner dim of the row-major matrices (`k`), and `ldc` is
    /// the output row stride `n`. Verified by `tests/cuda_gemm_probe.rs`.
    pub fn gemm_f16(
        &self,
        a: &GpuTensor, // [m, k] row-major
        b: &GpuTensor, // [k, n] row-major
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<GpuTensor> {
        let mut c = self.alloc_uninit_f16(m * n)?;
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_N,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: n as i32,
                        n: m as i32,
                        k: k as i32,
                        alpha: f16::from_f32(1.0),
                        lda: n as i32, // = cols of b (row-major) = rows of b col-major
                        ldb: k as i32, // = cols of a (row-major) = rows of a col-major
                        beta: f16::from_f32(0.0),
                        ldc: n as i32,
                    },
                    &b.data,
                    &a.data,
                    &mut c,
                )
                .map_err(|e| anyhow!("gemm_f16 failed: {:?}", e))?;
        }
        Ok(GpuTensor::new(c, vec![m, n]))
    }

    /// f16 strided-batched GEMM with full cuBLAS args exposed. Computes
    /// `C[m, n] = op(A) @ op(B)` col-major per batch. Output col-major
    /// `[m, n]` has the same memory layout as row-major `[n, m]`.
    /// For our MHA: pass `OP_T` for Q@K^T and `OP_N` for attn@V (see
    /// `cuda_ops::mha`).
    pub fn gemm_strided_batched_f16(
        &self,
        a_data: &CudaSlice<f16>, // first arg (cublasA)
        b_data: &CudaSlice<f16>, // second arg (cublasB)
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        batch_size: usize,
        m_cublas: usize,
        n_cublas: usize,
        k_cublas: usize,
        lda: usize,
        ldb: usize,
        stride_a: usize, // in elements
        stride_b: usize,
    ) -> Result<GpuTensor> {
        let mut c = self.alloc_uninit_f16(batch_size * m_cublas * n_cublas)?;
        unsafe {
            self.blas
                .gemm_strided_batched(
                    StridedBatchedConfig {
                        gemm: GemmConfig {
                            transa,
                            transb,
                            m: m_cublas as i32,
                            n: n_cublas as i32,
                            k: k_cublas as i32,
                            alpha: f16::from_f32(1.0),
                            lda: lda as i32,
                            ldb: ldb as i32,
                            beta: f16::from_f32(0.0),
                            ldc: m_cublas as i32,
                        },
                        batch_size: batch_size as i32,
                        stride_a: stride_a as i64,
                        stride_b: stride_b as i64,
                        stride_c: (m_cublas * n_cublas) as i64,
                    },
                    a_data,
                    b_data,
                    &mut c,
                )
                .map_err(|e| anyhow!("gemm_strided_batched_f16 failed: {:?}", e))?;
        }
        Ok(GpuTensor::new(c, vec![batch_size * m_cublas, n_cublas]))
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  CudaEngine — holds loaded GPU model(s) + selected stems
// ═══════════════════════════════════════════════════════════════════════

use crate::dsp::stft::Stft;
use crate::model::HTDemucs;
use crate::{ops_cpu, TRAINING_LENGTH};

/// One HTDemucs model resident on the GPU.
pub(crate) struct CudaModel {
    pub(crate) sig: String,
    pub(crate) model: crate::gpu_model::GpuHTDemucs,
}

pub struct CudaEngine {
    state: Arc<CudaState>,
    info: &'static ModelInfo,
    models: Vec<CudaModel>,
    selected_stems: Vec<crate::metadata::StemId>,
    n_sources: usize,
    bottom_channels: usize,
}

unsafe impl Send for CudaEngine {}
unsafe impl Sync for CudaEngine {}

impl CudaEngine {
    /// Load model weights from the store into GPU memory.
    pub(crate) fn load(
        store: WeightStore,
        info: &'static ModelInfo,
        opts: &LoadOptions,
        state: Arc<CudaState>,
    ) -> Result<Self> {
        let selected_stems: Vec<crate::metadata::StemId> = match &opts.stems {
            StemSelection::All => info.stems.to_vec(),
            StemSelection::Some(s) => s.clone(),
        };
        let n_sources = info.stems.len();
        let bottom_channels = if n_sources == 6 { 384 } else { 512 };

        let sigs_to_load: Vec<String> = if info.signatures.len() == 1 {
            vec![info.signatures[0].to_string()]
        } else {
            info.stems
                .iter()
                .enumerate()
                .filter(|(_, &s)| selected_stems.contains(&s))
                .map(|(i, _)| info.signatures[i].to_string())
                .collect()
        };

        let mut models = Vec::with_capacity(sigs_to_load.len());
        for sig in &sigs_to_load {
            anyhow::ensure!(
                store.signature(sig).is_some(),
                "signature {} not found in weight store",
                sig
            );
            // Build CPU model from store, then mirror onto the GPU.
            let cpu_model = HTDemucs::from_store(&store, sig, n_sources, bottom_channels)
                .map_err(|e| anyhow!("HTDemucs::from_store({sig}): {e}"))?;
            let gpu_model = crate::gpu_model::GpuHTDemucs::from_cpu(&state, &cpu_model)?;
            models.push(CudaModel {
                sig: sig.clone(),
                model: gpu_model,
            });
        }

        let dev_name = state.ctx.name().unwrap_or_default();
        log::info!(
            "CudaEngine: loaded {} model(s) for {} on {}",
            models.len(),
            info.id,
            dev_name
        );

        Ok(Self {
            state,
            info,
            models,
            selected_stems,
            n_sources,
            bottom_channels,
        })
    }

    /// GPU pipeline: H2D + forward + D2H. Returns raw denorm buffers (caller does post-processing).
    fn gpu_chunk(
        &self,
        freq_n: Vec<f32>,
        time_n: Vec<f32>,
        n_frames: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        Self::gpu_chunk_on(&self.state, &self.models[0].model, freq_n, time_n, n_frames)
    }

    /// Static version for use from a spawned thread where CudaEngine can't be borrowed.
    fn gpu_chunk_on(
        state: &Arc<CudaState>,
        model: &crate::gpu_model::GpuHTDemucs,
        freq_n: Vec<f32>,
        time_n: Vec<f32>,
        n_frames: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let freq_shape = [1, 4, crate::N_FFT / 2, n_frames];
        let time_shape = [1, 2, TRAINING_LENGTH];
        // Normal forward — output correctness preserved.
        let gf = state.upload_f32_as_f16(&freq_n, freq_shape.to_vec())?;
        let gt = state.upload_f32_as_f16(&time_n, time_shape.to_vec())?;
        let (gf_out, gt_out) = crate::cuda_ops::htdemucs_forward(state, &gf, &gt, model)?;
        let f_dl = state.download_to_f32_fast(&gf_out)?;
        let t_dl = state.download_to_f32_fast(&gt_out)?;

        Ok((f_dl, t_dl))
    }

    /// Short-audio path (no chunking): single segment.
    fn separate_single_segment(&self, left: &[f32], right: &[f32]) -> Result<Vec<Stem>> {
        let n_samples = left.len();
        let padded_len = TRAINING_LENGTH;
        let mut lp = vec![0.0f32; padded_len];
        let mut rp = vec![0.0f32; padded_len];
        lp[..n_samples].copy_from_slice(left);
        rp[..n_samples].copy_from_slice(right);

        let mut stft = Stft::new(crate::N_FFT, crate::HOP_LENGTH);
        let ls = stft.forward(&lp)?;
        let rs = stft.forward(&rp)?;
        let n_bins = crate::N_FFT / 2;
        let n_frames = ls.len() / n_bins;

        let lc = crate::dsp::cac::stft_to_cac(&ls, crate::N_FFT);
        let rc = crate::dsp::cac::stft_to_cac(&rs, crate::N_FFT);
        let freq = crate::cpu_engine::build_freq_tensor(&lc, &rc, n_bins, n_frames);
        let time = crate::cpu_engine::build_time_tensor(&lp, &rp, padded_len);

        let freq_shape = [1, 4, n_bins, n_frames];
        let time_shape = [1, 2, padded_len];
        let (freq_n, _, fmean, _, fstd, _) = ops_cpu::normalize_freq(&freq, freq_shape);
        let (time_n, _, tmean, _, tstd, _) = ops_cpu::normalize_time(&time, time_shape);

        let (f_dl, t_dl) = self.gpu_chunk(freq_n, time_n, n_frames)?;

        let f_shape = [1, self.n_sources * 4, f_dl.len() / (self.n_sources * 4 * n_frames), n_frames];
        let t_shape = [1, self.n_sources * 2, t_dl.len() / (self.n_sources * 2)];
        ops_cpu::denormalize_freq(&mut f_dl.clone(), f_shape, &fmean, &fstd);
        ops_cpu::denormalize_time(&mut t_dl.clone(), t_shape, &tmean, &tstd);
        let stems = ops_cpu::extract_stems(&f_dl, f_shape, &t_dl, t_shape, n_frames, padded_len, padded_len, &mut stft);

        let stems: Vec<_> = stems.into_iter().filter(|s| self.selected_stems.contains(&s.id)).collect();
        Ok(stems)
    }

    /// Run source separation on stereo audio.
    pub fn separate(
        &self,
        left: &[f32],
        right: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<Stem>> {
        use std::borrow::Cow;
        let needs_resample = sample_rate != crate::SAMPLE_RATE as u32;
        let (left_in, right_in): (Cow<[f32]>, Cow<[f32]>) = if needs_resample {
            let l = crate::dsp::resample::resample_channel(left, sample_rate, crate::SAMPLE_RATE as u32)
                .map_err(|e| anyhow!("resample: {e}"))?;
            let r = crate::dsp::resample::resample_channel(right, sample_rate, crate::SAMPLE_RATE as u32)
                .map_err(|e| anyhow!("resample: {e}"))?;
            (Cow::Owned(l), Cow::Owned(r))
        } else {
            (Cow::Borrowed(left), Cow::Borrowed(right))
        };
        let left = &*left_in;
        let right = &*right_in;
        let n_samples = left.len();

        if n_samples <= TRAINING_LENGTH {
            return self.separate_single_segment(left, right);
        }

        // Pipelined multi-threaded chunk processing:
        //
        // Timing per chunk:
        //   Background thread: prep (STFT+cac+normalize) ~24ms
        //   GPU forward:  ~92ms
        //   D2H:         CPU thread blocked ~345ms during D2H
        //   Post (denorm+extract+oa): ~75ms
        //
        // Pipeline: GPU runs on chunk N while background threads prep chunks N+1,N+2,...
        // D2H is the pipeline hazard: CPU thread blocks waiting for it while GPU is idle.
        // This means GPU idle time = D2H duration - post of next chunk.
        // With 345ms D2H and ~75ms post: GPU is idle for ~270ms per chunk.
        //
        // Expected wall: sum of (GPU+D2H_blocked) = ~437ms × 31 + startup ≈ 13.6s (vs 22s sequential = 1.6× win).
        let segment = TRAINING_LENGTH;
        let stride = segment * 3 / 4;
        let num_chunks = n_samples.saturating_sub(segment).div_ceil(stride) + 1;
        let n_stems = self.info.stems.len();
        let mut out_left = vec![vec![0.0f32; n_samples]; n_stems];
        let mut out_right = vec![vec![0.0f32; n_samples]; n_stems];
        let mut sum_weight = vec![0.0f32; n_samples];

        // prep → gpu channel
        let (tx_prep, rx_prep) = std::sync::mpsc::channel::<(
            usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize, usize,
        )>();
        // gpu → post channel
        let (tx_gpu, rx_gpu) = std::sync::mpsc::channel::<(
            usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize, usize,
        )>();

        let left_arc = std::sync::Arc::new(left.to_vec());
        let right_arc = std::sync::Arc::new(right.to_vec());

        let n_sources = self.n_sources;
        let selected_stems = self.selected_stems.clone();
        let info: &'static [_] = self.info.stems;
        let gpu_state = self.state.clone();
        let model = std::sync::Arc::new(self.models[0].model.clone());

        // GPU thread: owns engine, processes prepped chunks as they arrive.
        // This allows prep (next chunk) and post (prev chunk) to overlap with GPU.
        let gpu_handle = {
            let rx_prep = rx_prep;
            let tx_gpu = tx_gpu;
            std::thread::spawn(move || {
                for (chunk_idx, freq_n, time_n, fmean, fstd, tmean, tstd, n_frames, chunk_len) in rx_prep {
                    let (f_dl, t_dl) = Self::gpu_chunk_on(&gpu_state, &model, freq_n, time_n, n_frames).expect("gpu_chunk failed");
                    let _ = tx_gpu.send((chunk_idx, f_dl, t_dl, fmean, fstd, tmean, tstd, n_frames, chunk_len));
                }
            })
        };

        // Prep thread: STFT + CaC + normalize. Runs ahead of GPU.
        let prep_handle = {
            let left_arc = left_arc.clone();
            let right_arc = right_arc.clone();
            std::thread::spawn(move || {
                for chunk_idx in 0..num_chunks {
                    let start = chunk_idx * stride;
                    let end = (start + segment).min(n_samples);
                    let chunk_len = end - start;

                    let mut lp = vec![0.0f32; segment];
                    let mut rp = vec![0.0f32; segment];
                    lp[..chunk_len].copy_from_slice(&left_arc[start..end]);
                    rp[..chunk_len].copy_from_slice(&right_arc[start..end]);

                    let mut stft = Stft::new(crate::N_FFT, crate::HOP_LENGTH);
                    let ls = stft.forward(&lp).expect("stft failed");
                    let rs = stft.forward(&rp).expect("stft failed");
                    let n_bins = crate::N_FFT / 2;
                    let n_frames = ls.len() / n_bins;

                    let lc = crate::dsp::cac::stft_to_cac(&ls, crate::N_FFT);
                    let rc = crate::dsp::cac::stft_to_cac(&rs, crate::N_FFT);
                    let freq = crate::cpu_engine::build_freq_tensor(&lc, &rc, n_bins, n_frames);
                    let time = crate::cpu_engine::build_time_tensor(&lp, &rp, segment);

                    let freq_shape = [1, 4, n_bins, n_frames];
                    let time_shape = [1, 2, segment];
                    let (freq_n, _, fmean, _, fstd, _) =
                        ops_cpu::normalize_freq(&freq, freq_shape);
                    let (time_n, _, tmean, _, tstd, _) =
                        ops_cpu::normalize_time(&time, time_shape);

                    let _ = tx_prep.send((
                        chunk_idx, freq_n, time_n, fmean, fstd, tmean, tstd, n_frames, chunk_len,
                    ));
                }
                drop(tx_prep);
            })
        };

        // Post thread: denorm + extract + triangular OA.
        // Runs concurrently with GPU (overlapping the ~345ms D2H blocking).
        let out_left = std::sync::Arc::new(std::sync::Mutex::new(out_left));
        let out_right = std::sync::Arc::new(std::sync::Mutex::new(out_right));
        let sum_weight = std::sync::Arc::new(std::sync::Mutex::new(sum_weight));
        let post_handle = {
            let out_left = out_left.clone();
            let out_right = out_right.clone();
            let sum_weight = sum_weight.clone();
            std::thread::spawn(move || {
                let mut stft = Stft::new(crate::N_FFT, crate::HOP_LENGTH);
                for (chunk_idx, f_dl, t_dl, fmean, fstd, tmean, tstd, n_frames, chunk_len) in rx_gpu {
                    let start = chunk_idx * stride;
                    let end = (start + segment).min(n_samples);
                    let this_len = end - start;

                    let f_shape = [1, n_sources * 4, f_dl.len() / (n_sources * 4 * n_frames), n_frames];
                    let t_shape = [1, n_sources * 2, t_dl.len() / (n_sources * 2)];
                    ops_cpu::denormalize_freq(&mut f_dl.clone(), f_shape, &fmean, &fstd);
                    ops_cpu::denormalize_time(&mut t_dl.clone(), t_shape, &tmean, &tstd);

                    let stems = ops_cpu::extract_stems(
                        &f_dl, f_shape, &t_dl, t_shape, n_frames, segment, chunk_len, &mut stft,
                    );

                    let window = crate::cpu_engine::triangular_window(chunk_len);
                    {
                        let mut out_left = out_left.lock().unwrap();
                        let mut out_right = out_right.lock().unwrap();
                        let mut sum_weight = sum_weight.lock().unwrap();
                        for stem in stems {
                            let s = info.iter().position(|&id| id == stem.id).unwrap();
                            for i in 0..this_len {
                                let w = window[i];
                                out_left[s][start + i] += w * stem.left[i];
                                out_right[s][start + i] += w * stem.right[i];
                            }
                        }
                        for i in 0..this_len {
                            sum_weight[start + i] += window[i];
                        }
                    }
                }
            })
        };

        prep_handle.join().expect("prep thread panicked");
        gpu_handle.join().expect("gpu thread panicked");
        post_handle.join().expect("post thread panicked");

        // Normalize by window sum.
        let sum_weight = std::sync::Arc::try_unwrap(sum_weight).unwrap().into_inner().unwrap();
        let mut out_left = std::sync::Arc::try_unwrap(out_left).unwrap().into_inner().unwrap();
        let mut out_right = std::sync::Arc::try_unwrap(out_right).unwrap().into_inner().unwrap();

        // Normalize by window sum.
        let mut stems = Vec::with_capacity(n_stems);
        for (s, &stem_id) in info.iter().enumerate() {
            for i in 0..n_samples {
                let w = sum_weight[i];
                if w > 0.0 {
                    out_left[s][i] /= w;
                    out_right[s][i] /= w;
                }
            }
            stems.push(crate::Stem {
                id: stem_id,
                left: std::mem::take(&mut out_left[s]),
                right: std::mem::take(&mut out_right[s]),
            });
        }

        let stems: Vec<_> = stems.into_iter().filter(|s| selected_stems.contains(&s.id)).collect();

        if needs_resample {
            let mut resampled = Vec::with_capacity(stems.len());
            for mut stem in stems {
                stem.left = crate::dsp::resample::resample_channel(&stem.left, crate::SAMPLE_RATE as u32, sample_rate)
                    .map_err(|e| anyhow!("resample: {e}"))?;
                stem.right = crate::dsp::resample::resample_channel(&stem.right, crate::SAMPLE_RATE as u32, sample_rate)
                    .map_err(|e| anyhow!("resample: {e}"))?;
                resampled.push(stem);
            }
            Ok(resampled)
        } else {
            Ok(stems)
        }
    }
}

// Silence unused-import warnings for traits that are used via methods.
#[allow(unused_imports)]
use cudarc::driver::{DevicePtr as _, DriverError as _};
