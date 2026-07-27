//! Hand-written CUDA + CPU inference engines for HTDemucs v4.
//!
//! Standalone crate — does **not** depend on burn. The sibling `demucs-core`
//! crate (burn-based) serves as the golden reference for numerical comparison
//! during the rewrite.
//!
//! Two backends:
//! - **CPU** (default): gemm crate + rayon, pure Rust
//! - **CUDA** (`--features cuda`): cuBLAS GEMM + hand-written NVRTC kernels
//!
//! Both backends store weights as f16 and compute in f32.

pub mod backend;
pub mod dsp;
pub mod error;
pub mod metadata;
pub mod model;
pub mod ops_cpu;
pub mod raw_tensor;
pub mod weights;

#[cfg(feature = "cuda")]
pub mod cuda_engine;

#[cfg(feature = "cuda")]
pub mod cuda_ops;

#[cfg(feature = "cuda")]
pub mod gpu_model;

pub mod cpu_engine;

pub use backend::Backend;
pub use error::{DemucsError, Result};
pub use metadata::{ModelInfo, StemId, ALL_MODELS};

use std::path::Path;

// ─── Model hyperparameters (HTDemucs v4) ─────────────────────────────────────

pub const AUDIO_CHANNELS: usize = 2;
pub const N_FFT: usize = 4096;
pub const HOP_LENGTH: usize = 1024;
pub const CHANNELS: usize = 48;
pub const GROWTH: usize = 2;
pub const DEPTH: u32 = 4;
pub const KERNEL_SIZE: usize = 8;
pub const STRIDE: usize = 4;
pub const T_LAYERS: usize = 5;
pub const T_HEADS: usize = 8;
pub const T_HIDDEN_SCALE: f32 = 4.0;
pub const DCONV_COMP: usize = 8;
pub const DCONV_DEPTH: usize = 2;
pub const SAMPLE_RATE: usize = 44100;
/// Training segment length in samples (= 39/5 * 44100).
pub const TRAINING_LENGTH: usize = 343980;

// ─── Public API ──────────────────────────────────────────────────────────────

/// Which model variant to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    FourStem,
    SixStem,
    FineTuned,
}

impl ModelVariant {
    pub fn info(&self) -> &'static ModelInfo {
        match self {
            ModelVariant::FourStem => &metadata::HTDEMUCS,
            ModelVariant::SixStem => &metadata::HTDEMUCS_6S,
            ModelVariant::FineTuned => &metadata::HTDEMUCS_FT,
        }
    }
}

/// Selection of stems to extract.
#[derive(Debug, Clone)]
pub enum StemSelection {
    All,
    Some(Vec<StemId>),
}

/// One separated stem (stereo, left + right channels).
#[derive(Debug, Clone)]
pub struct Stem {
    pub id: StemId,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

/// Top-level inference handle.
pub struct Demucs {
    inner: DemucsInner,
}

enum DemucsInner {
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda_engine::CudaEngine),
    Cpu(crate::cpu_engine::CpuEngine),
}

/// Options for loading a model.
#[derive(Clone)]
pub struct LoadOptions {
    pub variant: ModelVariant,
    pub stems: StemSelection,
}

impl Demucs {
    pub fn load(
        model_path: &Path,
        opts: LoadOptions,
        backend: Backend,
    ) -> anyhow::Result<Self> {
        let weights = weights::WeightStore::load(model_path)?;
        Self::from_weights(weights, opts, backend)
    }

    pub fn from_bytes(
        bytes: &[u8],
        opts: LoadOptions,
        backend: Backend,
    ) -> anyhow::Result<Self> {
        let weights = weights::WeightStore::from_bytes(bytes)?;
        Self::from_weights(weights, opts, backend)
    }

    fn from_weights(
        weights: weights::WeightStore,
        opts: LoadOptions,
        backend: Backend,
    ) -> anyhow::Result<Self> {
        let resolved = backend.resolve()?;
        let info = opts.variant.info();
        let inner = match resolved {
            #[cfg(feature = "cuda")]
            backend::ResolvedBackend::Cuda(state) => {
                let engine = crate::cuda_engine::CudaEngine::load(weights, info, &opts, state)?;
                DemucsInner::Cuda(engine)
            }
            backend::ResolvedBackend::Cpu => {
                let engine = crate::cpu_engine::CpuEngine::load(weights, info, &opts)?;
                DemucsInner::Cpu(engine)
            }
        };
        Ok(Self { inner })
    }

    pub fn separate(
        &self,
        left: &[f32],
        right: &[f32],
        sample_rate: u32,
    ) -> anyhow::Result<Vec<Stem>> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            DemucsInner::Cuda(e) => e.separate(left, right, sample_rate),
            DemucsInner::Cpu(e) => e.separate(left, right, sample_rate),
        }
    }
}

/// Number of chunks for chunked inference over long audio.
pub fn num_chunks(n_samples: usize) -> usize {
    if n_samples <= TRAINING_LENGTH {
        return 1;
    }
    let segment = TRAINING_LENGTH;
    let stride = segment * 3 / 4;
    n_samples.saturating_sub(segment).div_ceil(stride) + 1
}
