//! Error types for the hand-written Demucs inference engines.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, DemucsError>;

#[derive(Debug, Error)]
pub enum DemucsError {
    #[error("weight loading failed: {0}")]
    Weight(String),

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("DSP error: {0}")]
    Dsp(String),

    #[error("shape mismatch: {0}")]
    Shape(String),

    #[error("internal: {0}")]
    Internal(String),
}
