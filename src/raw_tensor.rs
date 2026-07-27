//! Raw on-disk tensor view, loaded directly from safetensors bytes.
//!
//! Used only for weight loading — never as a GPU/CPU computation tensor (those
//! live in `cuda_engine` / `cpu_engine`). `RawTensor` is a deserialization
//! intermediate; the engines consume the raw bytes and upload them to their
//! respective devices.

use safetensors::Dtype;

/// One tensor as it sits in the safetensors file: raw bytes + shape + dtype.
#[derive(Debug, Clone)]
pub struct RawTensor {
    /// Raw little-endian bytes (f32 = 4 bytes, f16/bf16 = 2 bytes, etc.).
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl RawTensor {
    /// Convert raw bytes to a `Vec<f32>`. Supports F32 / F16 / BF16.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self.dtype {
            Dtype::F32 => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::F16 => self
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_ne_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Dtype::BF16 => self
                .data
                .chunks_exact(2)
                .map(|c| {
                    let b = u16::from_ne_bytes([c[0], c[1]]);
                    f32::from_bits((b as u32) << 16)
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Convert raw bytes to a `Vec<half::f16>`. Supports F32 / F16 / BF16.
    pub fn to_f16_vec(&self) -> Vec<half::f16> {
        match self.dtype {
            Dtype::F16 => self
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_ne_bytes([c[0], c[1]]))
                .collect(),
            Dtype::F32 => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .map(half::f16::from_f32)
                .collect(),
            Dtype::BF16 => self
                .data
                .chunks_exact(2)
                .map(|c| {
                    let b = u16::from_ne_bytes([c[0], c[1]]);
                    half::f16::from_f32(f32::from_bits((b as u32) << 16))
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// (f32_data, shape) — convenience for loaders that need both.
    pub fn as_f32(&self) -> (Vec<f32>, Vec<usize>) {
        (self.to_f32_vec(), self.shape.clone())
    }
}
