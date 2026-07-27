//! Safetensors weight loading for HTDemucs.
//!
//! HTDemucs weights are stored as a single `.safetensors` file. Each tensor key
//! is prefixed by an 8-hex-char "signature" (e.g. `955717e8.encoder.0.conv.weight`).
//! One signature = one complete HTDemucs model:
//!   - `htdemucs`     → 1 signature (`955717e8`)
//!   - `htdemucs_6s`  → 1 signature (`5c90dfd2`)
//!   - `htdemucs_ft`  → 4 signatures (one per stem model: vocals/drums/bass/other)
//!
//! This module loads the file and groups tensors by signature prefix.

use std::collections::HashMap;
use std::path::Path;

use crate::raw_tensor::RawTensor;

/// All tensors for one model file, grouped by signature prefix.
/// `map[sig][key]` = the tensor `sig.key` (key has the `sig.` prefix stripped).
pub struct WeightStore {
    /// signature → (key without prefix → RawTensor)
    pub by_signature: HashMap<String, HashMap<String, RawTensor>>,
}

impl WeightStore {
    /// Load a single safetensors file and group tensors by signature prefix.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let buf = std::fs::read(path)?;
        Self::from_bytes(&buf)
    }

    /// Load from in-memory bytes (e.g. already downloaded).
    pub fn from_bytes(buf: &[u8]) -> anyhow::Result<Self> {
        let st = safetensors::SafeTensors::deserialize(buf)
            .map_err(|e| anyhow::anyhow!("safetensors: {}", e))?;
        let names = st.names();
        let tensors = st.tensors();

        let mut by_signature: HashMap<String, HashMap<String, RawTensor>> = HashMap::new();

        for i in 0..names.len() {
            let full_key = names[i];
            let view = &tensors[i];
            let data = view.1.data().to_vec();
            let shape: Vec<usize> = view.1.shape().to_vec();
            let dtype = view.1.dtype();

            // Split on first '.' → (signature, rest)
            let (sig, rest) = match full_key.split_once('.') {
                Some((s, r)) => (s.to_string(), r.to_string()),
                None => continue, // skip keys without a dot (shouldn't happen)
            };

            by_signature
                .entry(sig)
                .or_default()
                .insert(rest, RawTensor { data, shape, dtype });
        }

        Ok(Self { by_signature })
    }

    /// Get the tensors for a specific signature (key without prefix).
    pub fn signature(&self, sig: &str) -> Option<&HashMap<String, RawTensor>> {
        self.by_signature.get(sig)
    }

    /// Take a tensor (consuming a clone) for a specific signature + key.
    pub fn take(&self, sig: &str, key: &str) -> anyhow::Result<RawTensor> {
        self.by_signature
            .get(sig)
            .and_then(|m| m.get(key))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing weight: {}.{}", sig, key))
    }

    /// Try to take a tensor; returns `None` if the key is absent.
    /// Useful for optional weights (e.g. 6-stem has no channel_upsampler).
    pub fn try_take(&self, sig: &str, key: &str) -> Option<RawTensor> {
        self.by_signature
            .get(sig)
            .and_then(|m| m.get(key))
            .cloned()
    }

    /// List all signatures present.
    pub fn signatures(&self) -> Vec<&str> {
        self.by_signature.keys().map(|s| s.as_str()).collect()
    }
}
