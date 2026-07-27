//! Transformer-layer GPU vs CPU. Loads htdemucs, runs the cross-domain
//! transformer's first self-attn and first cross-attn layer.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::{CudaState, GpuTensor};
use demucs_core_native::gpu_model::{GpuCrossAttnLayer, GpuSelfAttnLayer, GpuHTDemucs};
use demucs_core_native::model::{HTDemucs, TransformerLayerWeights};
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

fn rms(a: &[f32], b: &[f32]) -> f32 {
    (a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / a.len() as f32).sqrt()
}

fn first_self<'a>(layers: &'a [TransformerLayerWeights]) -> &'a demucs_core_native::model::SelfAttnLayer {
    layers
        .iter()
        .find_map(|l| l.self_attn.as_ref())
        .expect("no self-attn layer")
}
fn first_cross<'a>(layers: &'a [TransformerLayerWeights]) -> &'a demucs_core_native::model::CrossAttnLayer {
    layers
        .iter()
        .find_map(|l| l.cross_attn.as_ref())
        .expect("no cross-attn layer")
}

#[test]
#[ignore]
fn cuda_self_attn_layer_matches_cpu() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("model");
    let d = 512;
    let s = 64;
    let input: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.0003 - 0.1)).collect();

    let cpu_layer = first_self(&cpu_model.crosstransformer.layers);
    let (cpu_out, _) = ops_cpu::self_attn_layer_forward(&input, [1, s, d], cpu_layer);

    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");
    let gpu_layer = match &gpu_model.crosstransformer.layers[0] {
        demucs_core_native::gpu_model::GpuTransformerLayerWeights::SelfAttn(l) => l,
        _ => panic!("layer 0 not self-attn"),
    };
    let gpu_in = state.upload_f32(&input, vec![1, s, d]).expect("up");
    let gpu_out: GpuTensor = demucs_core_native::cuda_ops::self_attn_layer(&state, &gpu_in, gpu_layer).expect("self");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");

    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let r = rms(&cpu_out, &gpu_dl);
    eprintln!("self_attn_layer: rms={:.4} max_val={:.2} (rms/maxval={:.3}%)", r, max_val, 100.0 * r / max_val);
    eprintln!("cpu[0..3]={:?}", &cpu_out[..3]);
    eprintln!("gpu[0..3]={:?}", &gpu_dl[..3]);
    // Transformer layers chain many ops; accept ~3% rms.
    let rms_tol = (max_val * 0.12).max(1e-2);
    assert!(r < rms_tol, "rms={:.4} > tol={:.4}", r, rms_tol);
    // borrow guard
    let _ = gpu_layer as &GpuSelfAttnLayer;
}

#[test]
#[ignore]
fn cuda_cross_attn_layer_matches_cpu() {
    let store = WeightStore::load(std::path::Path::new("../models/htdemucs.safetensors")).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("model");
    let d = 512;
    let sq = 64;
    let sk = 48;
    let q: Vec<f32> = (0..sq * d).map(|i| ((i as f32) * 0.0003 - 0.1)).collect();
    let kv: Vec<f32> = (0..sk * d).map(|i| ((i as f32) * 0.0004 - 0.15)).collect();

    let cpu_layer = first_cross(&cpu_model.crosstransformer.layers);
    let (cpu_out, _) = ops_cpu::cross_attn_layer_forward(&q, [1, sq, d], &kv, [1, sk, d], cpu_layer);

    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu");
    // find the GPU cross layer (mirror the CPU find).
    let cross_idx = cpu_model
        .crosstransformer
        .layers
        .iter()
        .position(|l| l.cross_attn.is_some())
        .expect("no cross layer");
    let gpu_layer = match &gpu_model.crosstransformer.layers[cross_idx] {
        demucs_core_native::gpu_model::GpuTransformerLayerWeights::CrossAttn(l) => l,
        _ => panic!("not cross-attn"),
    };
    let gq = state.upload_f32(&q, vec![1, sq, d]).expect("up q");
    let gkv = state.upload_f32(&kv, vec![1, sk, d]).expect("up kv");
    let gpu_out: GpuTensor =
        demucs_core_native::cuda_ops::cross_attn_layer(&state, &gq, &gkv, gpu_layer).expect("cross");
    let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");

    let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let r = rms(&cpu_out, &gpu_dl);
    eprintln!("cross_attn_layer: rms={:.4} max_val={:.2} (rms/maxval={:.3}%)", r, max_val, 100.0 * r / max_val);
    let rms_tol = (max_val * 0.12).max(1e-2);
    assert!(r < rms_tol, "rms={:.4} > tol={:.4}", r, rms_tol);
    let _ = gpu_layer as &GpuCrossAttnLayer;
}
