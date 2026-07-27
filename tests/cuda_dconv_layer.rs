//! DConv layer test: load htdemucs weights, run one DConvLayer (inside HEncLayer[0])
//! on CPU vs GPU.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use demucs_core_native::cuda_engine::{CudaState, GpuTensor};
use demucs_core_native::gpu_model::{GpuDConv, GpuDConvLayer, GpuHTDemucs};
use demucs_core_native::model::HTDemucs;
use demucs_core_native::weights::WeightStore;
use demucs_core_native::ops_cpu;

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
#[ignore]
fn cuda_dconv_layer_0_in_henc0_matches_cpu() {
    let model_path = std::path::Path::new("../models/htdemucs.safetensors");
    let store = WeightStore::load(model_path).expect("load model");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("from_store");

    // Build input that matches what HEncLayer[0] would feed to dconv_layer[0]:
    //   [B*Fr=16, C=48, T=32]
    let n = 16;
    let c = 48;
    let t = 32;
    let input: Vec<f32> = (0..n * c * t)
        .map(|i| ((i as f32) * 0.013 - 0.3) * 0.5)
        .collect();

    // Test both dilation=1 (layer 0) and dilation=2 (layer 1).
    for (li, dilation) in [0usize, 1].iter().map(|li| (li, 1 << li)) {
        let dconv = &cpu_model.encoders[0].dconv;
        let layer = &dconv.layers[*li];
        let (cpu_out, cpu_shape) =
            ops_cpu::dconv_layer_forward(&input, [n, c, t], layer, dilation);
        eprintln!("cpu dconv[{li}] dilation={dilation} out shape = {:?}", cpu_shape);

        let state = Arc::new(CudaState::new(0).expect("cuda init"));
        let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu from_cpu");
        let gpu_dconv: &GpuDConv = &gpu_model.encoders[0].dconv;
        let gpu_layer: &GpuDConvLayer = &gpu_dconv.layers[*li];
        let gpu_in = state.upload_f32(&input, vec![n, c, t]).expect("up");
        let gpu_out: GpuTensor = demucs_core_native::cuda_ops::dconv_layer(&state, gpu_in, gpu_layer, dilation)
            .expect("dconv_layer");
        let gpu_dl = state.download_to_f32(&gpu_out).expect("dl");

        let diff = max_diff(&cpu_out, &gpu_dl);
        let max_val = cpu_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let rms = (cpu_out.iter().zip(&gpu_dl).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
            / cpu_out.len() as f32)
            .sqrt();
        eprintln!("dconv[{li}](d={dilation}) cpu vs gpu: max_diff={:.4}, rms={:.4}, max_val={:.2}", diff, rms, max_val);
        eprintln!("cpu[0..5]={:?}", &cpu_out[..5]);
        eprintln!("gpu[0..5]={:?}", &gpu_dl[..5]);
        let tol = (max_val * 0.05).max(5e-1);
        assert!(
            diff < tol,
            "dconv_layer[{li}](d={dilation}) cpu vs gpu: max_diff={:.4} exceeds tol={:.4} (max_val={:.2}, rms={:.4})",
            diff, tol, max_val, rms
        );
    }
}
