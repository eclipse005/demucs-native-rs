//! henc but fully on CPU to check the reference is sane.
#![cfg(feature = "cuda")]
use demucs_core_native::model::HTDemucs;
use demucs_core_native::weights::WeightStore;
use demucs_core_native::ops_cpu;

#[test]
#[ignore]
fn henc0_cpu_sanity() {
    let model_path = std::path::Path::new("../models/htdemucs.safetensors");
    let store = WeightStore::load(model_path).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "955717e8", 4, 512).expect("from_store");
    let layer = &cpu_model.encoders[0];
    let b = 1; let c_in = 4; let fr = 64; let t = 32;
    let input: Vec<f32> = (0..b*c_in*fr*t).map(|i| ((i as f32)*0.013 - 0.3)*0.5).collect();
    let (out, sh) = ops_cpu::henc_layer_forward(&input, [b, c_in, fr, t], layer);
    let max_val = out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let min_val = out.iter().fold(f32::MAX, |a, b| a.min(*b));
    let mut max_idx = 0;
    let mut max_v = 0.0f32;
    for (i, &v) in out.iter().enumerate() {
        if v.abs() > max_v.abs() {
            max_v = v;
            max_idx = i;
        }
    }
    eprintln!("henc[0] shape={:?} max_val={:.4} min_val={:.4}", sh, max_val, min_val);
    eprintln!("max at idx {} = {}", max_idx, max_v);
    eprintln!("out[24528] = {}", out[24528]);
}
