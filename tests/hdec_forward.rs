//! HDecLayer end-to-end forward test with real weights.
//!
//! Loads htdemucs_ft's vocals freq decoder.3 (the deepest decoder layer:
//! 384 → 192 channels, Fr 8 → 32, last=false) and runs a forward pass on a
//! synthetic input, verifying:
//!   - shape round-trip: [1, 384, 8, T] in, [1, 192, 32, T] out
//!   - residual + GLU + DConv + ConvTranspose2d compose without error
//!   - non-trivial output range
//!
//! Run: cargo test -p demucs-core-native --no-default-features --test hdec_forward -- --nocapture --ignored

use demucs_core_native::model::HDecLayer;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn hdec_layer3_forward_vocals() {
    let candidates = [
        "models/htdemucs_ft.safetensors",
        "../models/htdemucs_ft.safetensors",
        "../../models/htdemucs_ft.safetensors",
    ];
    let path = candidates
        .iter()
        .map(std::path::PathBuf::from)
        .find(|p| p.exists())
        .unwrap_or_else(|| std::path::PathBuf::from(candidates[0]));
    if !path.exists() {
        eprintln!("skipping: {} not found", path.display());
        return;
    }

    let store = WeightStore::load(&path).expect("load safetensors");
    let sig = "04573f0d"; // vocals
    // PyTorch decoder.0 is the deepest (closest to bottleneck): chin=384, chout=192.
    let layer = HDecLayer::from_store(&store, sig, "decoder.0", 384, 192, false)
        .expect("load HDecLayer decoder.0");

    // Sanity-check the loaded shapes.
    // rewrite: Conv2d(3,3) [2*384=768, 384, 3, 3]
    assert_eq!(layer.rewrite.out_ch, 768);
    assert_eq!(layer.rewrite.in_ch, 384);
    assert_eq!(layer.rewrite.kh, 3);
    assert_eq!(layer.rewrite.kw, 3);
    // dconv: 2 layers, compress=384/8=48
    assert_eq!(layer.dconv.layers.len(), 2);
    let dl0 = &layer.dconv.layers[0];
    assert_eq!(dl0.conv1.out_ch, 48);
    assert_eq!(dl0.conv1.in_ch, 384);
    // conv_tr: ConvTranspose2d [chin=384, chout=192, 8, 1] PyTorch layout
    assert_eq!(layer.conv_tr.in_ch, 384);
    assert_eq!(layer.conv_tr.out_ch, 192);
    assert_eq!(layer.conv_tr.kh, 8);
    assert_eq!(layer.conv_tr.kw, 1);

    // Construct a small input matching the bottleneck layout.
    // B=1, chin=384, Fr=8, T=4
    let b = 1;
    let chin = 384;
    let fr = 8;
    let t = 4;
    let x: Vec<f32> = (0..b * chin * fr * t)
        .map(|i| (i as f32 * 0.0017 - 0.5).sin() * 0.2)
        .collect();
    // Skip with the same shape (for the residual add).
    let skip: Vec<f32> = (0..b * chin * fr * t)
        .map(|i| (i as f32 * 0.0023 + 0.1).cos() * 0.1)
        .collect();
    // Target freq: ConvTranspose2d on Fr=8 with kH=8, stride=4, pad=2 produces
    // H_out = (8-1)*4 - 4 + 8 = 32. We trim to 32 (the natural target).
    let freq_target = 32;

    let (out, out_shape) = ops_cpu::hdec_layer_forward(
        &x,
        [b, chin, fr, t],
        &skip,
        [b, chin, fr, t],
        freq_target,
        &layer,
    );

    // Expected output: [1, 192, 32, 4]
    assert_eq!(out_shape, [b, 192, freq_target, t]);

    let max = out.iter().cloned().fold(0.0f32, f32::max);
    let min = out.iter().cloned().fold(0.0f32, f32::min);
    assert!(max > 0.0 || min < 0.0, "output should not be all zero");
    assert!(
        max.abs() < 100.0 && min.abs() < 100.0,
        "output magnitude too large: [{min}, {max}]"
    );
    let x_max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let x_min = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let s_max = skip.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let s_min = skip.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "✓ HDecLayer decoder.3 forward: [1,{},{},{}] → [1,{},{},{}], range [{:.4}, {:.4}] (input x=[{:.4}, {:.4}], skip=[{:.4}, {:.4}])",
        chin, fr, t,
        out_shape[1], out_shape[2], out_shape[3],
        min, max,
        x_min, x_max, s_min, s_max,
    );
}
