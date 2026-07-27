//! TEncLayer + TDecLayer end-to-end forward tests with real weights.
//!
//! Loads htdemucs_ft's vocals time encoder.0 and time decoder.0 (the deepest
//! time-decoder layer: 384 → 192 channels, time upsample 4×) and runs
//! forward passes on synthetic inputs.
//!
//! Run: cargo test -p demucs-core-native --no-default-features --test tenc_tdec_forward -- --nocapture --ignored

use demucs_core_native::model::{TDecLayer, TEncLayer};
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn tenc_layer0_forward_vocals() {
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
    let layer = TEncLayer::from_store(&store, sig, "tencoder.0", 2, 48)
        .expect("load TEncLayer tencoder.0");

    // Sanity-check the loaded shapes.
    assert_eq!(layer.conv.out_ch, 48, "tenc conv out_ch (chout)");
    assert_eq!(layer.conv.in_ch, 2, "tenc conv in_ch (chin, stereo)");
    assert_eq!(layer.conv.k, 8, "tenc conv kernel");
    assert_eq!(layer.rewrite.out_ch, 96, "tenc rewrite out (2*chout)");
    assert_eq!(layer.rewrite.in_ch, 48, "tenc rewrite in (chout)");
    assert_eq!(layer.rewrite.k, 1, "tenc rewrite kernel");

    // Construct a small input. We use a length that is a multiple of 4 so
    // padding is no-op (simplest test case).
    let b = 1;
    let chin = 2;
    let t = 16; // 16 samples, multiple of 4 → no right-pad
    let x: Vec<f32> = (0..b * chin * t)
        .map(|i| (i as f32 * 0.07 - 0.4).sin() * 0.3)
        .collect();

    let (out, out_shape) = ops_cpu::tenc_layer_forward(&x, [b, chin, t], &layer);

    // T_out = T / 4 = 16 / 4 = 4 (no right-pad because 16 % 4 == 0).
    assert_eq!(out_shape, [b, 48, t / 4]);
    assert_eq!(out_shape[2], 4, "T_out should be 4");

    let max = out.iter().cloned().fold(0.0f32, f32::max);
    let min = out.iter().cloned().fold(0.0f32, f32::min);
    assert!(max > 0.0 || min < 0.0, "output should not be all zero");
    assert!(
        max.abs() < 100.0 && min.abs() < 100.0,
        "output magnitude too large: [{min}, {max}]"
    );
    println!(
        "✓ TEncLayer tencoder.0 forward: [1,{},{}] → [1,{},{}], range [{:.4}, {:.4}]",
        chin, t,
        out_shape[1], out_shape[2],
        min, max,
    );
}

#[test]
#[ignore]
fn tenc_layer0_forward_vocals_with_right_pad() {
    // Verify the right-pad branch: T=10, not divisible by 4 → pad to 12.
    // T_out = 12 / 4 = 3.
    let path = std::path::PathBuf::from("../models/htdemucs_ft.safetensors");
    if !path.exists() {
        eprintln!("skipping");
        return;
    }
    let store = WeightStore::load(&path).expect("load safetensors");
    let sig = "04573f0d";
    let layer = TEncLayer::from_store(&store, sig, "tencoder.0", 2, 48)
        .expect("load TEncLayer tencoder.0");

    let t = 10;
    let x: Vec<f32> = (0..2 * t).map(|i| i as f32 * 0.01).collect();
    let (out, out_shape) = ops_cpu::tenc_layer_forward(&x, [1, 2, t], &layer);
    // T=10 → pad to 12 → T_out = 3.
    assert_eq!(out_shape, [1, 48, 3]);
    println!(
        "✓ TEncLayer tencoder.0 forward (right-pad): [1,2,10] → [1,48,3], len={}",
        out.len()
    );
}

#[test]
#[ignore]
fn tdec_layer0_forward_vocals() {
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
    let sig = "04573f0d";
    // PyTorch tdecoder.0 is the deepest time decoder: chin=384, chout=192, last=false.
    let layer = TDecLayer::from_store(&store, sig, "tdecoder.0", 384, 192, false)
        .expect("load TDecLayer tdecoder.0");

    assert_eq!(layer.rewrite.out_ch, 768, "tdec rewrite out (2*chin)");
    assert_eq!(layer.rewrite.in_ch, 384, "tdec rewrite in (chin)");
    assert_eq!(layer.rewrite.k, 3, "tdec rewrite kernel");
    // conv_tr: ConvTranspose1d [chin=384, chout=192, 8] after the swap loader.
    assert_eq!(layer.conv_tr.in_ch, 384, "tdec conv_tr in (chin)");
    assert_eq!(layer.conv_tr.out_ch, 192, "tdec conv_tr out (chout)");
    assert_eq!(layer.conv_tr.k, 8, "tdec conv_tr kernel");

    // Input: [1, 384, T_in] where T_in corresponds to the bottleneck time dim
    // after 4 levels of stride-4 conv. T_in=4 for a fast test.
    // T_out = 4 * 4 = 16 (4× upsample).
    let b = 1;
    let chin = 384;
    let t_in = 4;
    let x: Vec<f32> = (0..b * chin * t_in)
        .map(|i| (i as f32 * 0.0013 - 0.2).sin() * 0.1)
        .collect();
    // Skip with same shape.
    let skip: Vec<f32> = (0..b * chin * t_in)
        .map(|i| (i as f32 * 0.0027 + 0.1).cos() * 0.05)
        .collect();

    let (out, out_shape) = ops_cpu::tdec_layer_forward(
        &x,
        [b, chin, t_in],
        &skip,
        [b, chin, t_in],
        t_in * 4, // 4× upsample target
        &layer,
    );

    // Expected output: [1, 192, 16].
    assert_eq!(out_shape, [b, 192, t_in * 4]);

    let max = out.iter().cloned().fold(0.0f32, f32::max);
    let min = out.iter().cloned().fold(0.0f32, f32::min);
    assert!(max > 0.0 || min < 0.0, "output should not be all zero");
    assert!(
        max.abs() < 100.0 && min.abs() < 100.0,
        "output magnitude too large: [{min}, {max}]"
    );
    println!(
        "✓ TDecLayer tdecoder.0 forward: [1,{},{}] → [1,{},{}], range [{:.4}, {:.4}]",
        chin, t_in,
        out_shape[1], out_shape[2],
        min, max,
    );
}
