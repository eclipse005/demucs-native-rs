//! Weight loading integration test — loads a real htdemucs_ft safetensors
//! file and validates that HEncLayer weights parse with correct shapes.
//!
//! Run: cargo test -p demucs-core-native --no-default-features --test weight_loading -- --nocapture --ignored

use demucs_core_native::model::HEncLayer;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore] // requires models/htdemucs_ft.safetensors on disk
fn load_vocals_encoder_layer0_shapes() {
    // Search for the model file: tests run from the crate dir, but the model
    // lives at the workspace root (../../models/).
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

    // Vocals is the 4th stem → signature index 3 → "04573f0d".
    let sig = "04573f0d";
    assert!(
        store.signature(sig).is_some(),
        "signature {sig} should be present"
    );

    let layer = HEncLayer::from_store(&store, sig, "encoder.0").expect("load HEncLayer");

    // conv: [48, 4, 8, 1] (chout=48, chin=4 CaC, kH=8, kW=1)
    assert_eq!(layer.conv.out_ch, 48, "conv out_ch");
    assert_eq!(layer.conv.in_ch, 4, "conv in_ch (CaC = 4)");
    assert_eq!(layer.conv.kh, 8, "conv kH");
    assert_eq!(layer.conv.kw, 1, "conv kW");
    assert_eq!(layer.conv.data.len(), 48 * 4 * 8 * 1, "conv data len");
    assert_eq!(layer.conv_bias.len, 48, "conv bias len");

    // DConv: 2 layers
    assert_eq!(layer.dconv.layers.len(), 2, "dconv depth");

    // DConv layer 0: conv1 [6, 48, 3], norm1 [6], conv2 [96, 6, 1], norm2 [96], scale [48]
    let dl0 = &layer.dconv.layers[0];
    assert_eq!(dl0.conv1.out_ch, 6, "dconv0 conv1 out (compress=48/8)");
    assert_eq!(dl0.conv1.in_ch, 48, "dconv0 conv1 in");
    assert_eq!(dl0.conv1.k, 3, "dconv0 conv1 kernel");
    assert_eq!(dl0.norm1.num_channels, 6, "dconv0 norm1 channels");
    assert_eq!(dl0.conv2.out_ch, 96, "dconv0 conv2 out (2*48)");
    assert_eq!(dl0.conv2.in_ch, 6, "dconv0 conv2 in");
    assert_eq!(dl0.conv2.k, 1, "dconv0 conv2 kernel");
    assert_eq!(dl0.norm2.num_channels, 96, "dconv0 norm2 channels");
    assert_eq!(dl0.scale.scale.len(), 48, "dconv0 scale len");

    // DConv layer 1: same shapes but dilation=2 (we don't store dilation in weights,
    // it's applied at forward time).
    let dl1 = &layer.dconv.layers[1];
    assert_eq!(dl1.conv1.k, 3, "dconv1 conv1 kernel");

    // rewrite: [96, 48, 1, 1] (2*chout, chout, 1, 1)
    assert_eq!(layer.rewrite.out_ch, 96, "rewrite out (2*48)");
    assert_eq!(layer.rewrite.in_ch, 48, "rewrite in");
    assert_eq!(layer.rewrite.kh, 1, "rewrite kH");
    assert_eq!(layer.rewrite.kw, 1, "rewrite kW");
    assert_eq!(layer.rewrite_bias.len, 96, "rewrite bias len");

    println!("✓ HEncLayer encoder.0 shapes all correct for {}", sig);

    // Sanity-check a few weight values are non-trivial (not all zeros).
    let conv_max = layer.conv.data.iter().cloned().fold(0.0f32, f32::max);
    let conv_min = layer.conv.data.iter().cloned().fold(0.0f32, f32::min);
    assert!(conv_max > 0.0 && conv_min < 0.0, "conv weights should have sign variety");
    println!("  conv.weight range: [{:.4}, {:.4}]", conv_min, conv_max);
}
