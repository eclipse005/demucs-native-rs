//! HEncLayer end-to-end forward test with real weights.
//!
//! Loads htdemucs_ft's vocals encoder.0 and runs a forward pass on a small
//! random input, verifying the output shape matches the burn reference's
//! expected channel progression (4 → 48 CaC → 48 after encoder layer 0).
//!
//! Run: cargo test -p demucs-core-native --no-default-features --test henc_forward -- --nocapture --ignored

use demucs_core_native::{model::HEncLayer, ops_cpu, weights::WeightStore};

#[test]
#[ignore]
fn henc_layer0_forward_shape_and_range() {
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
    let layer = HEncLayer::from_store(&store, sig, "encoder.0").expect("load HEncLayer");

    // Construct a small CaC-shaped input: [1, 4, 16, 4]
    // (B=1, C_in=4 CaC, Fr=16 freq bins, T=4 time frames)
    let b = 1;
    let c_in = 4;
    let fr = 16;
    let t = 4;
    let x: Vec<f32> = (0..b * c_in * fr * t)
        .map(|i| (i as f32 * 0.01 - 0.3))
        .collect();

    let (out, out_shape) = ops_cpu::henc_layer_forward(&x, [b, c_in, fr, t], &layer);

    // Expected: Fr_out = (16 + 2*2 - 8)/4 + 1 = 12/4 + 1 = 4
    // Output channels = 48 (C_out of encoder layer 0)
    let [ob, oc, ofr, ot] = out_shape;
    assert_eq!(ob, 1, "batch");
    assert_eq!(oc, 48, "output channels (should be 48)");
    assert_eq!(ofr, 4, "output freq bins (16→4 with stride 4)");
    assert_eq!(ot, 4, "time frames preserved");

    // Range check: output should have sign variety and reasonable magnitude.
    let max = out.iter().cloned().fold(0.0f32, f32::max);
    let min = out.iter().cloned().fold(0.0f32, f32::min);
    assert!(max > 0.0 && min < 0.0, "output should have sign variety");
    assert!(
        max.abs() < 100.0 && min.abs() < 100.0,
        "output magnitude should be reasonable, got [{}, {}]",
        min,
        max
    );

    println!(
        "✓ HEncLayer encoder.0 forward: [{},{},{},{}] → [{},{},{},{}], range [{:.4}, {:.4}]",
        b, c_in, fr, t, ob, oc, ofr, ot, min, max
    );
}
