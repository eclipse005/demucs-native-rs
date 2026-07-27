//! FreqEncoder end-to-end forward test with real weights.
//!
//! Loads htdemucs_ft's vocals frequency encoder (4 HEncLayers + freq_emb)
//! and runs a forward pass on a CaC-shaped input, verifying:
//!   - channel progression 4 → 48 → 96 → 192 → 384
//!   - freq bins downsampled by 4^4 = 256
//!   - 4 skip connections saved
//!
//! Run: cargo test -p demucs-core-native --no-default-features --test freq_encoder -- --nocapture --ignored

use demucs_core_native::{model::FreqEncoder, ops_cpu, weights::WeightStore};

#[test]
#[ignore]
fn freq_encoder_forward_full_4_layers() {
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
    let enc = FreqEncoder::from_store(&store, sig).expect("load FreqEncoder");

    // CaC input: [1, 4, 256, 4]
    // (B=1, C=4 CaC, Fr=256 so after 4× stride-4 → 256/256=1, T=4)
    let b = 1;
    let c_in = 4;
    let fr = 256;
    let t = 4;
    // Use small random-ish values to avoid overflow through deep layers.
    let x: Vec<f32> = (0..b * c_in * fr * t)
        .map(|i| {
            let v = (i as f32 * 0.123).sin() * 0.1;
            v
        })
        .collect();

    let (out, out_shape, skips) = ops_cpu::freq_encoder_forward(&x, [b, c_in, fr, t], &enc);

    // Verify output shape: channels 384, freq 256/256=1, time 4
    let [ob, oc, ofr, ot] = out_shape;
    assert_eq!(ob, 1, "batch");
    assert_eq!(oc, 384, "output channels (4 layers: 48→96→192→384)");
    assert_eq!(ofr, 1, "freq bins (256 / 4^4 = 1)");
    assert_eq!(ot, 4, "time frames preserved");

    // Verify 4 skips saved, each with correct channels.
    assert_eq!(skips.len(), 4, "should have 4 skip connections");
    let expected_ch = [48usize, 96, 192, 384];
    let expected_fr = [64usize, 16, 4, 1]; // 256→64→16→4→1
    for (i, (skip_data, skip_shape)) in skips.iter().enumerate() {
        let [_, sc, sf, st] = *skip_shape;
        assert_eq!(sc, expected_ch[i], "skip[{}] channels", i);
        assert_eq!(sf, expected_fr[i], "skip[{}] freq bins", i);
        assert_eq!(st, 4, "skip[{}] time frames", i);
        // Skip data should be non-trivial.
        let max = skip_data.iter().cloned().fold(0.0f32, f32::max);
        assert!(max > 0.0, "skip[{}] should have non-zero values", i);
    }

    // Output range check.
    let max = out.iter().cloned().fold(0.0f32, f32::max);
    let min = out.iter().cloned().fold(0.0f32, f32::min);
    assert!(max > 0.0 && min < 0.0, "output should have sign variety");

    println!(
        "✓ FreqEncoder 4-layer forward: [{},{},{},{}] → [{},{},{},{}], {} skips",
        b, c_in, fr, t, ob, oc, ofr, ot, skips.len()
    );
    println!(
        "  skip channels: {:?}, skip freqs: {:?}",
        expected_ch, expected_fr
    );
    println!("  output range: [{:.4}, {:.4}]", min, max);
}
