//! Cross-domain Transformer end-to-end forward test with real weights.
//!
//! Loads htdemucs_ft's vocals cross-domain transformer and runs a full forward
//! pass on a small synthetic input, verifying:
//!   - shape: freq [1, 384, Fr, T] in, [1, 384, Fr, T] out (via channel
//!     resample 384→512→384, but the bottleneck channels round-trip)
//!   - 5 layers run without error
//!   - output has non-trivial range (LayerScale initialised to 1, so the
//!     pass-through value of the input is recoverable in scale)
//!
//! Run: cargo test -p demucs-core-native --no-default-features --test transformer_forward -- --nocapture --ignored

use demucs_core_native::model::CrossDomainTransformer;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

#[test]
#[ignore]
fn cross_domain_transformer_full_5_layers_vocals() {
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
    let bottleneck_ch = 384; // CHANNELS * GROWTH^(DEPTH-1)
    let bottom_channels = 512; // 4-stem/ft: d_model = 512
    let ct = CrossDomainTransformer::from_store(&store, sig, bottleneck_ch, bottom_channels)
        .expect("load CrossDomainTransformer");

    // Sanity-check the model structure: 5 layers each, alternating self/cross.
    assert_eq!(ct.layers.len(), 5, "should have 5 freq transformer layers");
    assert_eq!(ct.layers_t.len(), 5, "should have 5 time transformer layers");
    let pattern = [true, false, true, false, true]; // self, cross, self, cross, self
    for (i, expected_self) in pattern.iter().enumerate() {
        let is_self = ct.layers[i].self_attn.is_some();
        let is_cross = ct.layers[i].cross_attn.is_some();
        assert_eq!(
            is_self, *expected_self,
            "freq layer {i} self/cross mismatch (expected self={expected_self})"
        );
        assert_eq!(is_cross, !*expected_self);
        let is_self_t = ct.layers_t[i].self_attn.is_some();
        let is_cross_t = ct.layers_t[i].cross_attn.is_some();
        assert_eq!(is_self_t, *expected_self, "time layer {i} self/cross mismatch");
        assert_eq!(is_cross_t, !*expected_self);
    }

    // Construct a tiny input matching a real bottleneck layout.
    // FreqEncoder output for [1, 4, 256, T] is [1, 384, 1, T] — bottleneck Fr=1.
    // Pick T=4 (frames) for a fast test.
    let fr = 1;
    let t = 4;
    let t2 = 4; // time-domain length
    let b = 1;

    // Synthetic freq input [1, 384, 1, 4]
    let freq: Vec<f32> = (0..b * bottleneck_ch * fr * t)
        .map(|i| (i as f32 * 0.001 - 0.05).sin() * 0.1)
        .collect();
    // Synthetic time input [1, 384, 4]
    let time: Vec<f32> = (0..b * bottleneck_ch * t2)
        .map(|i| (i as f32 * 0.002 + 0.3).cos() * 0.05)
        .collect();

    let (freq_out, freq_out_shape, time_out, time_out_shape) =
        ops_cpu::cross_domain_transformer_forward(
            &freq,
            [b, bottleneck_ch, fr, t],
            &time,
            [b, bottleneck_ch, t2],
            &ct,
        );

    // Verify output shapes round-trip back to the input channel dim.
    assert_eq!(freq_out_shape, [b, bottleneck_ch, fr, t], "freq output shape");
    assert_eq!(time_out_shape, [b, bottleneck_ch, t2], "time output shape");

    // Verify outputs are non-trivial.
    let f_max = freq_out.iter().cloned().fold(0.0f32, f32::max);
    let f_min = freq_out.iter().cloned().fold(0.0f32, f32::min);
    let t_max = time_out.iter().cloned().fold(0.0f32, f32::max);
    let t_min = time_out.iter().cloned().fold(0.0f32, f32::min);
    println!(
        "✓ CrossDomainTransformer full 5-layer forward:\n  freq [1,{},{},{}] → [1,{},{},{}], range [{:.4}, {:.4}]\n  time [1,{},{}] → [1,{},{}], range [{:.4}, {:.4}]",
        bottleneck_ch, fr, t,
        freq_out_shape[1], freq_out_shape[2], freq_out_shape[3],
        f_min, f_max,
        bottleneck_ch, t2,
        time_out_shape[1], time_out_shape[2],
        t_min, t_max,
    );
    assert!(f_max > 0.0 || f_min < 0.0, "freq output should not be all zero");
    assert!(t_max > 0.0 || t_min < 0.0, "time output should not be all zero");
    // Sanity-bound: outputs shouldn't explode for a small input.
    assert!(
        f_max.abs() < 100.0 && f_min.abs() < 100.0,
        "freq output magnitude too large: [{f_min}, {f_max}]"
    );
    assert!(
        t_max.abs() < 100.0 && t_min.abs() < 100.0,
        "time output magnitude too large: [{t_min}, {t_max}]"
    );
}
