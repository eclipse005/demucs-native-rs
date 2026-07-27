//! Per-layer diagnostic: run htdemucs_forward on real audio and dump
//! min/max/mean/std of every intermediate tensor. Helps locate where the
//! e2e numerical gap originates (no burn comparison — just shows the growth
//! pattern).

use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;
use demucs_core_native::{LoadOptions, ModelVariant, StemSelection};

fn stats(name: &str, x: &[f32]) {
    if x.is_empty() {
        eprintln!("{name}: empty");
        return;
    }
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut nan_count = 0;
    for &v in x {
        if v.is_nan() || v.is_infinite() {
            nan_count += 1;
            continue;
        }
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
        sum += v as f64;
        sum_sq += (v as f64) * (v as f64);
    }
    let n = x.len() as f64;
    let mean = sum / n;
    let var = (sum_sq / n) - mean * mean;
    eprintln!(
        "{name}: shape={} min={:.4} max={:.4} mean={:.4} std={:.4} (NaN/Inf: {})",
        x.len(),
        min_v,
        max_v,
        mean,
        var.sqrt(),
        nan_count
    );
}

#[test]
#[ignore]
fn probe_per_layer_diagnostics() {
    let model_path = std::path::PathBuf::from("../models/htdemucs_ft.safetensors");
    let audio_path = std::path::PathBuf::from("../tests/15s.wav");
    if !model_path.exists() || !audio_path.exists() {
        eprintln!("skipping: missing model or audio");
        return;
    }

    // ─── Read audio ────────────────────────────────────────────────────
    let mut reader = hound::WavReader::open(&audio_path).expect("open wav");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / (1i32 << (bits - 1)) as f32)
                .collect()
        }
    };
    let sr = spec.sample_rate;
    let mut left = Vec::with_capacity(samples.len() / 2);
    let mut right = Vec::with_capacity(samples.len() / 2);
    for frame in samples.chunks_exact(2) {
        left.push(frame[0]);
        right.push(frame[1]);
    }

    // ─── Resample to 44100 (toy impl: just take every Nth sample — we just
    // want *any* input that exercises the full pipeline).
    if sr != 44100 {
        let ratio = sr as f32 / 44100.0;
        let n_44k = (left.len() as f32 / ratio) as usize;
        left = (0..n_44k).map(|i| left[(i as f32 * ratio) as usize]).collect();
        right = (0..n_44k).map(|i| right[(i as f32 * ratio) as usize]).collect();
    }
    let n_samples = left.len().min(demucs_core_native::TRAINING_LENGTH);
    left.truncate(n_samples);
    right.truncate(n_samples);
    // Pad to TRAINING_LENGTH.
    left.resize(demucs_core_native::TRAINING_LENGTH, 0.0);
    right.resize(demucs_core_native::TRAINING_LENGTH, 0.0);

    // ─── Build freq + time tensors ─────────────────────────────────────
    let mut stft = demucs_core_native::dsp::stft::Stft::new(
        demucs_core_native::N_FFT,
        demucs_core_native::HOP_LENGTH,
    );
    let left_spec = stft.forward(&left).expect("stft left");
    let right_spec = stft.forward(&right).expect("stft right");
    let n_bins = demucs_core_native::N_FFT / 2;
    let n_frames = left_spec.len() / n_bins;
    let left_cac = demucs_core_native::dsp::cac::stft_to_cac(&left_spec, demucs_core_native::N_FFT);
    let right_cac =
        demucs_core_native::dsp::cac::stft_to_cac(&right_spec, demucs_core_native::N_FFT);
    // Stack into [1, 4, n_bins, n_frames] freq tensor.
    let mut freq = vec![0.0f32; 4 * n_bins * n_frames];
    let bin_frames = n_bins * n_frames;
    for bin in 0..n_bins {
        for frame in 0..n_frames {
            freq[0 * bin_frames + bin * n_frames + frame] =
                left_cac[0 * bin_frames + bin * n_frames + frame];
            freq[1 * bin_frames + bin * n_frames + frame] =
                left_cac[1 * bin_frames + bin * n_frames + frame];
            freq[2 * bin_frames + bin * n_frames + frame] =
                right_cac[0 * bin_frames + bin * n_frames + frame];
            freq[3 * bin_frames + bin * n_frames + frame] =
                right_cac[1 * bin_frames + bin * n_frames + frame];
        }
    }
    let time = {
        let mut t = vec![0.0f32; 2 * demucs_core_native::TRAINING_LENGTH];
        t[..demucs_core_native::TRAINING_LENGTH].copy_from_slice(&left);
        t[demucs_core_native::TRAINING_LENGTH..].copy_from_slice(&right);
        t
    };
    stats("INPUT freq", &freq);
    stats("INPUT time", &time);

    // ─── Normalize ─────────────────────────────────────────────────────
    let (mut freq_n, freq_shape, freq_mean, _, freq_std, _) =
        ops_cpu::normalize_freq(&freq, [1, 4, n_bins, n_frames]);
    stats("NORMALIZED freq", &freq_n);
    eprintln!("  freq_mean={:.4} freq_std={:.4}", freq_mean[0], freq_std[0]);
    let (mut time_n, time_shape, time_mean, _, time_std, _) =
        ops_cpu::normalize_time(&time, [1, 2, demucs_core_native::TRAINING_LENGTH]);
    stats("NORMALIZED time", &time_n);
    eprintln!("  time_mean={:.4} time_std={:.4}", time_mean[0], time_std[0]);

    // ─── Load model ────────────────────────────────────────────────────
    let store = WeightStore::load(&model_path).expect("load model");
    let model = HTDemucs::from_store(&store, "04573f0d", 4, 512).expect("load HTDemucs");

    // ─── Step through encoder chain (collect skips) ────────────────────
    let mut freq = freq_n;
    let mut freq_shape = freq_shape;
    let depth = model.encoders.len();
    let mut freq_skips: Vec<(Vec<f32>, [usize; 4])> = Vec::with_capacity(depth);

    for i in 0..depth {
        let (out, out_shape) =
            ops_cpu::henc_layer_forward(&freq, freq_shape, &model.encoders[i]);
        stats(&format!("AFTER freq encoder[{}]", i), &out);
        freq = out;
        freq_shape = out_shape;

        // Apply freq_emb after layer 0 (matches burn).
        if i == 0 {
            let [b, c, fr_dim, t_dim] = freq_shape;
            let emb = &model.freq_emb;
            for bi in 0..b {
                for ci in 0..c {
                    for fi in 0..fr_dim {
                        let emb_val = emb.data[fi * emb.dim + ci] * 0.2;
                        for ti in 0..t_dim {
                            freq[((bi * c + ci) * fr_dim + fi) * t_dim + ti] += emb_val;
                        }
                    }
                }
            }
            stats("AFTER freq_emb", &freq);
        }
        freq_skips.push((freq.clone(), freq_shape));
    }
    stats("BEFORE Transformer (freq)", &freq);

    // ─── Time encoder chain (collect skips + time_lengths) ───────────
    let mut time = time_n;
    let mut time_shape = time_shape;
    let mut time_skips: Vec<(Vec<f32>, [usize; 3])> = Vec::with_capacity(depth);
    let mut time_lengths: Vec<usize> = Vec::with_capacity(depth);
    for i in 0..depth {
        time_lengths.push(time_shape[2]);
        let (out, out_shape) =
            ops_cpu::tenc_layer_forward(&time, time_shape, &model.tencoders[i]);
        stats(&format!("AFTER time encoder[{}]", i), &out);
        time = out;
        time_shape = out_shape;
        time_skips.push((time.clone(), time_shape));
    }
    stats("BEFORE Transformer (time)", &time);

    // ─── Transformer (per-layer trace via traced helper) ──────────────
    let (mut freq, mut freq_shape, mut time, mut time_shape, trace) =
        ops_cpu::cross_domain_transformer_forward_traced(
            &freq,
            freq_shape,
            &time,
            time_shape,
            &model.crosstransformer,
        );
    for (i, (f, t)) in trace.iter().enumerate() {
        stats(&format!("AFTER Transformer layer {i} (freq)"), f);
        stats(&format!("AFTER Transformer layer {i} (time)"), t);
    }
    stats("AFTER Transformer final (freq)", &freq);
    stats("AFTER Transformer final (time)", &time);

    // ─── Denormalize ──────────────────────────────────────────────────
    // First run decoder chains to dump intermediate shapes.
    let freq_dims: Vec<usize> = freq_skips.iter().map(|(_, s)| s[2]).collect();
    eprintln!("freq_dims = {:?}", freq_dims);
    for i in 0..depth {
        let (skip, skip_shape) = freq_skips.pop().expect("freq skip stack");
        let target = if i + 1 < freq_dims.len() {
            freq_dims[freq_dims.len() - 2 - i]
        } else {
            demucs_core_native::N_FFT / 2
        };
        eprintln!(
            "  [freq decoder {i}] in_shape={:?} skip_shape={:?} target={} (dec.0 conv_tr.in_ch={})",
            freq_shape, skip_shape, target, model.decoders[i].conv_tr.in_ch
        );
        let (out, out_shape) = ops_cpu::hdec_layer_forward(
            &freq,
            freq_shape,
            &skip,
            skip_shape,
            target,
            &model.decoders[i],
        );
        eprintln!("  [freq decoder {i}] out_shape={:?}", out_shape);
        stats(&format!("  [freq decoder {i}] out"), &out);
        freq = out;
        freq_shape = out_shape;
    }

    // ─── Per-stage trace of freq decoder 0 (the deep one) ──────────────
    // We need the transformer's final freq output, but it was already moved
    // into `freq` and consumed. The simplest path is to re-run forward up
    // through the transformer. We do this lazily — only if it fits in time.
    // For now, just dump a quick sanity for decoder 0 with a fresh input by
    // re-running everything. (Skipped to save 25 min; left here as a comment
    // for future debug.)
    eprintln!("\n(freq decoder 0 per-stage trace omitted — see _probe_layer4.rs probe_hdec_layer0_isolated_with_dummy)");
    for i in 0..depth {
        let (skip, skip_shape) = time_skips.pop().expect("time skip stack");
        let target = time_lengths[time_lengths.len() - 1 - i];
        eprintln!(
            "  [time decoder {i}] in_shape={:?} skip_shape={:?} target={} (expected ch={})",
            time_shape, skip_shape, target, model.tdecoders[i].conv_tr.in_ch
        );
        let (out, out_shape) = ops_cpu::tdec_layer_forward(
            &time,
            time_shape,
            &skip,
            skip_shape,
            target,
            &model.tdecoders[i],
        );
        eprintln!("  [time decoder {i}] out_shape={:?}", out_shape);
        stats(&format!("  [time decoder {i}] out"), &out);
        time = out;
        time_shape = out_shape;
    }

    // ─── Denormalize ──────────────────────────────────────────────────
    ops_cpu::denormalize_freq(&mut freq, freq_shape, &freq_mean, &freq_std);
    ops_cpu::denormalize_time(&mut time, time_shape, &time_mean, &time_std);
    stats("AFTER denormalize (freq)", &freq);
    stats("AFTER denormalize (time)", &time);

    // ─── 7. Run extract_stems to verify the post-extract WAV magnitudes ──
    // Note: synthetic 1s audio was resampled (actually the toy in this
    // test), so n_samples=343980 (TRAINING_LENGTH). For the burn baseline
    // comparison, we want to see what the WAV looks like after iSTFT + time
    // add (not the freq_out / time_out values directly).
    eprintln!("\n--- EXTRACT STEMS ---");
    eprintln!(
        "freq_out shape: {:?}, len={}",
        freq_shape,
        freq.len()
    );
    eprintln!(
        "time_out shape: {:?}, len={}",
        time_shape,
        time.len()
    );

    // Quick test: run extract_stems, get vocals.left, check range.
    let mut stft = demucs_core_native::dsp::stft::Stft::new(
        demucs_core_native::N_FFT,
        demucs_core_native::HOP_LENGTH,
    );
    let n_frames = 343980usize.div_ceil(demucs_core_native::HOP_LENGTH);
    let padded_len = demucs_core_native::TRAINING_LENGTH;
    let stems = ops_cpu::extract_stems(
        &freq,
        freq_shape,
        &time,
        time_shape,
        n_frames,
        padded_len,
        padded_len,
        &mut stft,
    );
    for s in &stems {
        let max = s.left.iter().cloned().fold(0.0f32, f32::max);
        let min = s.left.iter().cloned().fold(0.0f32, f32::min);
        let rms = (s.left.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()
            / s.left.len() as f64)
            .sqrt();
        eprintln!(
            "stem {:?} left: range=[{:.4}, {:.4}] rms={:.4} (n={})",
            s.id, min, max, rms, s.left.len()
        );
    }
}
