//! Single-segment comparison test: take 15s audio, truncate to ≤ 343980
//! samples @ 44100Hz, run native without chunking, compare with the burn
//! reference's first ~7.8 seconds.
//!
//! If single-segment matches the reference well but full chunked run does
//! not, the bug is in the chunked path. If single-segment also has a big
//! error, the bug is in the model itself.

use std::path::PathBuf;

use demucs_core_native::dsp::resample::resample_channel;
use demucs_core_native::dsp::stft::Stft;
use demucs_core_native::metadata::HTDEMUCS_FT_ID;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;
use demucs_core_native::{Demucs, LoadOptions, ModelVariant, StemSelection, TRAINING_LENGTH};

use hound::{SampleFormat, WavReader};

fn find_path(candidates: &[&str]) -> PathBuf {
    candidates
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from(candidates[0]))
}

fn read_wav_stereo(path: &PathBuf) -> (Vec<f32>, Vec<f32>, u32) {
    let mut reader = WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 2, "expected stereo");
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / (1i32 << (bits - 1)) as f32)
                .collect()
        }
    };
    let mut left = Vec::with_capacity(samples.len() / 2);
    let mut right = Vec::with_capacity(samples.len() / 2);
    for frame in samples.chunks_exact(2) {
        left.push(frame[0]);
        right.push(frame[1]);
    }
    (left, right, spec.sample_rate)
}

/// Single-segment run, no chunking. Truncate input to ≤ TRAINING_LENGTH.
#[test]
#[ignore]
fn end_to_end_15s_singleseg_first_chunk_vs_burn_ref() {
    let model_path = find_path(&[
        "models/htdemucs_ft.safetensors",
        "../models/htdemucs_ft.safetensors",
        "../../models/htdemucs_ft.safetensors",
    ]);
    if !model_path.exists() {
        eprintln!("skipping: model not found");
        return;
    }
    let audio_path = find_path(&["tests/15s.wav", "../tests/15s.wav", "../../tests/15s.wav"]);
    let ref_path = find_path(&[
        "tests/reference_burn/vocals.wav",
        "../tests/reference_burn/vocals.wav",
        "../../tests/reference_burn/vocals.wav",
    ]);
    if !audio_path.exists() || !ref_path.exists() {
        eprintln!("skipping: audio or ref missing");
        return;
    }

    let bytes = std::fs::read(&model_path).expect("read model file");
    let opts = LoadOptions {
        variant: ModelVariant::FineTuned,
        stems: StemSelection::Some(vec![demucs_core_native::StemId::Vocals]),
    };
    let demucs = Demucs::from_bytes(&bytes, opts.clone(), demucs_core_native::Backend::Cpu)
        .expect("load Demucs");

    let (left, right, sample_rate) = read_wav_stereo(&audio_path);

    // ─── Single segment: resample to 44100, truncate to first TRAINING_LENGTH
    // samples. This forces the `n_samples <= TRAINING_LENGTH` fast path in
    // CpuEngine::separate.
    let left_44k =
        resample_channel(&left, sample_rate, 44100).expect("resample left");
    let right_44k =
        resample_channel(&right, sample_rate, 44100).expect("resample right");
    let n_take = left_44k.len().min(TRAINING_LENGTH);
    let left_44k_trunc = &left_44k[..n_take];
    let right_44k_trunc = &right_44k[..n_take];
    eprintln!(
        "Single segment: input {} samples @ 44100, truncated to {} (TRAINING_LENGTH={})",
        left_44k.len(),
        n_take,
        TRAINING_LENGTH
    );

    let start = std::time::Instant::now();
    let stems = demucs
        .separate(left_44k_trunc, right_44k_trunc, 44100)
        .expect("separate");
    let elapsed = start.elapsed();
    eprintln!(
        "Single-segment inference took {:.2}s (RTFx {:.3}×)",
        elapsed.as_secs_f64(),
        n_take as f64 / 44100.0 / elapsed.as_secs_f64()
    );

    let vocals = stems
        .iter()
        .find(|s| s.id == demucs_core_native::StemId::Vocals)
        .expect("vocals missing");

    // ─── Reference: resample vocals back to 48000 to match the ref wav
    let vocals_at_ref_sr = resample_channel(&vocals.left, 44100, 48000).expect("resample vocals back");
    let vocals_r_at_ref_sr = resample_channel(&vocals.right, 44100, 48000).expect("resample vocals back");

    // Load burn ref.
    let (ref_l, ref_r, ref_sr) = read_wav_stereo(&ref_path);
    assert_eq!(ref_sr, 48000);

    // Compare only the first n_take_in_ref samples (= n_take * 48000 / 44100).
    let n_compare_in_ref = (n_take as f64 * 48000.0 / 44100.0) as usize;
    let n = vocals_at_ref_sr.len().min(ref_l.len()).min(n_compare_in_ref);
    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;
    let mut sum_diff = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    for i in 0..n {
        let d_l = (vocals_at_ref_sr[i] - ref_l[i]).abs();
        let d_r = (vocals_r_at_ref_sr[i] - ref_r[i]).abs();
        let d = d_l.max(d_r);
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
        sum_diff += d_l.max(d_r) as f64;
        sum_sq_diff += (d_l.max(d_r) as f64).powi(2);
        sum_sq_ref += (ref_l[i].abs().max(ref_r[i].abs()) as f64).powi(2);
    }
    eprintln!(
        "Single-segment vocals (first {:.2}s = {} samples): max_abs_diff={:.6e} at idx={} (native={:.4}, ref={:.4})",
        n as f64 / 48000.0,
        n,
        max_diff,
        max_idx,
        vocals_at_ref_sr[max_idx].abs().max(vocals_r_at_ref_sr[max_idx].abs()),
        ref_l[max_idx].abs().max(ref_r[max_idx].abs()),
    );
    eprintln!(
        "  rms_diff={:.6e}, rms_ref={:.4}, mean_diff={:.6e}",
        (sum_sq_diff / n as f64).sqrt(),
        (sum_sq_ref / n as f64).sqrt(),
        sum_diff / n as f64
    );

    // Diagnostic: print diff at multiple time points
    for &frac in &[0.0f64, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let i = (frac * n as f64) as usize;
        let d_l = (vocals_at_ref_sr[i] - ref_l[i]).abs();
        let d_r = (vocals_r_at_ref_sr[i] - ref_r[i]).abs();
        eprintln!(
            "    t={:.2}s idx={}: native L={:.4} R={:.4}, ref L={:.4} R={:.4}, diff L={:.4} R={:.4}",
            i as f64 / 48000.0, i,
            vocals_at_ref_sr[i], vocals_r_at_ref_sr[i],
            ref_l[i], ref_r[i], d_l, d_r,
        );
    }
}