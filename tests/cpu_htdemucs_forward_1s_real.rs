//! CPU htdemucs_forward diagnostic on 1s real audio.
//! Goal: locate which component (freq or time) is 0 in the CPU
//! 1s real audio path. Bypasses iSTFT to keep iteration fast.
//!
//! Run: cargo test -p demucs-core-native --no-default-features --test
//!      cpu_htdemucs_forward_1s_real -- --nocapture --ignored

#![cfg(not(feature = "cuda"))]

use std::path::PathBuf;

use demucs_core_native::dsp::resample::resample_channel;
use demucs_core_native::dsp::stft::Stft;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;
use demucs_core_native::model::HTDemucs;

use hound::{SampleFormat, WavReader};

fn find_path(candidates: &[&str]) -> PathBuf {
    candidates.iter().map(PathBuf::from).find(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from(candidates[0]))
}

#[test]
#[ignore]
fn cpu_htdemucs_forward_1s_real_diagnostic() {
    let model_path = find_path(&[
        "models/htdemucs_ft.safetensors",
        "../models/htdemucs_ft.safetensors",
        "../../models/htdemucs_ft.safetensors",
    ]);
    if !model_path.exists() { eprintln!("skipping: model"); return; }
    let audio_path = find_path(&["tests/15s.wav", "../tests/15s.wav", "../../tests/15s.wav"]);
    if !audio_path.exists() { eprintln!("skipping: audio"); return; }

    let mut reader = WavReader::open(&audio_path).expect("open wav");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / (1i32 << (bits - 1)) as f32).collect()
        }
    };
    let mut left = vec![]; let mut right = vec![];
    for frame in samples.chunks_exact(2) { left.push(frame[0]); right.push(frame[1]); }
    let left_44k = resample_channel(&left, 48000, 44100).unwrap();
    let right_44k = resample_channel(&right, 48000, 44100).unwrap();
    let n = 44100;
    let left_1s = &left_44k[..n];
    let right_1s = &right_44k[..n];
    let padded_len = demucs_core_native::TRAINING_LENGTH;
    let mut left_padded = vec![0.0f32; padded_len];
    let mut right_padded = vec![0.0f32; padded_len];
    left_padded[..n].copy_from_slice(left_1s);
    right_padded[..n].copy_from_slice(right_1s);

    let store = WeightStore::load(&model_path).expect("load");
    let cpu_model = HTDemucs::from_store(&store, "04573f0d", 4, 512).expect("cpu model");

    let mut stft = Stft::new(demucs_core_native::N_FFT, demucs_core_native::HOP_LENGTH);
    let left_spec = stft.forward(&left_padded).expect("stft");
    let right_spec = stft.forward(&right_padded).expect("stft");
    let n_bins = demucs_core_native::N_FFT / 2;
    let n_frames = left_spec.len() / n_bins;
    let left_cac = demucs_core_native::dsp::cac::stft_to_cac(&left_spec, demucs_core_native::N_FFT);
    let right_cac = demucs_core_native::dsp::cac::stft_to_cac(&right_spec, demucs_core_native::N_FFT);
    // Inline build_freq_tensor/build_time_tensor (pub(crate) not visible to tests).
    let bin_frames = n_bins * n_frames;
    let mut freq = vec![0.0f32; 4 * bin_frames];
    for bin in 0..n_bins {
        for frame in 0..n_frames {
            freq[0 * bin_frames + bin * n_frames + frame] = left_cac[0 * bin_frames + bin * n_frames + frame];
            freq[1 * bin_frames + bin * n_frames + frame] = left_cac[1 * bin_frames + bin * n_frames + frame];
            freq[2 * bin_frames + bin * n_frames + frame] = right_cac[0 * bin_frames + bin * n_frames + frame];
            freq[3 * bin_frames + bin * n_frames + frame] = right_cac[1 * bin_frames + bin * n_frames + frame];
        }
    }
    let mut time = vec![0.0f32; 2 * padded_len];
    time[..padded_len].copy_from_slice(&left_padded);
    time[padded_len..].copy_from_slice(&right_padded);
    let freq_shape = [1, 4, n_bins, n_frames];
    let time_shape = [1, 2, padded_len];

    // Input stats
    let fmax = freq.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tmax = time.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("INPUT: freq max={:.4e} | time max={:.4e} | freq.len={} time.len={}",
              fmax, tmax, freq.len(), time.len());

    // SINGLE normalize: pass raw freq/time to htdemucs_forward (it normalizes
    // internally). Previous version double-normalized (normalized here AND
    // inside forward), which gave bogus per-layer magnitudes.
    eprintln!("Running CPU htdemucs_forward (single normalize, ~9 min for 1s real audio)...");
    let t0 = std::time::Instant::now();
    let (freq_out, freq_shape_out, time_out, time_shape_out) = ops_cpu::htdemucs_forward(
        &freq, freq_shape, &time, time_shape, &cpu_model,
    );
    eprintln!("CPU forward took {:.2}s", t0.elapsed().as_secs_f64());
    let fmax_o = freq_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tmax_o = time_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("OUT:   freq shape={:?} len={} max={:.4e}",
              freq_shape_out, freq_out.len(), fmax_o);
    eprintln!("       time shape={:?} len={} max={:.4e}",
              time_shape_out, time_out.len(), tmax_o);
}
