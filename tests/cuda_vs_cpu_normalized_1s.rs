//! Diagnostic: compare CPU vs GPU freq forward on the SAME normalized
//! input (1s real audio). Goal: verify GPU and CPU produce similar
//! forward output, isolating the freq drift to either (a) GPU-only bug
//! or (b) CPU/GPU common bug (model/normalize/weight).

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;

use demucs_core_native::cuda_engine::CudaState;
use demucs_core_native::gpu_model::GpuHTDemucs;
use demucs_core_native::model::HTDemucs;
use demucs_core_native::ops_cpu;
use demucs_core_native::weights::WeightStore;

use hound::{SampleFormat, WavReader};

fn find_path(candidates: &[&str]) -> PathBuf {
    candidates.iter().map(PathBuf::from).find(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from(candidates[0]))
}

fn read_wav_stereo(path: &PathBuf) -> (Vec<f32>, Vec<f32>) {
    let mut reader = WavReader::open(path).expect("open wav");
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
    (left, right)
}

#[test]
#[ignore]
fn gpu_vs_cpu_forward_on_normalized_real_audio() {
    let model_path = find_path(&[
        "models/htdemucs_ft.safetensors",
        "../models/htdemucs_ft.safetensors",
        "../../models/htdemucs_ft.safetensors",
    ]);
    if !model_path.exists() { eprintln!("skipping: model"); return; }
    let audio_path = find_path(&["tests/15s.wav", "../tests/15s.wav", "../../tests/15s.wav"]);
    if !audio_path.exists() { eprintln!("skipping: audio"); return; }

    let (left_48k, right_48k) = read_wav_stereo(&audio_path);
    let left_44k = demucs_core_native::dsp::resample::resample_channel(&left_48k, 48000, 44100).unwrap();
    let right_44k = demucs_core_native::dsp::resample::resample_channel(&right_48k, 48000, 44100).unwrap();
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
    let state = Arc::new(CudaState::new(0).expect("cuda"));
    let gpu_model = GpuHTDemucs::from_cpu(&state, &cpu_model).expect("gpu model");

    // Build freq/time tensors (inline since helpers are pub(crate))
    let mut stft = demucs_core_native::dsp::stft::Stft::new(
        demucs_core_native::N_FFT, demucs_core_native::HOP_LENGTH);
    let left_spec = stft.forward(&left_padded).expect("stft");
    let right_spec = stft.forward(&right_padded).expect("stft");
    let n_bins = demucs_core_native::N_FFT / 2;
    let n_frames = left_spec.len() / n_bins;
    let left_cac = demucs_core_native::dsp::cac::stft_to_cac(&left_spec, demucs_core_native::N_FFT);
    let right_cac = demucs_core_native::dsp::cac::stft_to_cac(&right_spec, demucs_core_native::N_FFT);
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

    // CPU normalize, then re-use the same normalized input for both CPU and GPU.
    let (freq_n, _, _, _, fstd, _) = ops_cpu::normalize_freq(&freq, freq_shape);
    let (time_n, _, _, _, tstd, _) = ops_cpu::normalize_time(&time, time_shape);
    eprintln!("Normalized input: freq max={:.4} | time max={:.4} | fstd={:.4e} tstd={:.4e}",
              freq_n.iter().fold(0.0f32, |a, b| a.max(b.abs())),
              time_n.iter().fold(0.0f32, |a, b| a.max(b.abs())),
              fstd[0], tstd[0]);

    // CPU forward on normalized input (BEFORE denormalize — to compare with GPU raw).
    // We re-implement htdemucs_forward without denormalize by calling
    // cross_domain_transformer_forward + decoder chains. But that's
    // a lot of code; just compare final post-denormalize for now.
    eprintln!("Running CPU htdemucs_forward (slow, ~9 min)...");
    let t0 = std::time::Instant::now();
    let (cpu_freq, cpu_fsh, cpu_time, cpu_tsh) = ops_cpu::htdemucs_forward(
        &freq_n, freq_shape, &time_n, time_shape, &cpu_model);
    eprintln!("CPU forward took {:.2}s", t0.elapsed().as_secs_f64());
    let cpu_fmax = cpu_freq.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let cpu_tmax = cpu_time.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("CPU post-denorm: freq max={:.4} time max={:.4}", cpu_fmax, cpu_tmax);
    eprintln!("CPU pre-denorm estimate: freq max={:.4} time max={:.4}",
              cpu_fmax / fstd[0], cpu_tmax / tstd[0]);

    // GPU forward on the SAME normalized input.
    let gf = state.upload_f32(&freq_n, freq_shape.to_vec()).expect("up f");
    let gt = state.upload_f32(&time_n, time_shape.to_vec()).expect("up t");
    let (gf_out, gt_out) = demucs_core_native::cuda_ops::htdemucs_forward(
        &state, &gf, &gt, &gpu_model).expect("gpu forward");
    let gf_dl = state.download_to_f32(&gf_out).expect("dl f");
    let gt_dl = state.download_to_f32(&gt_out).expect("dl t");
    let gpu_fmax = gf_dl.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let gpu_tmax = gt_dl.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("GPU pre-denorm: freq max={:.4} time max={:.4}", gpu_fmax, gpu_tmax);
    eprintln!("GPU post-denorm estimate: freq max={:.4} time max={:.4}",
              gpu_fmax * fstd[0], gpu_tmax * tstd[0]);

    eprintln!("\nSUMMARY (pre-denorm comparison):");
    eprintln!("  CPU/GPU freq ratio: {:.2}", cpu_fmax / fstd[0] / gpu_fmax);
    eprintln!("  CPU/GPU time ratio: {:.2}", cpu_tmax / tstd[0] / gpu_tmax);
    eprintln!("  burn gold max: 0.66 (vs CPU {:.0}x, vs GPU {:.0}x)",
              cpu_fmax / fstd[0] / 0.66, gpu_fmax / 0.66);
}
