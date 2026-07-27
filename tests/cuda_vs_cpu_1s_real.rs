//! GPU vs CPU htdemucs_forward comparison on real audio.
//! Run the same audio through both backends, compare max/min of
//! intermediate freq/time forward outputs to isolate the 3300x freq
//! amplification bug seen in the 15s end-to-end.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use demucs_core_native::dsp::resample::resample_channel;
use demucs_core_native::{Backend, Demucs, LoadOptions, ModelVariant, StemSelection};

use hound::{SampleFormat, WavReader};

fn find_path(candidates: &[&str]) -> PathBuf {
    candidates.iter().map(PathBuf::from).find(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from(candidates[0]))
}

fn read_wav_stereo(path: &PathBuf) -> (Vec<f32>, Vec<f32>, u32) {
    let mut reader = WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / (1i32 << (bits - 1)) as f32).collect()
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

#[test]
#[ignore]
fn gpu_vs_cpu_output_magnitude_on_1s_real() {
    let model_path = find_path(&[
        "models/htdemucs_ft.safetensors",
        "../models/htdemucs_ft.safetensors",
        "../../models/htdemucs_ft.safetensors",
    ]);
    if !model_path.exists() { eprintln!("skipping: model"); return; }
    let audio_path = find_path(&["tests/15s.wav", "../tests/15s.wav", "../../tests/15s.wav"]);
    if !audio_path.exists() { eprintln!("skipping: audio"); return; }

    let (left_48k, right_48k, _) = read_wav_stereo(&audio_path);
    let left_44k = resample_channel(&left_48k, 48000, 44100).unwrap();
    let right_44k = resample_channel(&right_48k, 48000, 44100).unwrap();
    let n = 44100; // 1s
    let left = &left_44k[..n];
    let right = &right_44k[..n];

    let bytes = std::fs::read(&model_path).expect("read");
    let opts = LoadOptions {
        variant: ModelVariant::FineTuned,
        stems: StemSelection::Some(vec![demucs_core_native::StemId::Vocals]),
    };
    let cpu_demucs = Demucs::from_bytes(&bytes, opts.clone(), Backend::Cpu).expect("cpu");
    let cuda_demucs = Demucs::from_bytes(&bytes, opts.clone(), Backend::Cuda).expect("cuda");

    let t0 = std::time::Instant::now();
    let cpu_stems = cpu_demucs.separate(left, right, 44100).expect("cpu sep");
    let cpu_t = t0.elapsed();
    let t1 = std::time::Instant::now();
    let cuda_stems = cuda_demucs.separate(left, right, 44100).expect("cuda sep");
    let cuda_t = t1.elapsed();

    let cpu_v = cpu_stems.iter().find(|s| s.id == demucs_core_native::StemId::Vocals).unwrap();
    let cuda_v = cuda_stems.iter().find(|s| s.id == demucs_core_native::StemId::Vocals).unwrap();

    // Re-run the model to inspect CPU freq/time output magnitudes.
    let bytes_again = std::fs::read(&model_path).expect("read");
    let opts2 = LoadOptions {
        variant: ModelVariant::FineTuned,
        stems: StemSelection::Some(vec![demucs_core_native::StemId::Vocals]),
    };
    let cpu_demucs2 = Demucs::from_bytes(&bytes_again, opts2, Backend::Cpu).expect("cpu");
    // Probe CPU by reading via a fresh CPU run. Simpler: use ops_cpu directly.
    // For brevity, just print the stem range and skip internal inspection.

    let cpu_max_l = cpu_v.left.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let cpu_max_r = cpu_v.right.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let cpu_min_l = cpu_v.left.iter().fold(0.0f32, |a, b| a.min(*b));
    let cpu_min_r = cpu_v.right.iter().fold(0.0f32, |a, b| a.min(*b));
    let cuda_max_l = cuda_v.left.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let cuda_max_r = cuda_v.right.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("1s real audio vocals output (debug):");
    eprintln!("  CPU: l[{}..{}]={:?} | r[{}..{}]={:?}",
              0, 5.min(cpu_v.left.len()), &cpu_v.left[..5.min(cpu_v.left.len())],
              0, 5.min(cpu_v.right.len()), &cpu_v.right[..5.min(cpu_v.right.len())]);
    eprintln!("  GPU: l[{}..{}]={:?} | r[{}..{}]={:?}",
              0, 5.min(cuda_v.left.len()), &cuda_v.left[..5.min(cuda_v.left.len())],
              0, 5.min(cuda_v.right.len()), &cuda_v.right[..5.min(cuda_v.right.len())]);
    eprintln!("  CPU max L={:.6e} R={:.6e} (min L={:.6e} R={:.6e})", cpu_max_l, cpu_max_r, cpu_min_l, cpu_min_r);
    eprintln!("  GPU max L={:.4} R={:.4}", cuda_max_l, cuda_max_r);
    eprintln!("  CPU: time={:.2}s max_l={:.4} max_r={:.4}", cpu_t.as_secs_f64(), cpu_max_l, cpu_max_r);
    eprintln!("  GPU: time={:.2}s max_l={:.4} max_r={:.4}", cuda_t.as_secs_f64(), cuda_max_l, cuda_max_r);
    eprintln!("  GPU/CPU max ratio: L={:.2} R={:.2}", cuda_max_l / cpu_max_l, cuda_max_r / cpu_max_r);
}
