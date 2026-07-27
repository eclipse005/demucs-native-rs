//! End-to-end CUDA test against the burn golden reference.
//!
//! Loads `models/htdemucs_ft.safetensors`, runs the native CUDA inference on
//! `tests/15s.wav`, and compares the vocals stem against
//! `tests/reference_burn/vocals.wav` (burn wgpu CLI baseline).
//!
//! Run: cargo test -p demucs-core-native --features cuda \
//!      --test cuda_end_to_end_15s -- --nocapture --ignored --test-threads=1

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use demucs_core_native::metadata::HTDEMUCS_FT_ID;
use demucs_core_native::{Demucs, LoadOptions, ModelVariant, StemSelection};

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

/// Read python_vocals.npy shape [4, 2, 720000] and extract vocals (index 3) as stereo.
fn read_python_vocals(path: &PathBuf) -> (Vec<f32>, Vec<f32>) {
    let bytes = std::fs::read(path).expect("read npy");
    // npy format: 6-byte header \x93NUMPY, version, then header_len, then data.
    // We don't have a npy crate in dev-deps; use a simple Python-style parse.
    // Quick parse: skip until 0x93NUMPY then read dict header.
    assert!(bytes.starts_with(b"\x93NUMPY"), "not a npy file");
    let major = bytes[6];
    let minor = bytes[7];
    let header_len = if major <= 1 {
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize
    } else {
        let mut arr = [0u8; 4];
        arr.copy_from_slice(&bytes[8..12]);
        u32::from_le_bytes(arr) as usize
    };
    let header = std::str::from_utf8(&bytes[10 + if major <= 1 { 0 } else { 0 }..10 + header_len])
        .expect("header utf8");
    // The header contains a "shape" entry. Parse it.
    eprintln!("npy header: {}", header.lines().filter(|l| l.contains("shape")).next().unwrap_or(""));
    // Parse shape from header string like "{'shape': (4, 2, 720000), ..."
    // (npy uses tuple notation in newer versions).
    let shape_start = header.find("'shape':").expect("shape in header") + "'shape':".len();
    let shape_rest = &header[shape_start..];
    let lbracket = shape_rest.find(|c: char| c == '[' || c == '(').expect("[ or (");
    let rbracket = shape_rest.find(|c: char| c == ']' || c == ')').expect("] or )");
    let shape_str = &shape_rest[lbracket + 1..rbracket];
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim().parse::<usize>().expect("dim"))
        .collect();
    let n_stems = shape[0];
    let n_channels = shape[1];
    let n_samples = shape[2];
    eprintln!("npy shape: {}x{}x{}", n_stems, n_channels, n_samples);
    // Data starts at offset 10 + header_len. dtype: float32 little-endian.
    let data_offset = 10 + header_len;
    let total_floats = n_stems * n_channels * n_samples;
    let mut all: Vec<f32> = vec![0.0; total_floats];
    let src = &bytes[data_offset..data_offset + total_floats * 4];
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), all.as_mut_ptr() as *mut u8, total_floats * 4);
    }
    // Extract vocals (index 3): [2, 720000]
    let vocals_offset = 3 * n_channels * n_samples;
    let left = all[vocals_offset..vocals_offset + n_samples].to_vec();
    let right = all[vocals_offset + n_samples..vocals_offset + 2 * n_samples].to_vec();
    (left, right)
}

fn write_wav_stereo(path: &PathBuf, left: &[f32], right: &[f32], sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
    for (l, r) in left.iter().zip(right.iter()) {
        writer.write_sample(*l).unwrap();
        writer.write_sample(*r).unwrap();
    }
    writer.finalize().expect("finalize wav");
}

#[test]
#[ignore]
fn cuda_end_to_end_15s_vocals_burn_vs_native() {
    let model_path = find_path(&[
        "models/htdemucs_ft.safetensors",
        "../models/htdemucs_ft.safetensors",
        "../../models/htdemucs_ft.safetensors",
    ]);
    if !model_path.exists() {
        eprintln!("skipping: model not found");
        return;
    }
    let audio_path = find_path(&[
        "tests/15s.wav",
        "../tests/15s.wav",
        "../../tests/15s.wav",
    ]);
    if !audio_path.exists() {
        eprintln!("skipping: audio not found");
        return;
    }
    let ref_path = find_path(&[
        "tests/python_vocals.npy",
        "../tests/python_vocals.npy",
        "../../tests/python_vocals.npy",
    ]);
    if !ref_path.exists() {
        eprintln!("skipping: python reference not found");
        return;
    }

    let bytes = std::fs::read(&model_path).expect("read model file");
    let opts = LoadOptions {
        variant: ModelVariant::FineTuned,
        stems: StemSelection::Some(vec![demucs_core_native::StemId::Vocals]),
    };
    let demucs = Demucs::from_bytes(&bytes, opts.clone(), demucs_core_native::Backend::Cuda)
        .expect("load Demucs (cuda)");

    let (left, right, sample_rate) = read_wav_stereo(&audio_path);
    eprintln!(
        "Loaded {} samples @ {} Hz, {:.2}s",
        left.len(),
        sample_rate,
        left.len() as f64 / sample_rate as f64
    );

    let start = std::time::Instant::now();
    let stems = demucs.separate(&left, &right, sample_rate).expect("cuda separate");
    let elapsed = start.elapsed();
    eprintln!(
        "CUDA inference took {:.2}s (RTFx {:.3}×)",
        elapsed.as_secs_f64(),
        15.0 / elapsed.as_secs_f64()
    );

    let vocals = stems
        .iter()
        .find(|s| s.id == demucs_core_native::StemId::Vocals)
        .expect("vocals stem missing");

    let (ref_l, ref_r) = read_python_vocals(&ref_path);

    // Always write the native output first so we can do Python comparison
    // even if the burn-vs-native tolerance fails.
    let out_dir = PathBuf::from("stems/cuda_15s");
    let _ = std::fs::create_dir_all(&out_dir);
    write_wav_stereo(
        &out_dir.join("vocals.wav"),
        &vocals.left,
        &vocals.right,
        sample_rate,
    );
    eprintln!("Wrote {}", out_dir.join("vocals.wav").display());

    let n = vocals.left.len().min(ref_l.len());
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0usize;
    let mut max_ref = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    for i in 0..n {
        let d_l = (vocals.left[i] - ref_l[i]).abs();
        let d_r = (vocals.right[i] - ref_r[i]).abs();
        let d = d_l.max(d_r);
        if d > max_diff {
            max_diff = d;
            max_diff_idx = i;
        }
        max_ref = max_ref.max(ref_l[i].abs()).max(ref_r[i].abs());
        let d_for_stats = d_l.max(d_r);
        let r_for_stats = ref_l[i].abs().max(ref_r[i].abs());
        sum_diff += d_for_stats as f64;
        sum_sq_diff += (d_for_stats as f64) * (d_for_stats as f64);
        sum_sq_ref += (r_for_stats as f64) * (r_for_stats as f64);
    }
    let mean_diff = sum_diff / n as f64;
    let rms_diff = (sum_sq_diff / n as f64).sqrt();
    let rms_ref = (sum_sq_ref / n as f64).sqrt();
    eprintln!(
        "vocals (n={} samples): max_abs_diff={:.6e} at idx={}, mean_diff={:.6e}, rms_diff={:.4}, rms_ref={:.4}",
        n, max_diff, max_diff_idx, mean_diff, rms_diff, rms_ref
    );
    eprintln!("  max ref amplitude = {:.4}", max_ref);
    // Show the values around the max-diff index.
    let lo = max_diff_idx.saturating_sub(3);
    let hi = (max_diff_idx + 4).min(n);
    eprintln!("  native L[{}..{}] = {:?}", lo, hi, &vocals.left[lo..hi]);
    eprintln!("  burn   L[{}..{}] = {:?}", lo, hi, &ref_l[lo..hi]);
    eprintln!("  native R[{}..{}] = {:?}", lo, hi, &vocals.right[lo..hi]);
    eprintln!("  burn   R[{}..{}] = {:?}", lo, hi, &ref_r[lo..hi]);
    // Native-vs-native: mean amplitude and a few spot stats.
    let native_max = vocals.left.iter().chain(vocals.right.iter())
        .fold(0.0f32, |a, b| a.max(b.abs()));
    let native_mean = vocals.left.iter().chain(vocals.right.iter())
        .sum::<f32>() as f64 / (2.0 * n as f64);
    let ref_max = ref_l.iter().chain(ref_r.iter())
        .fold(0.0f32, |a, b| a.max(b.abs()));
    let ref_mean = ref_l.iter().chain(ref_r.iter())
        .sum::<f32>() as f64 / (2.0 * n as f64);
    eprintln!("  native_max={:.4} native_mean={:.6e} | ref_max={:.4} ref_mean={:.6e}",
              native_max, native_mean, ref_max, ref_mean);

    // Tolerance: native vs burn wgpu gold. NOTE: CUDA native currently
    // produces values ~80x larger than burn on real audio (max native
    // 54, burn 0.66). Root cause unknown — GPU/CPU agree on small
    // synthetic input (rms 4.41%), so it's not a structural bug in the
    // GPU forward. Likely a precision drift in the f16 path that
    // only manifests on real audio (whose activations are 10-100x
    // larger than synthetic). TOLERANCE LEFT AT 1.0 (burn wgpu vs
    // Python ground truth is ~0.5; native should match that) so the
    // test surfaces the regression.
    assert!(
        max_diff < 1.0,
        "vocals max_abs_diff {max_diff} exceeds tolerance 1.0 (vs wgpu ref) — freq path has ~80x scale drift on real audio, root cause not fixed"
    );

    assert_eq!(opts.variant.info().id, HTDEMUCS_FT_ID);
}
