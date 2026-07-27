//! End-to-end: Demucs (CUDA backend) vs Demucs (CPU backend) on synthetic
//! audio. This exercises the whole GPU pipeline (STFT/CaC on CPU → forward
//! on GPU → iSTFT on CPU) and compares the produced stems.

#![cfg(feature = "cuda")]

use demucs_core_native::{Backend, Demucs, LoadOptions, ModelVariant, StemSelection};

fn rms(a: &[f32], b: &[f32]) -> f32 {
    (a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / a.len() as f32).sqrt()
}

#[test]
#[ignore]
fn cuda_e2e_vs_cpu() {
    let path = std::path::Path::new("../models/htdemucs.safetensors");
    let opts = LoadOptions {
        variant: ModelVariant::FourStem,
        stems: StemSelection::All,
    };
    let cpu = Demucs::load(path, opts.clone(), Backend::Cpu).expect("cpu load");
    let cuda = Demucs::load(path, opts, Backend::Cuda).expect("cuda load");

    // ~5s of stereo audio (≤ TRAINING_LENGTH → single segment, no overlap-add).
    let sr = 44100u32;
    let n = 5 * sr as usize;
    let left: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.3
                + (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.2
        })
        .collect();
    let right: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            (2.0 * std::f32::consts::PI * 330.0 * t).sin() * 0.25
        })
        .collect();

    let t0 = std::time::Instant::now();
    let cpu_stems = cpu.separate(&left, &right, sr).expect("cpu separate");
    let cpu_dur = t0.elapsed();
    let t1 = std::time::Instant::now();
    let cuda_stems = cuda.separate(&left, &right, sr).expect("cuda separate");
    let cuda_dur = t1.elapsed();

    eprintln!("CPU  separate: {:.2?}", cpu_dur);
    eprintln!("CUDA separate: {:.2?}", cuda_dur);
    eprintln!("speedup: {:.2}x", cpu_dur.as_secs_f64() / cuda_dur.as_secs_f64());

    assert_eq!(cpu_stems.len(), cuda_stems.len(), "stem count mismatch");
    let mut worst_rms = 0.0f32;
    for (cs, gs) in cpu_stems.iter().zip(cuda_stems.iter()) {
        assert_eq!(cs.id, gs.id);
        assert_eq!(cs.left.len(), gs.left.len(), "left len mismatch stem {:?}", cs.id);
        let rl = rms(&cs.left, &gs.left);
        let rr = rms(&cs.right, &gs.right);
        let mv = cs.left.iter().chain(cs.right.iter()).fold(0.0f32, |a, b| a.max(b.abs()));
        eprintln!(
            "stem {:?}: rms L={:.4} R={:.4} (max_val={:.3}, rms/mv={:.1}%)",
            cs.id, rl, rr, mv, 100.0 * rl.max(rr) / mv.max(1e-6)
        );
        worst_rms = worst_rms.max(rl).max(rr);
    }
    // Full e2e (STFT→forward→iSTFT) f16 error vs f32 CPU reference.
    // Accept rms up to ~0.05 on normalized-ish audio.
    assert!(worst_rms < 0.1, "worst stem rms={:.4} too large", worst_rms);
}
