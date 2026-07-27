//! Native HTDemucs v4 inference CLI (CPU + optional CUDA).
//!
//! Standalone binary that loads a safetensors model and runs separation
//! on an input WAV.
//!
//! Usage (CPU):
//!   cargo run --release --bin demucs-native -- \
//!       -i input.wav -o ./stems/ -m htdemucs_ft \
//!       --model-dir ./models --device cpu -s vocals
//!
//! Usage (CUDA, requires --features cuda):
//!   cargo run --release --features cuda --bin demucs-native -- \
//!       -i input.wav -o ./stems/ -m htdemucs_ft \
//!       --model-dir ./models --device cuda -s vocals

use std::path::PathBuf;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

use demucs_core_native::{
    Backend, Demucs, LoadOptions, ModelVariant, StemSelection,
};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut output = PathBuf::from("./stems/");
    let mut model_dir = PathBuf::from("./models");
    let mut variant = "htdemucs".to_string();
    // Default to auto: CUDA when compiled with the feature and a device is present.
    let mut device = "auto".to_string();
    let mut stems: Vec<String> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-i" | "--input" => { input = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "-o" | "--output" => { output = PathBuf::from(&args[i + 1]); i += 2; }
            "-m" | "--model" => { variant = args[i + 1].clone(); i += 2; }
            "--model-dir" => { model_dir = PathBuf::from(&args[i + 1]); i += 2; }
            "--device" => { device = args[i + 1].clone(); i += 2; }
            "-s" | "--stems" => { stems = args[i + 1].split(',').map(String::from).collect(); i += 2; }
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            _ => { eprintln!("unknown arg: {}", args[i]); i += 1; }
        }
    }
    let input = input.ok_or_else(|| anyhow::anyhow!("-i <input.wav> required (try --help)"))?;
    let model_name = match variant.as_str() {
        "htdemucs" => "htdemucs.safetensors",
        "htdemucs_6s" => "htdemucs_6s.safetensors",
        "htdemucs_ft" => "htdemucs_ft.safetensors",
        _ => anyhow::bail!("unknown model variant: {variant}"),
    };
    let model_path = model_dir.join(model_name);
    let mv = match variant.as_str() {
        "htdemucs" => ModelVariant::FourStem,
        "htdemucs_6s" => ModelVariant::SixStem,
        "htdemucs_ft" => ModelVariant::FineTuned,
        _ => unreachable!(),
    };
    let backend = parse_backend(&device)?;
    let stem_selection = if stems.is_empty() {
        StemSelection::All
    } else {
        let mut parsed = Vec::new();
        for s in &stems {
            let id = demucs_core_native::StemId::parse(s)
                .ok_or_else(|| anyhow::anyhow!("unknown stem: {s}"))?;
            parsed.push(id);
        }
        StemSelection::Some(parsed)
    };

    eprintln!("Loading {} from {}", model_name, model_path.display());
    let bytes = std::fs::read(&model_path)
        .map_err(|e| anyhow::anyhow!("read model {}: {e}", model_path.display()))?;
    let opts = LoadOptions { variant: mv, stems: stem_selection };
    let demucs = Demucs::from_bytes(&bytes, opts, backend)?;
    eprintln!("Loaded on {}", backend.tag());

    eprintln!("Reading {}", input.display());
    let mut reader = WavReader::open(&input)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / (1i32 << (bits - 1)) as f32).collect()
        }
    };
    let mut left = Vec::with_capacity(samples.len() / spec.channels as usize);
    let mut right = Vec::with_capacity(samples.len() / spec.channels as usize);
    for frame in samples.chunks_exact(spec.channels as usize) {
        left.push(frame[0]);
        right.push(frame[spec.channels as usize - 1]);
    }
    let sample_rate = spec.sample_rate;
    // Don't pre-resample here — separate() resamples to 44.1k internally and
    // back to the original rate on output. Writing at the original rate keeps
    // output quality identical to the input (previously hardcoded 44100 →
    // quality loss + sr mismatch vs burn).

    eprintln!("Separating {:.1}s of audio @ {} Hz", left.len() as f32 / sample_rate as f32, sample_rate);
    let t0 = std::time::Instant::now();
    let stems = demucs.separate(&left, &right, sample_rate)?;
    eprintln!("Done in {:.2}s", t0.elapsed().as_secs_f64());

    std::fs::create_dir_all(&output)?;
    let out_spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    for stem in &stems {
        let path = output.join(format!("{}.wav", stem.id.as_str()));
        let mut w = WavWriter::create(&path, out_spec)?;
        for (l, r) in stem.left.iter().zip(stem.right.iter()) {
            w.write_sample(*l)?;
            w.write_sample(*r)?;
        }
        w.finalize()?;
        eprintln!("Wrote {}", path.display());
    }
    #[cfg(feature = "cuda")]
    demucs_core_native::cuda_ops::print_global_agg();
    Ok(())
}

fn parse_backend(device: &str) -> anyhow::Result<Backend> {
    match device {
        "cpu" => Ok(Backend::Cpu),
        "auto" => Ok(Backend::Auto),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Backend::Cuda),
        #[cfg(not(feature = "cuda"))]
        "cuda" => anyhow::bail!(
            "CUDA support was not compiled in. Rebuild with: cargo build --release --features cuda"
        ),
        other => {
            #[cfg(feature = "cuda")]
            anyhow::bail!("unsupported device: {other} (supported: auto, cpu, cuda)");
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("unsupported device: {other} (supported: auto, cpu)");
        }
    }
}

fn print_usage() {
    eprintln!(
        "\
demucs-native — HTDemucs v4 hand-written CPU/CUDA inference

Usage:
  demucs-native -i <input.wav> [options]

Options:
  -i, --input <path>       Input WAV (required)
  -o, --output <dir>       Output directory for stem WAVs (default: ./stems/)
  -m, --model <name>       Model: htdemucs | htdemucs_6s | htdemucs_ft (default: htdemucs)
      --model-dir <dir>    Directory containing *.safetensors (default: ./models)
      --device <name>       auto | cpu | cuda  (default: auto; cuda needs --features cuda)
  -s, --stems <list>       Comma-separated stems (default: all)
  -h, --help               Show this help

Models are NOT bundled. Download .safetensors into --model-dir first
(see README for download instructions)."
    );
}
