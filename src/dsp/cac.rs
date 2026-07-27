//! "Complex as Channels" (CaC) format conversions — pure CPU, no GPU tensors.

use realfft::num_complex::Complex;

/// Convert a spectrogram from STFT format (flat `[frame × bin]` layout with
/// complex values) to "Complex as Channels" (CaC) format — a flat
/// `[2, n_fft/2, num_frames]` row-major f32 buffer.
pub fn stft_to_cac(spectrogram: &[Complex<f32>], n_fft: usize) -> Vec<f32> {
    let freq_bins = n_fft / 2;
    let num_frames = spectrogram.len() / freq_bins;

    let mut data = vec![0.0f32; 2 * freq_bins * num_frames];

    for bin in 0..freq_bins {
        for frame in 0..num_frames {
            let c = spectrogram[frame * freq_bins + bin];
            data[bin * num_frames + frame] = c.re;
            data[freq_bins * num_frames + bin * num_frames + frame] = c.im;
        }
    }

    data
}

/// Convert CaC data from a flat `[2, freq_bins, num_frames]` row-major f32
/// slice to STFT format. Returns `num_frames × (freq_bins + 1)` complex values
/// with zeroed Nyquist bins appended to each frame.
pub fn cac_data_to_complex(data: &[f32], freq_bins: usize, num_frames: usize) -> Vec<Complex<f32>> {
    let bins = freq_bins + 1;
    let mut spectrogram = vec![Complex::new(0.0, 0.0); num_frames * bins];

    let (reals, imags) = data.split_at(freq_bins * num_frames);

    reals
        .iter()
        .zip(imags)
        .map(|(&re, &im)| Complex::new(re, im))
        .enumerate()
        .for_each(|(i, c)| {
            let frame = i % num_frames;
            let bin = i / num_frames;
            spectrogram[frame * bins + bin] = c;
        });

    spectrogram
}
