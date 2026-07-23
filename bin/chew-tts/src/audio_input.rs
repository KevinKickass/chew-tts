use anyhow::{Context, ensure};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;
use std::io::Cursor;

pub fn decode_wav(bytes: &[u8]) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes)).context("invalid WAV reference")?;
    let spec = reader.spec();
    ensure!(spec.channels > 0, "reference WAV has no channels");
    let interleaved = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("could not decode float WAV")?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << spec.bits_per_sample.saturating_sub(1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
                .context("could not decode PCM WAV")?
        }
    };
    let channels = spec.channels as usize;
    let mono = interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect::<Vec<_>>();
    ensure!(!mono.is_empty(), "reference WAV is empty");
    Ok((mono, spec.sample_rate))
}

pub fn resample(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate {
        return input.to_vec();
    }
    let output_len =
        ((input.len() as u64 * target_rate as u64) / source_rate as u64).max(1) as usize;
    let ratio = source_rate as f64 / target_rate as f64;
    let cutoff = (target_rate as f64 / source_rate as f64).min(1.0) * 0.95;
    const HALF_TAPS: isize = 16;
    (0..output_len)
        .map(|index| {
            let source = index as f64 * ratio;
            let center = source.floor() as isize;
            let mut value = 0.0f64;
            let mut weight_sum = 0.0f64;
            for tap in -HALF_TAPS + 1..=HALF_TAPS {
                let sample_index = (center + tap).clamp(0, input.len() as isize - 1) as usize;
                let distance = source - (center + tap) as f64;
                let phase = std::f64::consts::PI * distance * cutoff;
                let sinc = if phase.abs() < 1e-12 {
                    1.0
                } else {
                    phase.sin() / phase
                };
                let window_position = distance / HALF_TAPS as f64;
                let window = if window_position.abs() <= 1.0 {
                    0.5 + 0.5 * (std::f64::consts::PI * window_position).cos()
                } else {
                    0.0
                };
                let weight = cutoff * sinc * window;
                value += input[sample_index] as f64 * weight;
                weight_sum += weight;
            }
            (value / weight_sum.max(1e-12)) as f32
        })
        .collect()
}

/// Match Qwen's torch.stft + librosa Slaney-filter speaker mel frontend.
pub fn speaker_mel(samples: &[f32]) -> anyhow::Result<(Vec<f32>, usize)> {
    const N_FFT: usize = 1024;
    const HOP: usize = 256;
    const PADDING: usize = (N_FFT - HOP) / 2;
    const BINS: usize = N_FFT / 2 + 1;
    const MELS: usize = 128;
    ensure!(
        samples.len() > PADDING,
        "reference audio must be longer than {:.1} ms",
        PADDING as f32 / 24.0
    );
    let mut padded = Vec::with_capacity(samples.len() + 2 * PADDING);
    for index in (1..=PADDING).rev() {
        padded.push(samples[index]);
    }
    padded.extend_from_slice(samples);
    for index in 0..PADDING {
        padded.push(samples[samples.len() - 2 - index]);
    }
    let frames = (padded.len() - N_FFT) / HOP + 1;
    let window = (0..N_FFT)
        .map(|index| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / N_FFT as f32).cos())
        .collect::<Vec<_>>();
    let filterbank = slaney_filterbank();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut spectrum = vec![Complex32::default(); N_FFT];
    let mut mel = vec![0.0; frames * MELS];
    for frame in 0..frames {
        let offset = frame * HOP;
        for index in 0..N_FFT {
            spectrum[index] = Complex32::new(padded[offset + index] * window[index], 0.0);
        }
        fft.process(&mut spectrum);
        let magnitude = spectrum[..BINS]
            .iter()
            .map(|value| (value.norm_sqr() + 1e-9).sqrt())
            .collect::<Vec<_>>();
        for mel_bin in 0..MELS {
            let value = filterbank[mel_bin * BINS..(mel_bin + 1) * BINS]
                .iter()
                .zip(&magnitude)
                .map(|(weight, magnitude)| weight * magnitude)
                .sum::<f32>();
            mel[frame * MELS + mel_bin] = value.max(1e-5).ln();
        }
    }
    Ok((mel, frames))
}

fn slaney_filterbank() -> Vec<f32> {
    const SAMPLE_RATE: f32 = 24_000.0;
    const N_FFT: usize = 1024;
    const BINS: usize = N_FFT / 2 + 1;
    const MELS: usize = 128;
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(SAMPLE_RATE / 2.0);
    let frequencies = (0..MELS + 2)
        .map(|index| {
            let mel = min_mel + (max_mel - min_mel) * index as f32 / (MELS + 1) as f32;
            mel_to_hz(mel)
        })
        .collect::<Vec<_>>();
    let fft_frequencies = (0..BINS)
        .map(|index| index as f32 * SAMPLE_RATE / N_FFT as f32)
        .collect::<Vec<_>>();
    let mut filters = vec![0.0; MELS * BINS];
    for mel in 0..MELS {
        let lower = frequencies[mel];
        let center = frequencies[mel + 1];
        let upper = frequencies[mel + 2];
        let normalization = 2.0 / (upper - lower);
        for (bin, frequency) in fft_frequencies.iter().copied().enumerate() {
            let lower_slope = (frequency - lower) / (center - lower);
            let upper_slope = (upper - frequency) / (upper - center);
            filters[mel * BINS + bin] = lower_slope.min(upper_slope).max(0.0) * normalization;
        }
    }
    filters
}

fn hz_to_mel(frequency: f32) -> f32 {
    const FREQ_STEP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / FREQ_STEP;
    const LOG_STEP: f32 = 0.068_751_78;
    if frequency >= MIN_LOG_HZ {
        MIN_LOG_MEL + (frequency / MIN_LOG_HZ).ln() / LOG_STEP
    } else {
        frequency / FREQ_STEP
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    const FREQ_STEP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / FREQ_STEP;
    const LOG_STEP: f32 = 0.068_751_78;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (LOG_STEP * (mel - MIN_LOG_MEL)).exp()
    } else {
        FREQ_STEP * mel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_frontend_has_expected_frame_geometry() {
        let samples = vec![0.0; 24_000];
        let (mel, frames) = speaker_mel(&samples).unwrap();
        assert_eq!(frames, 93);
        assert_eq!(mel.len(), frames * 128);
        assert!(mel.iter().all(|value| value.is_finite()));
    }
}
