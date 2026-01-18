use ndarray::Array2;
use rubato::{FftFixedIn, Resampler};
use rustfft::{num_complex::Complex, FftPlanner};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use std::f32::consts::PI;
use std::io::Cursor;
use std::time::Instant;

use crate::asr::AsrError;

// NeMo Parakeet preprocessing parameters
const TARGET_SAMPLE_RATE: u32 = 16000;
const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160; // 10ms at 16kHz
const WIN_LENGTH: usize = 400; // 25ms at 16kHz
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960464477539063e-8; // 2^-24, matching NeMo

/// Audio preprocessing pipeline for NeMo Parakeet TDT
pub struct AudioProcessor {
    mel_filterbank: Array2<f32>,
    hann_window: Vec<f32>,
}

impl AudioProcessor {
    pub fn new() -> Self {
        Self {
            mel_filterbank: create_mel_filterbank_slaney(TARGET_SAMPLE_RATE, N_FFT, N_MELS),
            hann_window: create_hann_window(WIN_LENGTH),
        }
    }

    /// Process raw audio bytes (from file) into mel spectrogram
    pub fn process_audio(&self, audio_bytes: &[u8]) -> Result<Array2<f32>, AsrError> {
        let samples = self.decode_audio(audio_bytes)?;
        log::info!("Decoded {} samples", samples.len());
        self.compute_features(&samples)
    }

    /// Process raw audio samples (from Web Audio API) into mel spectrogram
    pub fn process_raw_samples(&self, samples: &[f32], sample_rate: u32) -> Result<Array2<f32>, AsrError> {
        if samples.is_empty() {
            return Err(AsrError::AudioError("Empty audio samples".to_string()));
        }

        log::info!("Processing {} raw samples at {} Hz", samples.len(), sample_rate);

        // Resample to 16kHz if needed
        let resampled = if sample_rate != TARGET_SAMPLE_RATE {
            log::info!("Resampling from {} to {} Hz", sample_rate, TARGET_SAMPLE_RATE);
            self.resample(samples, sample_rate, TARGET_SAMPLE_RATE)?
        } else {
            samples.to_vec()
        };

        log::info!("Processing {} samples after resampling", resampled.len());
        self.compute_features(&resampled)
    }

    /// Compute mel spectrogram features (matching NeMo preprocessing)
    fn compute_features(&self, samples: &[f32]) -> Result<Array2<f32>, AsrError> {
        if samples.is_empty() {
            return Err(AsrError::AudioError("Empty audio samples".to_string()));
        }

        // 1. Apply pre-emphasis filter
        let preemphasized = apply_preemphasis(samples, PREEMPH);

        // 2. Pad signal for STFT
        let pad_len = N_FFT / 2;
        let mut padded = vec![0.0f32; pad_len];
        padded.extend_from_slice(&preemphasized);
        padded.extend(vec![0.0f32; pad_len]);

        // 3. Compute STFT
        let num_frames = (padded.len() - N_FFT) / HOP_LENGTH + 1;
        if num_frames == 0 {
            return Err(AsrError::AudioError("Audio too short for processing".to_string()));
        }

        let mut fft_planner = FftPlanner::new();
        let fft = fft_planner.plan_fft_forward(N_FFT);
        
        // Padded Hann window to N_FFT size
        let padded_window = pad_window(&self.hann_window, N_FFT);
        
        let mut power_spectrogram = vec![vec![0.0f32; N_FFT / 2 + 1]; num_frames];

        for (frame_idx, start) in (0..=padded.len() - N_FFT).step_by(HOP_LENGTH).enumerate() {
            if frame_idx >= num_frames {
                break;
            }

            // Apply window
            let mut windowed: Vec<Complex<f32>> = padded[start..start + N_FFT]
                .iter()
                .zip(padded_window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();

            // FFT
            fft.process(&mut windowed);

            // Power spectrum (only positive frequencies)
            for (k, c) in windowed.iter().take(N_FFT / 2 + 1).enumerate() {
                power_spectrogram[frame_idx][k] = c.norm_sqr();
            }
        }

        // 4. Apply mel filterbank using matrix multiplication
        // power_spectrogram: [num_frames, N_FFT/2+1]
        // mel_filterbank: [N_MELS, N_FFT/2+1]
        // result: [num_frames, N_MELS]
        let power_arr = Array2::from_shape_vec(
            (num_frames, N_FFT / 2 + 1),
            power_spectrogram.into_iter().flatten().collect()
        ).map_err(|e| AsrError::AudioError(format!("Failed to create power array: {}", e)))?;
        
        // mel_spec = power_arr @ mel_filterbank.T
        let mel_spec = power_arr.dot(&self.mel_filterbank.t());

        // 5. Log compression
        let log_mel = mel_spec.mapv(|x| (x + LOG_ZERO_GUARD).ln());

        // 6. Normalize per mel-channel over time (matching NeMo)
        // Input shape: [time, mels]
        // Need to transpose to [mels, time] for per-channel normalization
        let log_mel_t = log_mel.t(); // Now [mels, time]
        
        let n_time = num_frames as f32;
        let mut normalized_t = Array2::<f32>::zeros((N_MELS, num_frames));
        
        for mel_idx in 0..N_MELS {
            let channel = log_mel_t.row(mel_idx);
            
            // Mean over time
            let mean: f32 = channel.iter().sum::<f32>() / n_time;
            
            // Sample variance (divide by n-1)
            let var: f32 = if n_time > 1.0 {
                channel.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (n_time - 1.0)
            } else {
                1.0
            };
            let std = (var + 1e-5).sqrt();
            
            // Normalize
            for (t, &val) in channel.iter().enumerate() {
                normalized_t[[mel_idx, t]] = (val - mean) / std;
            }
        }
        
        // Transpose back to [time, mels] for output
        let normalized = normalized_t.t().to_owned();

        Ok(normalized)
    }

    /// Decode audio bytes to f32 samples at 16kHz mono
    fn decode_audio(&self, audio_bytes: &[u8]) -> Result<Vec<f32>, AsrError> {
        let decode_start = Instant::now();
        
        let cursor = Cursor::new(audio_bytes.to_vec());
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

        let hint = Hint::new();
        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| AsrError::AudioError(format!("Failed to probe audio format: {}", e)))?;

        let mut format = probed.format;

        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or_else(|| AsrError::AudioError("No audio track found".to_string()))?;

        let track_id = track.id;
        let sample_rate = track.codec_params.sample_rate.unwrap_or(16000);
        let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
        
        // Estimate total samples for pre-allocation (assume ~3 min per MB, 16kHz mono)
        let estimated_samples = (audio_bytes.len() / 1_000_000 + 1) * 3 * 60 * TARGET_SAMPLE_RATE as usize;

        let decoder_opts = DecoderOptions::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .map_err(|e| AsrError::AudioError(format!("Failed to create decoder: {}", e)))?;

        // Pre-allocate with estimated capacity
        let mut all_samples: Vec<f32> = Vec::with_capacity(estimated_samples);
        
        // Reuse sample buffer across packets for efficiency
        let mut sample_buf: Option<SampleBuffer<f32>> = None;

        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => {
                    log::warn!("Error reading packet: {}", e);
                    break;
                }
            };

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = match decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(e) => {
                    log::warn!("Error decoding packet: {}", e);
                    continue;
                }
            };

            let spec = *decoded.spec();
            let duration = decoded.capacity() as u64;

            // Reuse or create sample buffer
            let buf = sample_buf.get_or_insert_with(|| SampleBuffer::<f32>::new(duration, spec));
            if buf.capacity() < duration as usize {
                *buf = SampleBuffer::<f32>::new(duration, spec);
            }
            buf.copy_interleaved_ref(decoded);

            let samples = buf.samples();

            // Convert to mono if stereo (optimized)
            if channels > 1 {
                let inv_channels = 1.0 / channels as f32;
                all_samples.extend(
                    samples.chunks_exact(channels)
                        .map(|chunk| chunk.iter().sum::<f32>() * inv_channels)
                );
            } else {
                all_samples.extend_from_slice(samples);
            }
        }
        
        log::info!("Audio decode: {:.2}s for {} samples", decode_start.elapsed().as_secs_f32(), all_samples.len());

        // Resample to 16kHz if needed
        if sample_rate != TARGET_SAMPLE_RATE {
            let resample_start = Instant::now();
            all_samples = self.resample_fast(&all_samples, sample_rate, TARGET_SAMPLE_RATE)?;
            log::info!("Resample: {:.2}s", resample_start.elapsed().as_secs_f32());
        }

        Ok(all_samples)
    }

    /// Resample audio to target sample rate (original, slower)
    fn resample(
        &self,
        samples: &[f32],
        from_rate: u32,
        to_rate: u32,
    ) -> Result<Vec<f32>, AsrError> {
        self.resample_fast(samples, from_rate, to_rate)
    }
    
    /// Fast resampling with large chunks
    fn resample_fast(
        &self,
        samples: &[f32],
        from_rate: u32,
        to_rate: u32,
    ) -> Result<Vec<f32>, AsrError> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // Use much larger chunk size for efficiency (16K samples = 1 second at 16kHz)
        let chunk_size = 16384;
        let mut resampler = FftFixedIn::<f32>::new(
            from_rate as usize,
            to_rate as usize,
            chunk_size,
            2,
            1,
        )
        .map_err(|e| AsrError::AudioError(format!("Failed to create resampler: {}", e)))?;

        // Pre-allocate output based on ratio
        let ratio = to_rate as f64 / from_rate as f64;
        let estimated_output = (samples.len() as f64 * ratio * 1.1) as usize;
        let mut output = Vec::with_capacity(estimated_output);
        
        let mut input_buffer = vec![vec![0.0f32; chunk_size]; 1];

        for chunk in samples.chunks(chunk_size) {
            // Copy chunk into buffer
            input_buffer[0][..chunk.len()].copy_from_slice(chunk);
            
            // Zero-pad if needed
            if chunk.len() < chunk_size {
                input_buffer[0][chunk.len()..].fill(0.0);
            }

            let resampled = resampler
                .process(&input_buffer, None)
                .map_err(|e| AsrError::AudioError(format!("Resampling failed: {}", e)))?;

            output.extend_from_slice(&resampled[0]);
        }

        Ok(output)
    }
}

/// Apply pre-emphasis filter: y[n] = x[n] - preemph * x[n-1]
fn apply_preemphasis(samples: &[f32], preemph: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let mut result = vec![samples[0]];
    for i in 1..samples.len() {
        result.push(samples[i] - preemph * samples[i - 1]);
    }
    result
}

/// Create Hann window
fn create_hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (length - 1) as f32).cos()))
        .collect()
}

/// Pad window to FFT size (center padding)
fn pad_window(window: &[f32], n_fft: usize) -> Vec<f32> {
    let pad_left = (n_fft - window.len()) / 2;
    let pad_right = n_fft - window.len() - pad_left;
    
    let mut padded = vec![0.0f32; pad_left];
    padded.extend_from_slice(window);
    padded.extend(vec![0.0f32; pad_right]);
    padded
}

/// Convert Hz to Mel scale
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert Mel to Hz scale
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Create mel filterbank with slaney normalization (matching torchaudio/librosa)
fn create_mel_filterbank_slaney(sample_rate: u32, n_fft: usize, n_mels: usize) -> Array2<f32> {
    let n_freqs = n_fft / 2 + 1;
    let f_min = 0.0f32;
    let f_max = sample_rate as f32 / 2.0;

    // Mel scale points
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Create n_mels + 2 equally spaced points in mel scale
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| (n_fft as f32 + 1.0) * hz / sample_rate as f32)
        .collect();

    // Create filterbank with slaney normalization
    let mut filterbank = Array2::<f32>::zeros((n_mels, n_freqs));

    for m in 0..n_mels {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        // Slaney normalization: normalize each filter to have unit area
        let enorm = 2.0 / (hz_points[m + 2] - hz_points[m]);

        for k in 0..n_freqs {
            let freq = k as f32;

            if freq >= f_left && freq < f_center {
                // Rising slope
                filterbank[[m, k]] = enorm * (freq - f_left) / (f_center - f_left);
            } else if freq >= f_center && freq <= f_right {
                // Falling slope
                filterbank[[m, k]] = enorm * (f_right - freq) / (f_right - f_center);
            }
        }
    }

    filterbank
}
