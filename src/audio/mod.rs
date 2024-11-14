use rubato::{SincFixedIn, SincInterpolationParameters, WindowFunction, Resampler};
use hound::{SampleFormat, WavReader};
use std::path::Path;
use std::process::Command;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;
use anyhow::Result;
use tracing::{info, error};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("Failed to read audio file: {0}")]
    ReadError(#[from] hound::Error),
    
    #[error("Failed to process audio: {0}")]
    ProcessError(String),
    
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),
    
    #[error("FFmpeg error: {0}")]
    FfmpegError(String)
}

pub enum AudioFormat {
    Wav,
    Aac,
    Amr,
    M4a,
    Ogg,
    Opus,
    Wma,
    Mp3,
    Flac,
}

/// Denoise configuration parameters
#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    pub frame_size: usize,
    pub overlap: f32,
    pub strength: f32,
    pub noise_gate: f32,
    pub enable_noise_reduction: bool,
    pub threshold: f32,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            frame_size: 2048,
            overlap: 0.75,
            strength: 0.2,
            noise_gate: 0.003,
            enable_noise_reduction: true,
            threshold: 0.002,
        }
    }
}

pub type ProgressCallback = Box<dyn Fn(f32) + Send>;

pub type AudioCallback = Box<dyn FnMut(Vec<f32>) + Send>;

/// Stream audio processor
#[allow(dead_code)]
pub struct StreamAudioProcessor {
    config: DenoiseConfig,
    buffer: Vec<f32>,
    frame_size: usize,
    sample_rate: u32,
    callback: AudioCallback,
    prev_energy: f32,
    noise_floor: f32,
}

impl StreamAudioProcessor {
    pub fn new(config: DenoiseConfig, callback: AudioCallback) -> Self {
        Self {
            config,
            buffer: Vec::new(),
            frame_size: 2048,
            sample_rate: 16000,
            callback,
            prev_energy: 0.0,
            noise_floor: 0.0,
        }
    }

    pub fn process_chunk(&mut self, chunk: &[f32]) {
        // Normalize input data
        let normalized = normalize_audio(chunk);
        self.buffer.extend_from_slice(&normalized);
        
        // When the buffer is large enough, process data
        while self.buffer.len() >= self.frame_size {
            let frame = self.buffer.drain(..self.frame_size).collect::<Vec<_>>();
            
            // If it's the first frame, initialize the noise floor
            if self.noise_floor == 0.0 {
                self.noise_floor = estimate_noise_floor(&frame);
            }
            
            let processed = self.process_frame(&frame);
            (self.callback)(processed);
        }
    }

    fn process_frame(&mut self, frame: &[f32]) -> Vec<f32> {
        // 1. Preemphasis processing
        let preemphasized = preemphasis(frame, 0.97);
        
        // 2. Voice activity detection
        let energy = preemphasized.iter().map(|&s| s * s).sum::<f32>() / frame.len() as f32;
        let threshold = self.noise_floor * 1.2 + self.prev_energy * 0.1;
        
        let gain = if energy > threshold {
            1.0
        } else {
            (energy / threshold).max(0.1)
        };
        
        // Update state
        self.prev_energy = energy;
        self.noise_floor = self.noise_floor * 0.95 + energy.min(self.noise_floor) * 0.05;
        
        // 3. Apply gain
        let mut processed: Vec<f32> = frame.iter().map(|&s| s * gain).collect();
        
        // 4. Denoise (if enabled)
        if self.config.enable_noise_reduction {
            processed = denoise_audio(&processed, &self.config);
        }
        
        // 5. Apply noise gate
        processed = apply_noise_gate(&processed, self.config.noise_gate);
        
        processed
    }

    pub fn finish(&mut self) {
        if !self.buffer.is_empty() {
            // Fill remaining data
            while self.buffer.len() < self.frame_size {
                self.buffer.push(0.0);
            }
            let frame = self.buffer.drain(..).collect::<Vec<_>>();
            let processed = self.process_frame(&frame);
            (self.callback)(processed);
        }
    }
}

/// Stream parsing audio file
pub async fn parse_audio_file_stream(
    path: &Path, 
    config: &DenoiseConfig,
    mut callback: impl FnMut(Vec<f32>) + Send + 'static
) -> Result<(), AudioError> {
    info!("ensuring audio file format: {}", path.display());
    let wav_path = ensure_wav_format(path)?;
    
    info!("reading audio file: {}", wav_path.display());
    let mut reader = WavReader::open(&wav_path)
        .map_err(AudioError::ReadError)?;
    
    let num_channels = reader.spec().channels as usize;
    let original_sample_rate = reader.spec().sample_rate;

    // Create resampler (if needed)
    let mut resampler = if original_sample_rate != 16000 {
        Some(create_resampler(original_sample_rate, 16000)?)
    } else {
        None
    };

    // Create audio processor
    let mut processor = StreamAudioProcessor::new(
        config.clone(),
        Box::new(move |processed| callback(processed))
    );

    // Read and process audio in chunks
    const CHUNK_SIZE: usize = 4096;
    let mut chunk = Vec::with_capacity(CHUNK_SIZE);
    
    info!("processing audio file: {}", wav_path.display());
    for sample in reader.samples::<i16>() {
        let sample = sample.map_err(AudioError::ReadError)? as f32 / 32768.0;
        chunk.push(sample);
        
        if chunk.len() >= CHUNK_SIZE {
            // Convert to mono
            let mono_chunk = convert_to_mono(&chunk, num_channels);
            
            // Resample (if needed)
            let resampled_chunk = if let Some(ref mut resampler) = resampler {
                resample_chunk(resampler, &mono_chunk)?
            } else {
                mono_chunk
            };
            
            // Process audio chunk
            processor.process_chunk(&resampled_chunk);
            chunk.clear();
        }
    }

    // Process remaining samples
    if !chunk.is_empty() {
        let mono_chunk = convert_to_mono(&chunk, num_channels);
        let resampled_chunk = if let Some(ref mut resampler) = resampler {
            resample_chunk(resampler, &mono_chunk)?
        } else {
            mono_chunk
        };
        processor.process_chunk(&resampled_chunk);
    }

    // Finish processing
    processor.finish();
    info!("finished processing audio file: {}", wav_path.display());
    // Clean up temporary file
    if wav_path != path {
        if let Err(e) = std::fs::remove_file(&wav_path) {
            error!("Failed to remove temporary WAV file: {}", e);
        }
    }

    Ok(())
}

fn create_resampler(from_rate: u32, to_rate: u32) -> Result<SincFixedIn<f32>, AudioError> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: rubato::SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    SincFixedIn::new(
        to_rate as f64 / from_rate as f64,
        2.0,
        params,
        4096,
        1,
    ).map_err(|e| AudioError::ProcessError(e.to_string()))
}

fn resample_chunk(resampler: &mut SincFixedIn<f32>, chunk: &[f32]) -> Result<Vec<f32>, AudioError> {
    let output = resampler.process(&[chunk.to_vec()], None)
        .map_err(|e| AudioError::ProcessError(e.to_string()))?;
    Ok(output[0].clone())
}

/// Preemphasis processing, enhancing high-frequency speech features
fn preemphasis(samples: &[f32], coefficient: f32) -> Vec<f32> {
    let mut result = Vec::with_capacity(samples.len());
    result.push(samples[0]);
    
    for i in 1..samples.len() {
        result.push(samples[i] - coefficient * samples[i-1]);
    }
    
    result
}

/// Adaptive voice activity detection
#[allow(dead_code)]
fn adaptive_voice_activity_detection(samples: &[f32], frame_size: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(samples.len());
    let mut noise_floor = estimate_noise_floor(samples);
    let mut prev_energy = 0.0;
    
    for chunk in samples.chunks(frame_size) {
        let energy = chunk.iter().map(|&s| s * s).sum::<f32>() / chunk.len() as f32;
        
        // Adaptive threshold
        let threshold = noise_floor * 1.2 + prev_energy * 0.1;
        
        // Smoothing
        let gain = if energy > threshold {
            1.0
        } else {
            (energy / threshold).max(0.1) // Keep some low energy signals
        };
        
        // Apply gain
        result.extend(chunk.iter().map(|&s| s * gain));
        
        // Update state
        prev_energy = energy;
        noise_floor = noise_floor * 0.95 + energy.min(noise_floor) * 0.05;
    }
    
    result
}

/// Ensure audio file is in WAV format
/// 
/// If the input file is not in WAV format, use FFmpeg to convert it to WAV format
/// 
/// # Parameters
/// * `path` - Path to the input audio file
/// 
/// # Returns
/// * `std::path::PathBuf` - Path to the WAV format file (possibly the original file path or a new WAV file path)
/// 
/// # Note
/// This function relies on the FFmpeg library installed on the system
fn ensure_wav_format(path: &Path) -> Result<std::path::PathBuf, AudioError> {
    if let Some(extension) = path.extension() {
        if extension.to_str().unwrap_or("").to_lowercase() == "wav" {
            return Ok(path.to_path_buf());
        }
    }

    let output_path = path.with_extension("wav");
    info!("Converting audio file to WAV format...");
    
    let status = Command::new("ffmpeg")
        .arg("-i")
        .arg(path)
        .arg("-acodec")
        .arg("pcm_s16le")
        .arg("-ar")
        .arg("44100")
        .arg(&output_path)
        .status()
        .map_err(|e| AudioError::FfmpegError(e.to_string()))?;

    if !status.success() {
        return Err(AudioError::FfmpegError(format!("FFmpeg conversion failed with status: {}", status)));
    }

    Ok(output_path)
}

/// Read WAV file
/// 
/// Reads a WAV file and returns its samples data, number of channels, and sample rate
/// 
/// # Parameters
/// * `path` - Path to the WAV file
/// 
/// # Returns
/// * `(Vec<f32>, usize, u32)` - Tuple containing samples data, number of channels, and sample rate
/// 
/// # Panics
/// If the file format is not as expected (non-integer sample format or non-16-bit samples), the function will panic
#[allow(dead_code)]
fn read_wav_file(path: &Path) -> Result<(Vec<f32>, usize, u32), AudioError> {
    let mut reader = WavReader::open(path)
        .map_err(|e| AudioError::ReadError(e))?;
    
    let num_channels = reader.spec().channels as usize;
    let sample_rate = reader.spec().sample_rate;

    if reader.spec().sample_format != SampleFormat::Int {
        return Err(AudioError::UnsupportedFormat(format!("Unsupported sample format: expected integer format")));
    }

    if reader.spec().bits_per_sample != 16 {
        return Err(AudioError::UnsupportedFormat(format!("Unsupported bits per sample: expected 16 bits")));
    }

    info!("Original sample rate: {} Hz", sample_rate);

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|val| val as f32))
        .collect::<std::result::Result<Vec<f32>, _>>()
        .map_err(|e| AudioError::ReadError(e))?;

    Ok((samples, num_channels, sample_rate))
}

/// Convert multi-channel audio to mono
/// 
/// Converts multi-channel audio to mono by averaging the samples of each channel
/// 
/// # Parameters
/// * `samples` - Input audio samples
/// * `num_channels` - Number of channels in the input audio
/// 
/// # Returns
/// * `Vec<f32>` - Converted mono audio samples
fn convert_to_mono(samples: &[f32], num_channels: usize) -> Vec<f32> {
    samples.par_chunks(num_channels)
        .map(|chunk| {
            chunk.iter().sum::<f32>() / num_channels as f32
        })
        .collect()
}

/// Normalize audio
/// 
/// Normalizes audio samples to the [-1, 1] range
/// 
/// # Parameters
/// * `samples` - Input audio samples
/// 
/// # Returns
/// * `Vec<f32>` - Normalized audio samples
fn normalize_audio(samples: &[f32]) -> Vec<f32> {
    let max_abs = samples.par_iter().map(|&s| s.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(1.0);
    samples.par_iter().map(|&s| s / max_abs).collect()
}

/// Resample audio
/// 
/// Resamples audio to 16kHz sample rate
/// 
/// # Parameters
/// * `samples` - Input audio samples
/// * `original_sample_rate` - Original sample rate
/// 
/// # Returns
/// * `Vec<f32>` - Resampled audio samples (16kHz)
#[allow(dead_code)]
fn resample_audio(samples: &[f32], original_sample_rate: u32) -> Vec<f32> {
    println!("Resampling from {} Hz to 16000 Hz", original_sample_rate);

    let params = SincInterpolationParameters {
        sinc_len: 512,
        f_cutoff: 0.98,
        interpolation: rubato::SincInterpolationType::Cubic,
        oversampling_factor: 512,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        16000.0 / original_sample_rate as f64,
        2.0,
        params,
        samples.len(),
        1,
    )
    .expect("Failed to create resampler");

    let resampled = resampler
        .process(&[samples.to_vec()], None)
        .expect("Resampling failed");

    resampled[0].clone()
}

/// Voice activity detection
/// 
/// Detects speech activity in audio, setting parts with energy below a threshold to silence
/// 
/// # Parameters
/// * `samples` - Input audio samples
/// * `frame_size` - Size of each analysis frame
/// * `threshold` - Energy threshold
/// 
/// # Returns
/// * `Vec<f32>` - Processed audio samples, silence parts set to zero
pub fn voice_activity_detection(samples: &[f32], frame_size: usize, threshold: f32) -> Vec<f32> {
    let mut result = Vec::with_capacity(samples.len());
    let mut prev_active = false;
    
    for chunk in samples.chunks(frame_size) {
        let energy = chunk.iter().map(|&s| s * s).sum::<f32>() / frame_size as f32;
        let is_active = energy > threshold;

        // Simple smoothing
        let should_keep = is_active || prev_active;
        prev_active = is_active;
        
        if should_keep {
            result.extend_from_slice(chunk);
        } else {
            result.extend(vec![0.0; chunk.len()]);
        }
    }
    
    result
}


/// Apply noise gate
/// 
/// Sets samples below a specified threshold to zero to reduce background noise
/// 
/// # Parameters
/// * `samples` - Input audio samples
/// * `noise_gate` - Noise gate threshold
/// 
/// # Returns
/// * `Vec<f32>` - Audio samples after applying noise gate
fn apply_noise_gate(samples: &[f32], noise_gate: f32) -> Vec<f32> {
    samples.par_iter()
        .map(|&s| if s.abs() < noise_gate { 0.0 } else { s })
        .collect()
}

fn hann_window(i: usize, size: usize) -> f32 {
    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
}

/// Smart noise reduction processing    
/// Selects the appropriate denoising algorithm based on audio characteristics
pub fn denoise_audio(samples: &[f32], config: &DenoiseConfig) -> Vec<f32> {
    let noise_type = analyze_noise_characteristics(samples, config.frame_size);
    
    match noise_type {
        NoiseType::Stationary => {
            spectral_subtraction(samples, config.frame_size, config.overlap, config.strength)
        }
        NoiseType::NonStationary => {
            wiener_filter(samples, config.frame_size, config.overlap, config.strength)
        }
        NoiseType::Mixed => {
            let spectral_result = spectral_subtraction(samples, config.frame_size, config.overlap, config.strength);
            wiener_filter(&spectral_result, config.frame_size, config.overlap, config.strength)
        }
    }
}

/// Noise type enumeration
#[derive(Debug)]
enum NoiseType {
    Stationary,    // Stationary noise (e.g., wind, electrical noise)
    NonStationary, // Non-stationary noise (e.g., burst noise)
    Mixed,         // Mixed noise
}

/// Analyze audio characteristics to determine noise type
fn analyze_noise_characteristics(samples: &[f32], frame_size: usize) -> NoiseType {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(frame_size);
    
    // Calculate power spectrum for each frame
    let frames = samples.chunks(frame_size);
    let mut spectral_variance = 0.0;
    let mut prev_spectrum: Option<Vec<f32>> = None;
    
    for frame in frames {
        if frame.len() == frame_size {
            let mut fft_input: Vec<Complex<f32>> = frame.iter()
                .enumerate()
                .map(|(i, &s)| Complex::new(s * hann_window(i, frame_size), 0.0))
                .collect();
            
            fft.process(&mut fft_input);
            
            // Calculate power spectrum
            let power_spectrum: Vec<f32> = fft_input.iter()
                .map(|c| c.norm_sqr())
                .collect();
            
            // Calculate spectral variance
            if let Some(prev) = &prev_spectrum {
                let frame_variance = power_spectrum.iter()
                    .zip(prev.iter())
                    .map(|(&curr, &prev)| (curr - prev).powi(2))
                    .sum::<f32>() / frame_size as f32;
                spectral_variance += frame_variance;
            }
            
            prev_spectrum = Some(power_spectrum);
        }
    }
    
    // Determine noise type based on spectral variance
    let normalized_variance = spectral_variance / samples.len() as f32;
    if normalized_variance < 0.1 {
        NoiseType::Stationary
    } else if normalized_variance > 0.5 {
        NoiseType::NonStationary
    } else {
        NoiseType::Mixed
    }
}

/// Spectral subtraction denoising
fn spectral_subtraction(samples: &[f32], frame_size: usize, overlap: f32, strength: f32) -> Vec<f32> {
    let step_size = (frame_size as f32 * (1.0 - overlap)) as usize;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(frame_size);
    let ifft = planner.plan_fft_inverse(frame_size);

    // Estimate noise spectrum
    let noise_spectrum = estimate_noise_spectrum(samples, frame_size, &fft);
    
    let frames = samples.windows(frame_size).step_by(step_size).collect::<Vec<_>>();
    let processed_frames: Vec<Vec<Complex<f32>>> = frames.par_iter().map(|frame| {
        let mut fft_input: Vec<Complex<f32>> = frame.iter()
            .enumerate()
            .map(|(i, &s)| Complex::new(s * hann_window(i, frame_size), 0.0))
            .collect();
        
        fft.process(&mut fft_input);
        
        // More gentle spectral subtraction
        for (i, complex) in fft_input.iter_mut().enumerate() {
            let power = complex.norm_sqr();
            let noise = noise_spectrum[i];
            
            // Very mild parameters
            let alpha = 1.0;  // Do not over-subtract noise
            let beta = 0.1;   // Preserve more signal
            
            // Frequency-dependent processing
            let freq_factor = (i as f32 / frame_size as f32).min(1.0);
            let freq_strength = strength * (1.0 - 0.3 * freq_factor); // High frequencies use smaller strength
            
            // Calculate gain
            let gain = ((1.0 - alpha * (noise / (power + 1e-6)).powf(freq_strength)).max(beta)).sqrt();
            *complex *= gain;
        }
        
        ifft.process(&mut fft_input);
        fft_input
    }).collect();

    // Overlap-add synthesis
    overlap_add(&processed_frames, samples.len(), step_size)
}

/// Wiener filter denoising
fn wiener_filter(samples: &[f32], frame_size: usize, overlap: f32, strength: f32) -> Vec<f32> {
    let step_size = (frame_size as f32 * (1.0 - overlap)) as usize;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(frame_size);
    let ifft = planner.plan_fft_inverse(frame_size);

    // Estimate noise and signal power spectra
    let noise_spectrum = estimate_noise_spectrum(samples, frame_size, &fft);
    let signal_spectrum = estimate_signal_spectrum(samples, frame_size, &fft);
    
    let frames = samples.windows(frame_size).step_by(step_size).collect::<Vec<_>>();
    let processed_frames: Vec<Vec<Complex<f32>>> = frames.par_iter().map(|frame| {
        let mut fft_input: Vec<Complex<f32>> = frame.iter()
            .enumerate()
            .map(|(i, &s)| Complex::new(s * hann_window(i, frame_size), 0.0))
            .collect();
        
        fft.process(&mut fft_input);
        
        // Wiener filter modification
        for (i, complex) in fft_input.iter_mut().enumerate() {
            let noise = noise_spectrum[i];
            let signal = signal_spectrum[i];
            
            // More conservative SNR estimation and gain calculation
            let snr = signal / (noise + 1e-6);
            let gain = (snr / (1.0 + snr)).powf(strength * 0.7); 
            *complex *= gain;
        }
        
        ifft.process(&mut fft_input);
        fft_input
    }).collect();

    // Overlap-add synthesis
    overlap_add(&processed_frames, samples.len(), step_size)
}

/// Estimate noise spectrum
fn estimate_noise_spectrum(samples: &[f32], frame_size: usize, fft: &Arc<dyn rustfft::Fft<f32>>) -> Vec<f32> {
    let num_frames = 20;  
    let mut noise_spectrum = vec![0.0; frame_size];
    
    for frame in samples.chunks(frame_size).take(num_frames) {
        if frame.len() == frame_size {
            let mut fft_input: Vec<Complex<f32>> = frame.iter()
                .enumerate()
                .map(|(i, &s)| Complex::new(s * hann_window(i, frame_size), 0.0))
                .collect();
            
            fft.process(&mut fft_input);
            
            for (i, complex) in fft_input.iter().enumerate() {
                noise_spectrum[i] += complex.norm_sqr() / num_frames as f32;
            }
        }
    }
    
    noise_spectrum
}

/// Estimate signal spectrum
fn estimate_signal_spectrum(samples: &[f32], frame_size: usize, fft: &Arc<dyn rustfft::Fft<f32>>) -> Vec<f32> {
    let mut signal_spectrum = vec![0.0; frame_size];
    let num_frames = samples.len() / frame_size;
    
    for frame in samples.chunks(frame_size) {
        if frame.len() == frame_size {
            let mut fft_input: Vec<Complex<f32>> = frame.iter()
                .enumerate()
                .map(|(i, &s)| Complex::new(s * hann_window(i, frame_size), 0.0))
                .collect();
            
            fft.process(&mut fft_input);
            
            for (i, complex) in fft_input.iter().enumerate() {
                signal_spectrum[i] += complex.norm_sqr() / num_frames as f32;
            }
        }
    }
    
    signal_spectrum
}

/// Overlap-add synthesis
fn overlap_add(frames: &[Vec<Complex<f32>>], output_len: usize, step_size: usize) -> Vec<f32> {
    let mut output = vec![0.0; output_len];
    let mut normalization = vec![0.0; output_len];
    let frame_size = frames[0].len();

    for (i, frame) in frames.iter().enumerate() {
        let start = i * step_size;
        for (j, &complex) in frame.iter().enumerate() {
            if start + j < output.len() {
                let window_val = hann_window(j, frame_size);
                output[start + j] += complex.re * window_val;
                normalization[start + j] += window_val * window_val;
            }
        }
    }

    // Normalize and amplify
    for i in 0..output.len() {
        if normalization[i] > 1e-10 {
            output[i] = (output[i] / normalization[i]) * 10.0;  // Increase amplification factor
        }
    }

    output
}

/// Estimate noise floor level by analyzing silent segments of audio
/// 
/// # Parameters
/// * `samples` - Input audio samples
/// 
/// # Returns
/// * `f32` - Estimated noise floor level
fn estimate_noise_floor(samples: &[f32]) -> f32 {
    // Split samples into small segments for analysis
    let frame_size = 1024;
    let mut frame_energies: Vec<f32> = samples
        .chunks(frame_size)
        .map(|frame| {
            frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32
        })
        .collect();
    
    // Sort energy values
    frame_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // Use the lowest 10% of frames for noise estimation
    let noise_frame_count = (frame_energies.len() as f32 * 0.1) as usize;
    let noise_frames = &frame_energies[..noise_frame_count];
    
    noise_frames.iter().sum::<f32>() / noise_frame_count as f32
}

/// Calculate Signal-to-Noise Ratio (SNR)
/// 
/// # Parameters
/// * `samples` - Input audio samples
/// * `noise_floor` - Estimated noise floor level
/// 
/// # Returns
/// * `f32` - SNR (in dB)
fn calculate_snr(samples: &[f32], noise_floor: f32) -> f32 {
    // Calculate average signal power
    let signal_power = samples.iter()
        .map(|&x| x * x)
        .sum::<f32>() / samples.len() as f32;
    
    // Avoid division by zero
    if noise_floor < 1e-10 {
        return 100.0; // If noise is almost zero, return a high SNR value
    }
    
    // Calculate SNR (in dB)
    10.0 * (signal_power / noise_floor).log10()
}

/// Audio quality evaluation results
#[derive(Debug)]
pub struct AudioQualityMetrics {
    pub snr: f32,              // SNR (dB)
    pub noise_level: f32,      // Noise level
    pub signal_level: f32,     // Signal level
    pub quality_score: f32,    // Overall quality score (0-100)
    pub needs_denoising: bool, // Whether denoising is needed
}

/// Evaluate audio quality
/// 
/// # Parameters
/// * `samples` - Input audio samples
/// 
/// # Returns
/// * `AudioQualityMetrics` - Audio quality evaluation results
pub fn evaluate_audio_quality(samples: &[f32]) -> AudioQualityMetrics {
    let noise_floor = estimate_noise_floor(samples);
    let signal_level = samples.iter()
        .map(|&x| x * x)
        .sum::<f32>() / samples.len() as f32;
    
    let snr = calculate_snr(samples, noise_floor);
    
    // Adjust quality score standard
    let quality_score = {
        let snr_score = (snr.min(30.0) / 30.0) * 40.0; // SNR contribution reduced to 40 points
        
        // Speech features score
        let speech_features = calculate_speech_features(samples);
        let speech_score = speech_features * 35.0; // Speech features contribute 35 points
        
        // Dynamic range score
        let dynamic_range = {
            let max = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
            let min = samples.iter().map(|&x| x.abs()).fold(f32::INFINITY, f32::min);
            ((max / (min + 1e-6)).log10() * 8.0).min(25.0) // Dynamic range contributes 25 points
        };
        
        snr_score + speech_score + dynamic_range
    };
    
    // More conservative denoising decision
    let needs_denoising = snr < 10.0 && quality_score < 50.0;
    
    AudioQualityMetrics {
        snr,
        noise_level: noise_floor,
        signal_level,
        quality_score,
        needs_denoising,
    }
}

/// Calculate speech features score
fn calculate_speech_features(samples: &[f32]) -> f32 {
    // Calculate zero-crossing rate
    let zero_crossings = samples.windows(2)
        .filter(|w| w[0].signum() != w[1].signum())
        .count() as f32 / samples.len() as f32;
    
    // Calculate short-term energy variance
    let energy_variance = {
        let frame_size = 512;
        let frame_energies: Vec<f32> = samples.chunks(frame_size)
            .map(|chunk| chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32)
            .collect();
        
        let mean_energy = frame_energies.iter().sum::<f32>() / frame_energies.len() as f32;
        let variance = frame_energies.iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f32>() / frame_energies.len() as f32;
        
        variance
    };
    
    // Overall score (normalized to 0-1 range)
    let zero_crossing_score = (zero_crossings * 1000.0).min(1.0);
    let energy_variance_score = (energy_variance * 100.0).min(1.0);
    
    (zero_crossing_score + energy_variance_score) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{WavSpec, WavWriter};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::f32::consts::PI;

    #[test]
    fn test_audio_processing() -> Result<()> {
        let input_path = Path::new("./test/a.wav");
        let (samples, num_channels, sample_rate) = read_wav_file(input_path)?;

        // Get timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Create output directory
        fs::create_dir_all("./test/processed").unwrap();

        // Test configurations
        let configs = vec![
            // (Denoising strength, Frame size, Overlap rate, Description)
            (0.5, 2048, 0.75, "mild"),
            (0.7, 2048, 0.75, "moderate"),
            (1.0, 2048, 0.75, "standard"),
            (0.5, 4096, 0.75, "large_frame"),
            (0.7, 2048, 0.85, "high_overlap"),
        ];

        for (strength, frame_size, overlap, desc) in configs {
            println!("\nProcessing configuration: {}", desc);
            println!("Strength: {}, Frame Size: {}, Overlap: {}", 
                    strength, frame_size, overlap);

            // Process with denoising
            let processed = denoise_audio(&samples, &DenoiseConfig::default());

            // Generate output file name
            let output_file_name = format!(
                "denoised_{}_s{}_f{}_o{}.wav",
                timestamp,
                (strength * 100.0) as i32,
                frame_size,
                (overlap * 100.0) as i32
            );
            let output_path = Path::new("./test/processed").join(output_file_name);

            // Save audio file
            save_audio_file(&processed, &output_path, num_channels, sample_rate)?;

            // Print audio statistics
            print_audio_stats(&samples, &processed, &output_path);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_parse_audio_file() -> Result<()> {
        let input_path = Path::new("./test/a.wav");
        let config = DenoiseConfig::default();
        let audio = parse_audio_file(input_path, &config, None)?;

        // Write audio data to file
        let output_path = Path::new("./test/processed/parsed.wav");
        save_audio_file(&audio, output_path, 1, 16000)?;

        Ok(())
    } 

    fn print_audio_stats(original: &[f32], processed: &[f32], output_path: &Path) {
        // Original audio statistics
        let original_rms = calculate_rms(original);
        let original_peak = original.iter()
            .map(|&x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        // Processed audio statistics
        let processed_rms = calculate_rms(processed);
        let processed_peak = processed.iter()
            .map(|&x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        // Noise type analysis
        let noise_type = analyze_noise_characteristics(original, 2048);

        println!("\nAudio Statistics:");
        println!("Detected noise type: {:?}", noise_type);
        println!("Original - RMS: {:.2} dB, Peak: {:.2} dB", 
                to_db(original_rms), to_db(original_peak));
        println!("Processed - RMS: {:.2} dB, Peak: {:.2} dB", 
                to_db(processed_rms), to_db(processed_peak));
        println!("Difference - RMS: {:.2} dB, Peak: {:.2} dB",
                to_db(processed_rms) - to_db(original_rms),
                to_db(processed_peak) - to_db(original_peak));
        println!("Output saved to: {:?}", output_path);
    }

    fn save_audio_file(
        samples: &[f32],
        output_path: &Path,
        num_channels: usize,
        sample_rate: u32,
    ) -> Result<()> {
        let spec = WavSpec {
            channels: num_channels as u16,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut writer = WavWriter::create(output_path, spec)
            .map_err(|e| anyhow::anyhow!("Failed to create WAV writer: {}", e))?;

        // Calculate RMS value
        let rms = calculate_rms(samples);
        
        // Target RMS value (-12dB)
        let target_rms = 0.25;  // Approximately -12dB
        
        // Calculate gain
        let gain = target_rms / rms;
        
        // Apply gain and convert to 16-bit integers
        for &sample in samples {
            let amplified = sample * gain;
            // Soft clipping
            let limited = if amplified > 1.0 {
                1.0
            } else if amplified < -1.0 {
                -1.0
            } else {
                amplified
            };
            
            let scaled_sample = (limited * 32767.0) as i16;
            writer.write_sample(scaled_sample)
                .map_err(|e| anyhow::anyhow!("Failed to write sample: {}", e))?;
        }

        writer.finalize()
            .map_err(|e| anyhow::anyhow!("Failed to finalize WAV file: {}", e))?;

        Ok(())
    }

    fn calculate_rms(samples: &[f32]) -> f32 {
        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    fn to_db(value: f32) -> f32 {
        20.0 * value.max(1e-10).log10()
    }

    #[test]
    fn test_audio_quality_evaluation() {
        // Generate test audio: clean sine wave
        let clean_signal: Vec<f32> = (0..44100)
            .map(|i| (i as f32 * 440.0 * 2.0 * PI / 44100.0).sin())
            .collect();
            
        // Add noise to the signal
        let noisy_signal: Vec<f32> = clean_signal.iter()
            .map(|&x| x + rand::random::<f32>() * 0.1)
            .collect();
            
        // Evaluate quality
        let clean_metrics = evaluate_audio_quality(&clean_signal);
        let noisy_metrics = evaluate_audio_quality(&noisy_signal);
        
        // Verify results
        assert!(clean_metrics.snr > noisy_metrics.snr);
        assert!(clean_metrics.quality_score > noisy_metrics.quality_score);
        assert!(!clean_metrics.needs_denoising);
        assert!(noisy_metrics.needs_denoising);
        
        println!("Clean audio metrics: {:?}", clean_metrics);
        println!("Noisy audio metrics: {:?}", noisy_metrics);
    }
}
