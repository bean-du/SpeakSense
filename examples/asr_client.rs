use std::path::PathBuf;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use clap::Parser;
use hound::{WavReader, SampleFormat};
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters};
use anyhow::{Result, Context};
use futures_util::stream;
use tokio_stream::StreamExt;

use tonic::Request;
use asr_rs::grpc::pb::asr::{TranscribeRequest, asr_client::AsrClient};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input audio file path (WAV format)
    #[arg(short, long)]
    input: PathBuf,

    /// Device ID for the request
    #[arg(short, long, default_value = "test-device")]
    device_id: String,

    /// Server address
    #[arg(short, long, default_value = "http://[::1]:50051")]
    server: String,
}

async fn convert_and_send_audio(args: Args) -> Result<()> {
    // read input WAV file
    let mut reader = WavReader::open(&args.input)
        .context("Failed to open input file")?;
    let spec = reader.spec();

    println!("Input audio spec: {:?}", spec);

    // read all samples
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()?,
        SampleFormat::Int => reader.samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<f32>, _>>()?,
    };

    // save original sample count for statistics
    let original_sample_count = samples.len();

    // if stereo, convert to mono
    let mono_samples = if spec.channels == 2 {
        println!("Converting stereo to mono...");
        let mut mono = Vec::with_capacity(samples.len() / 2);
        for chunk in samples.chunks(2) {
            mono.push((chunk[0] + chunk[1]) / 2.0);
        }
        mono
    } else {
        samples
    };

    // save mono sample count for statistics
    let mono_sample_count = mono_samples.len();

    // resample to 16kHz
    let output_samples = if spec.sample_rate != 16000 {
        println!("Resampling from {}Hz to 16000Hz...", spec.sample_rate);
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: rubato::WindowFunction::BlackmanHarris2,
        };

        let resample_ratio = 16000 as f64 / spec.sample_rate as f64;
        let mut resampler = SincFixedIn::<f32>::new(
            resample_ratio,
            1.1,
            params,
            mono_samples.len(),
            1,
        ).context("Failed to create resampler")?;

        let waves_in = vec![mono_samples];
        let mut waves_out = resampler.process(&waves_in, None)
            .context("Failed to resample audio")?;
        waves_out.pop().unwrap()
    } else {
        mono_samples
    };

    // convert to 16bit PCM
    let pcm_data: Vec<u8> = output_samples.iter()
        .flat_map(|&sample| {
            let clamped = sample.max(-1.0).min(1.0);
            let sample_i16 = (clamped * 32767.0).round() as i16;
            sample_i16.to_le_bytes().to_vec()
        })
        .collect();

    println!("Audio conversion completed. Connecting to ASR server...");

    println!("\nAudio statistics:");
    println!("Original sample rate: {} Hz", spec.sample_rate);
    println!("Original channels: {}", spec.channels);
    println!("Original bits per sample: {}", spec.bits_per_sample);
    println!("Original sample count: {}", original_sample_count);
    println!("Mono sample count: {}", mono_sample_count);
    println!("Resampled sample count: {}", output_samples.len());
    println!("PCM data size: {} bytes", pcm_data.len());
    
    let max_amplitude = output_samples.iter().fold(0f32, |a, &b| a.max(b.abs()));
    println!("Max amplitude: {}", max_amplitude);
    if max_amplitude > 1.0 {
        println!("Warning: Audio may be clipping!");
    } else if max_amplitude < 0.1 {
        println!("Warning: Audio may be too quiet!");
    }

    let duration = output_samples.len() as f32 / 16000.0;
    println!("Audio duration: {:.2} seconds", duration);
    if duration < 0.5 {
        println!("Warning: Audio may be too short!");
    }

    let non_zero_samples = pcm_data.chunks(2)
        .filter(|chunk| {
            if chunk.len() == 2 {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample != 0
            } else {
                false
            }
        })
        .count();

    println!("Non-zero samples: {} ({:.2}%)", 
        non_zero_samples, 
        (non_zero_samples as f32 / (pcm_data.len() / 2) as f32) * 100.0
    );

    for (i, chunk) in pcm_data.chunks(32 * 1024).enumerate() {
        println!("Chunk {}: {} bytes", i, chunk.len());
    }

    let non_zero_count = pcm_data.chunks(2)
        .filter(|chunk| {
            if chunk.len() == 2 {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample != 0
            } else {
                false
            }
        })
        .count();

    println!("Client side - Non-zero samples: {}/{}", non_zero_count, pcm_data.len() / 2);

    for (i, chunk) in pcm_data.chunks(2).take(10).enumerate() {
        if chunk.len() == 2 {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            println!("Sample {}: {}", i, sample);
        }
    }

    let mut client = AsrClient::connect(args.server).await?;

    let device_id = args.device_id;
    let chunks: Vec<_> = pcm_data.chunks(32 * 1024)
        .enumerate()
        .map(|(i, chunk)| {
            let is_last_chunk = (i + 1) * 32 * 1024 >= pcm_data.len();
            TranscribeRequest {
                audio: BASE64.encode(chunk).into_bytes(),
                device_id: device_id.clone(),
                r#type: 0,
                end: if is_last_chunk { 1 } else { 0 },
            }
        })
        .collect();

    // create request stream
    let outbound = stream::iter(chunks);

    println!("\nStart transcribing...");
    let request = Request::new(outbound);
    let response = client.transcribe(request).await?;
    let mut inbound = response.into_inner();

    println!("Receiving recognition results...");
    
    // process stream response
    while let Some(response) = inbound.message().await? {
        let text = String::from_utf8_lossy(&response.text);
        if !text.trim().is_empty() {
            print!("\r"); 
            // print segments info
            for segment in &response.segments {
                let segment_text = String::from_utf8_lossy(&segment.text);
                println!("  Segment: {} -> {} Text: {}", 
                    format_timestamp(segment.start),
                    format_timestamp(segment.end),
                    segment_text.trim()
                );
            }
        }
        
        if response.end == 1 {
            println!("\nRecognition completed!");
            break;
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    convert_and_send_audio(args).await
}

// format timestamp
fn format_timestamp(ms: i64) -> String {
    let seconds = ms as f64 / 1000.0; 
    format!("{:.2}ms", seconds)
} 