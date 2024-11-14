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
    // 读取输入WAV文件
    let mut reader = WavReader::open(&args.input)
        .context("Failed to open input file")?;
    let spec = reader.spec();

    println!("Input audio spec: {:?}", spec);

    // 读取所有采样点
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()?,
        SampleFormat::Int => reader.samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<f32>, _>>()?,
    };

    // 保存原始样本数量用于统计
    let original_sample_count = samples.len();

    // 如果是立体声，转换为单声道
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

    // 保存单声道样本数量用于统计
    let mono_sample_count = mono_samples.len();

    // 重采样到16kHz
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

    // 转换为16bit PCM
    let pcm_data: Vec<u8> = output_samples.iter()
        .flat_map(|&sample| {
            let clamped = sample.max(-1.0).min(1.0);
            let sample_i16 = (clamped * 32767.0).round() as i16;
            sample_i16.to_le_bytes().to_vec()
        })
        .collect();

    println!("Audio conversion completed. Connecting to ASR server...");

    // 添加音频统计信息
    println!("\nAudio statistics:");
    println!("Original sample rate: {} Hz", spec.sample_rate);
    println!("Original channels: {}", spec.channels);
    println!("Original bits per sample: {}", spec.bits_per_sample);
    println!("Original sample count: {}", original_sample_count);
    println!("Mono sample count: {}", mono_sample_count);
    println!("Resampled sample count: {}", output_samples.len());
    println!("PCM data size: {} bytes", pcm_data.len());
    
    // 检查音频振幅
    let max_amplitude = output_samples.iter().fold(0f32, |a, &b| a.max(b.abs()));
    println!("Max amplitude: {}", max_amplitude);
    if max_amplitude > 1.0 {
        println!("Warning: Audio may be clipping!");
    } else if max_amplitude < 0.1 {
        println!("Warning: Audio may be too quiet!");
    }

    // 检查音频时长
    let duration = output_samples.len() as f32 / 16000.0;
    println!("Audio duration: {:.2} seconds", duration);
    if duration < 0.5 {
        println!("Warning: Audio may be too short!");
    }

    // 检查PCM数据
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

    // 检查每个块的大小
    for (i, chunk) in pcm_data.chunks(32 * 1024).enumerate() {
        println!("Chunk {}: {} bytes", i, chunk.len());
    }

    // 在发送之前检查 PCM 数据
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

    // 检查一些样本值
    for (i, chunk) in pcm_data.chunks(2).take(10).enumerate() {
        if chunk.len() == 2 {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            println!("Sample {}: {}", i, sample);
        }
    }

    // 连接到服务器
    let mut client = AsrClient::connect(args.server).await?;

    // 创建请求数据
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

    // 创建请求流
    let outbound = stream::iter(chunks);

    // 发送请求并获取响应流
    println!("\n开始识别...");
    let request = Request::new(outbound);
    let response = client.transcribe(request).await?;
    let mut inbound = response.into_inner();

    // 使用新的流式处理逻辑处理响应
    println!("正在接收识别结果...");
    
    // 使用 while let Some 来处理流式响应
    while let Some(response) = inbound.message().await? {
        let text = String::from_utf8_lossy(&response.text);
        if !text.trim().is_empty() {
            print!("\r"); // 清除当前行
            println!("实时识别: {}", text.trim());
        }
        
        if response.end == 1 {
            println!("\n识别完成!");
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