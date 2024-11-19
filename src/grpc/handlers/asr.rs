use crate::asr::{AsrEngine, AsrParams};
use tonic::{async_trait, Status, Request, Response};
use crate::grpc::pb::asr::{TranscribeRequest, TranscribeResponse, Segment, asr_server::Asr};
use std::pin::Pin;
use tokio_stream::{Stream, StreamExt};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error};
use crate::asr::TranscribeSegment;
use crate::audio::{DenoiseConfig, denoise_audio};

// Sample rate
const SAMPLE_RATE: usize = 16000;
// Save 5 seconds
const CHUNK_SIZE: usize = SAMPLE_RATE * 10;
// Overlap 1 second
const OVERLAP_SIZE: usize = SAMPLE_RATE;

pub struct AsrService {
    asr_engine: Arc<dyn AsrEngine>,
}

struct StreamContext {
    block_index: usize,  
    last_text: String,
    last_end_time: f64,  
}

impl StreamContext {
    fn new() -> Self {
        Self {
            block_index: 0,
            last_text: String::new(),
            last_end_time: 0.0,
        }
    }

    fn calculate_segment_time(&mut self, segment_start: f64, segment_end: f64) -> (i64, i64) {
        let block_base_time = self.block_index as f64 * 5.0;
        
        let mut abs_start = ((block_base_time + segment_start) * 1000.0) as i64;
        let mut abs_end = ((block_base_time + segment_end) * 1000.0) as i64;
        
        let last_end_ms = (self.last_end_time * 1000.0) as i64;
        if abs_start < last_end_ms {
            let diff = last_end_ms - abs_start;
            abs_start = last_end_ms;
            abs_end += diff;  
        }
        
        self.last_end_time = abs_end as f64 / 1000.0;
        
        (abs_start, abs_end)
    }

    fn next_block(&mut self) {
        self.block_index += 1;
    }
}

impl AsrService {
    pub fn new(asr_engine: Arc<dyn AsrEngine>) -> Self {
        debug!("Creating new ASR service");
        Self { asr_engine }
    }

    // Improved text processing logic, preserving punctuation
    fn process_text(new_text: &str, last_text: &str, segments: &[TranscribeSegment]) -> Option<String> {
        if last_text.is_empty() {
            return Some(new_text.to_string());
        }

        // If there's a new segment, check the last segment
        if let Some(last_segment) = segments.last() {
            // If the last segment's text is not in the previous text, it's new content
            if !last_text.contains(&last_segment.text) {
                return Some(last_segment.text.clone());
            }
        }

        // If the new text is longer than the old text and starts with the old text, return the added part (preserving punctuation)
        if new_text.len() > last_text.len() && new_text.starts_with(last_text) {
            let added_text = &new_text[last_text.len()..];
            if !added_text.trim().is_empty() {
                return Some(added_text.trim().to_string());
            }
        }

        // If the new text is completely different and the length difference is large, return the new text
        if new_text.len() > last_text.len() * 2 || last_text.len() > new_text.len() * 2 {
            return Some(new_text.to_string());
        }

        // Compare text differences, while considering punctuation
        if new_text != last_text {
            // Split the text into sentences
            let new_sentences: Vec<&str> = new_text.split(['。', '！', '？', '.', '!', '?'])
                .filter(|s| !s.trim().is_empty())
                .collect();
            let last_sentences: Vec<&str> = last_text.split(['。', '！', '？', '.', '!', '?'])
                .filter(|s| !s.trim().is_empty())
                .collect();

            // If the number of sentences is different or the last sentence is different, return the new part
            if new_sentences.len() > last_sentences.len() {
                let new_content = new_sentences[last_sentences.len()..]
                    .join("")
                    .trim()
                    .to_string();
                if !new_content.is_empty() {
                    // Ensure the returned text contains appropriate punctuation
                    let mut result = new_content;
                    if let Some(last_char) = new_text.chars().last() {
                        if "。！？.!?".contains(last_char) {
                            result.push(last_char);
                        }
                    }
                    return Some(result);
                }
            } else if let (Some(last_new), Some(last_old)) = (new_sentences.last(), last_sentences.last()) {
                if last_new.trim() != last_old.trim() {
                    let mut result = last_new.trim().to_string();
                    // Add the last punctuation mark
                    if let Some(last_char) = new_text.chars().last() {
                        if "。！？.!?".contains(last_char) {
                            result.push(last_char);
                        }
                    }
                    return Some(result);
                }
            }
        }

        None
    }

}

type ResponseStream = Pin<Box<dyn Stream<Item = Result<TranscribeResponse, Status>> + Send>>;

#[async_trait]
impl Asr for AsrService {
    type TranscribeStream = ResponseStream;

    async fn transcribe(
        &self,
        request: Request<tonic::Streaming<TranscribeRequest>>,
    ) -> Result<Response<Self::TranscribeStream>, Status> {
        debug!("Starting new transcription request");
        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel(32);
        
        let mut params = AsrParams::new();
        params.set_language(Some("zh".to_string()));
        params.set_stream_mode(true);
        params.set_min_segment_length(5);
        
        let mut audio_buffer = Vec::new();
        let mut device_id = String::new();
        let asr_engine = self.asr_engine.clone();
        let mut stream_ctx = StreamContext::new();

        let state = match asr_engine.create_state() {
            Ok(state) => state,
            Err(e) => return Err(Status::internal(e.to_string())),
        };

        tokio::spawn(async move {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(req) => {
                        if device_id.is_empty() {
                            device_id = req.device_id.clone();
                        }

                        let decoded_audio = match BASE64.decode(req.audio) {
                            Ok(audio) => audio,
                            Err(e) => {
                                error!("Failed to decode audio: {}", e);
                                continue;
                            }
                        };

                        audio_buffer.extend(decoded_audio);

                        if audio_buffer.len() >= CHUNK_SIZE {
                            let float_data: Vec<f32> = audio_buffer[..CHUNK_SIZE]
                                .chunks_exact(2)
                                .map(|chunk| {
                                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                                    sample as f32 / 32767.0
                                })
                                .collect();

                            let denoised_data = denoise_audio(&float_data, &DenoiseConfig::default());
                            
                            match asr_engine.transcribe_with_state(state.clone(), denoised_data, params.clone()).await {
                                Ok(result) => {
                                    for segment in result.segments {
                                        if let Some(new_text) = Self::process_text(&segment.text, &stream_ctx.last_text, &[segment.clone()]) {
                                            stream_ctx.last_text = segment.text.clone();
                                            
                                            // 计算当前 segment 的绝对时间
                                            let (start_time, end_time) = stream_ctx.calculate_segment_time(segment.start, segment.end);
                                            
                                            let adjusted_segment = Segment {
                                                start: start_time,
                                                end: end_time,
                                                text: segment.text.as_bytes().to_vec(),
                                            };

                                            if let Err(e) = tx.send(Ok(TranscribeResponse {
                                                device_id: device_id.clone(),
                                                text: new_text.as_bytes().to_vec(),
                                                end: 0,
                                                segments: vec![adjusted_segment],
                                            })).await {
                                                error!("Failed to send response: {}", e);
                                                break;
                                            }
                                        }
                                    }
                                    
                                    // 处理完当前块后，移动到下一个块
                                    stream_ctx.next_block();
                                }
                                Err(e) => error!("ASR processing failed: {}", e),
                            }

                            audio_buffer = audio_buffer[CHUNK_SIZE - OVERLAP_SIZE..].to_vec();
                        }

                        if req.end == 1 && !audio_buffer.is_empty() {
                            let float_data: Vec<f32> = audio_buffer
                                .chunks(2)
                                .map(|chunk| {
                                    if chunk.len() == 2 {
                                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                                        sample as f32 / 32767.0
                                    } else {
                                        0.0
                                    }
                                })
                                .collect();

                            if let Ok(result) = asr_engine.transcribe(float_data, params.clone()).await {
                                if let Some(final_text) = Self::process_text(&result.full_text, &stream_ctx.last_text, &result.segments) {
                                    let adjusted_segments: Vec<Segment> = result.segments.iter().map(|seg| {
                                        let (start_time, end_time) = stream_ctx.calculate_segment_time(seg.start, seg.end);
                                        Segment {
                                            start: start_time,
                                            end: end_time,
                                            text: seg.text.as_bytes().to_vec(),
                                        }
                                    }).collect();

                                    if let Err(e) = tx.send(Ok(TranscribeResponse {
                                        device_id: device_id.clone(),
                                        text: final_text.as_bytes().to_vec(),
                                        end: 1,
                                        segments: adjusted_segments,
                                    })).await {
                                        error!("Failed to send final response: {}", e);
                                    }
                                }
                            }
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Error receiving request: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(Response::new(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))))
    }
}
