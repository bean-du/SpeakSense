use crate::asr::{AsrEngine, AsrParams};
use tonic::{async_trait, Status, Request, Response};
use crate::grpc::pb::asr::{TranscribeRequest, TranscribeResponse, asr_server::Asr};
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
        params.set_stream_mode(true);  // Ensure stream mode is enabled
        params.set_min_segment_length(5);  // Set a smaller segment length to return more frequently
        debug!("ASR params: {:?}", params);

        let mut audio_buffer = Vec::new();
        let mut device_id = String::new();
        let mut last_text = String::new();
        let asr_engine = self.asr_engine.clone();

        let state = match asr_engine.create_state() {
            Ok(state) => state,
            Err(e) => return Err(Status::internal(e.to_string())),
        };

        // Start an async task to process the audio stream
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

                        debug!("Received audio chunk: {} bytes", decoded_audio.len());
                        audio_buffer.extend(decoded_audio);

                        // Process when enough audio data is accumulated
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
                                    // Process each new text segment
                                    for segment in result.segments {
                                        if let Some(new_text) = Self::process_text(&segment.text, &last_text, &[segment.clone()]) {
                                            last_text = segment.text;
                                            
                                            // Immediately send the newly recognized text
                                            if let Err(e) = tx.send(Ok(TranscribeResponse {
                                                device_id: device_id.clone(),
                                                text: new_text.as_bytes().to_vec(),
                                                end: 0,
                                            })).await {
                                                error!("Failed to send response: {}", e);
                                                break;
                                            }
                                        }
                                    }
                                }
                                Err(e) => error!("ASR processing failed: {}", e),
                            }

                            // Keep the remaining audio data
                            audio_buffer = audio_buffer[CHUNK_SIZE - OVERLAP_SIZE..].to_vec();
                        }

                        // Process the remaining audio data
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
                                if let Some(final_text) = Self::process_text(&result.full_text, &last_text, &result.segments) {
                                    if let Err(e) = tx.send(Ok(TranscribeResponse {
                                        device_id: device_id.clone(),
                                        text: final_text.as_bytes().to_vec(),
                                        end: 1,
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
