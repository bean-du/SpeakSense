use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};
use anyhow::Result;
use crate::asr::{AsrEngine, AsrParams, TranscribeResult, TranscribeSegment};
use tracing::debug;
use std::sync::{Arc, Mutex};
use std::mem;


const PROMOTIONAL_TEXT: [&str; 14] = [
    "请不吝点赞", "請不吝點贊", "點贊", "訂閱", "订阅", "打赏", "打賞", "打賞支持明鏡與點點欄目", "打赏支持明镜与点点栏目",
    "並且按下小鈴鐺才能收到最新消息哦!", "請按讚、訂閱、分享!", "明镜需要您的支持 欢迎收看订阅明镜",
    "請按讚,訂閱,分享,打開小鈴鐺,並且按下小鈴鐺才能收到最新消息謝謝觀看",
    "請按讚,訂閱,分享,打開小鈴鐺,並且按下小鈴鐺才能收到最新消息哦!"
];

pub struct WhisperAsr {
    whisper_ctx: Arc<WhisperContext>,
}

impl WhisperAsr {
    pub fn new(model_path: String) -> Result<Self> {
        debug!("Initializing WhisperAsr with model: {}", model_path);
        let whisper_ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
            .map_err(|e| anyhow::anyhow!("failed to open whisper model: {}", e))?;
        Ok(Self { 
            whisper_ctx: Arc::new(whisper_ctx)
        })
    }

    pub fn create_state(&self) -> Result<Arc<Mutex<Box<WhisperState<'static>>>>> {
        let state = self.whisper_ctx.create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create whisper state: {}", e))?;
        
        let state = unsafe {
            mem::transmute::<Box<WhisperState<'_>>, Box<WhisperState<'static>>>(Box::new(state))
        };
        
        Ok(Arc::new(Mutex::new(state)))
    }

    fn is_promotional_text(&self, text: &str) -> bool {
        PROMOTIONAL_TEXT.iter().any(|&promo| text.contains(promo))
    }

    pub async fn transcribe_with_state(
        &self,
        state: Arc<Mutex<Box<WhisperState<'static>>>>,
        audio: Vec<f32>,
        user_params: AsrParams
    ) -> Result<TranscribeResult> {
        let mut state_guard = state.lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock state: {}", e))?;
        
        let state = state_guard.as_mut();
        
        debug!("Starting transcription with audio length: {}", audio.len());
        debug!("Audio data checksum: {:x}", calculate_checksum(&audio));
        
        let mut params = self.build_params(user_params.clone());
        
        if let Some(ref lang) = user_params.language {
            debug!("Setting language to: {}", lang);
            params.set_language(Some(lang.as_str()));
        }

        if user_params.stream_mode {
            params.set_single_segment(true);
            params.set_no_context(true);
            params.set_audio_ctx(0);
        }

        debug!("Running full inference");
        state.full(params, &audio)?;
        
        let num_segments = state.full_n_segments()?;
        debug!("Got {} segments", num_segments);

        let mut segments = Vec::new();
        let mut full_text = String::new();
        let mut current_speaker = 0;

        for i in 0..num_segments {
            let text = state.full_get_segment_text(i)?;
            
            if self.is_promotional_text(&text) {
                debug!("Filtered out promotional text segment: {}", text);
                continue;
            }
            
            let start = state.full_get_segment_t0(i)?;
            let end = state.full_get_segment_t1(i)?;
            
            if i > 0 && state.full_get_segment_speaker_turn_next(i - 1) {
                current_speaker += 1;
            }

            let processed_text = self.add_punctuation(&text);
            debug!("Segment {}: {} -> {} : {}", i, start, end, processed_text);
            
            if user_params.stream_mode {
                if i == num_segments - 1 {
                    segments.push(TranscribeSegment {
                        text: processed_text.clone(),
                        speaker_id: current_speaker,
                        start: start as f64,
                        end: end as f64,
                    });
                    full_text = processed_text;
                }
            } else {
                segments.push(TranscribeSegment {
                    text: processed_text.clone(),
                    speaker_id: current_speaker,
                    start: start as f64,
                    end: end as f64,
                });
                full_text.push_str(&processed_text);
            }
        }

        debug!("Transcription completed, full text length: {}", full_text.len());
        
        Ok(TranscribeResult {
            segments,
            full_text,
        })
    }

    fn build_params(&self, ap: AsrParams) -> FullParams {
        let mut params = FullParams::new(SamplingStrategy::Greedy { 
            best_of: 5

        });

         if ap.speaker_diarization {
            params.set_tdrz_enable(true);        // 启用说话人检测
        }

        
        params.set_n_threads(16);
        params.set_audio_ctx(1500);
        params.set_print_timestamps(true);

        params.set_single_segment(false);
        
        params.set_print_progress(true);
        params.set_print_realtime(true);
        params.set_print_special(false);           // 启用特殊标记输出
        params.set_suppress_non_speech_tokens(false);  // 不抑制非语音标记（包括标点）
        params.set_max_initial_ts(1.0);          // 允许更多的初始时间戳
        
        params.set_no_context(false);            // 启用上下文
        params.set_single_segment(false);        // 禁用单段模式
        params.set_token_timestamps(true);       // 启用时间戳
        params.set_split_on_word(true);         // 按词分割，有助��更好的标点处理
        
        params.set_temperature(0.0);             // 使用确定性输出
        params.set_entropy_thold(2.4);           // 设置熵阈值
        params.set_logprob_thold(-1.0);         // 设置对数概率阈值
        params.set_no_speech_thold(0.6);        // 设置无语音阈值
        
        params.set_max_len(0);                  // 不限制段落长度
        params.set_max_tokens(0);               // 不限制标记数量
        params.set_speed_up(false);             // 不启用加速以保证质量
        
        params.set_thold_pt(0.01);              // 时间戳标记概率阈值
        params.set_thold_ptsum(0.01);           // 时间戳标记概率和阈值
        params.set_length_penalty(-1.0);        // 长度惩罚

        params
    }

    fn add_punctuation(&self, text: &str) -> String {
        if text.ends_with(['。', '！', '？', '，']) {
            return text.to_string();
        }

        let mut result = String::with_capacity(text.len() + 1);
        
        let contains_question = text.contains("吗") || text.contains("呢") || 
                              text.contains("什么") || text.contains("为何") || 
                              text.contains("怎么");
        
        let contains_exclaim = text.contains("啊") || text.contains("哇") || 
                              text.contains("太") || text.contains("真") ||
                              text.contains("好") || text.contains("真是");

        result.push_str(&text);
        
        if contains_question {
            result.push('？');
        } else if contains_exclaim {
            result.push('！');
        } else {
            result.push(' ');
        }

        result
    }
}

#[async_trait::async_trait]
impl AsrEngine for WhisperAsr {
    fn create_state(&self) -> Result<Arc<Mutex<Box<WhisperState<'static>>>>> {
        self.create_state()
    }

    async fn transcribe_with_state(
        &self,
        state: Arc<Mutex<Box<WhisperState<'static>>>>,
        audio: Vec<f32>,
        user_params: AsrParams
    ) -> Result<TranscribeResult> {
        self.transcribe_with_state(state, audio, user_params).await
    }

    async fn transcribe(&self, audio: Vec<f32>, params: AsrParams) -> Result<TranscribeResult> {
        let state = self.create_state()?;
        self.transcribe_with_state(state, audio, params).await
    }
}

fn calculate_checksum(audio: &[f32]) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    
    let mut hasher = DefaultHasher::new();
    for sample in audio.iter() {
        sample.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}
