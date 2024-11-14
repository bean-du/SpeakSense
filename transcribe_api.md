


## ASR Service

### 1. Audio Transcription
**POST** `/asr/transcribe`

Request Header:
| Field | Type | Description |
|------|------|------|
| Authorization | String | Bearer `API Key` |

Request Body:
| Field | Type | Description |
|------|------|------|
| path | String | Audio file path or URL |
| path_type | String | Path Type (url, local) |
| callback_url | String | Callback URL |
| language | optional | Language code (zh, en, ...) |
| speaker_diarization | bool | Speaker Diarization |
| emotion_recognition | bool | Emotion Recognition |
| filter_dirty_words | bool | Filter Dirty Words |


### 2. Callbacks
Request Body
| Field | Type | Description |
|------|------|------|
| task_id | String | Task ID |
| status | String | Task Status |
| result.text | String | Transcription Text |
| result.segments.text | String | Transcription Segment Text |
| result.segments.speaker_id | Number | Transcription Segment Speaker ID |
| result.segments.start_time | Number | Transcription Segment Start Time |
| result.segments.end_time | Number | Transcription Segment End Time |
