# SpeakSense ASR Server

[English](README.md) | [中文](docs/README_zh.md)

A high-performance ASR (Automatic Speech Recognition) server implementation using Whisper, supporting both gRPC and REST APIs.

## Overview
This project provides a server implementation for speech-to-text transcription using OpenAI's Whisper model, optimized for different platforms and hardware acceleration options.

### Features
- [x] gRPC Server
  - [x] Stream Transcription
- [x] Web API
  - [x] Task Management
  - [x] Task Status
  - [x] Create Task By URL
  - [x] Create Task By Local File
  - [x] Authentication API Key Management 
- [x] Schedule Task
  - [x] Download Audio File
  - [x] Transcription
  - [x] Http Callback
- [x] Authentication
- [x] Multiple Platform Support
  - [x] MacOS (Metal)
  - [x] Linux (CUDA)
  - [x] Windows (CUDA)

## Quick Start

### Prerequisites
- Rust toolchain (1.70 or later)
- For CUDA support: CUDA toolkit 11.x or later
- For Metal support (MacOS): XCode and Metal SDK
- etcd server running locally or accessible (unnecessary, only for microservice go-micro)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bean-du/SpeakSense
cd SpeakSense
```

2. Download the Whisper model:
```bash
./script/download-ggml-model.sh
```

3. Build the project:
```bash
# Standard build
cargo build --release

# With CUDA support
cargo build --release --features cuda

# With Metal support (MacOS)
cargo build --release --features metal
```

### Environment Variables
- `ASR_SQLITE_PATH` SQLite Path (default: `sqlite://./asr_data/database/storage.db?mode=rwc`)
- `ASR_AUDIO_PATH` Audio Path (default: `./asr_data/audio/`)
- `ETCD_DEFAULT_ENDPOINT` Etcd Endpoint (default: `http://localhost:2379`)
- `ASR_MODEL_PATH` Whisper Model Path (default: `./models/ggml-large-v3.bin`)

### Running the Server

#### Standard Run (CPU)
```bash
cargo run --release
```

#### Run with CUDA Support
```bash
cargo run --release --features cuda
```

#### Run with Metal Support (MacOS)
First, set the Metal resources path:
```bash
export GGML_METAL_PATH_RESOURCES="./resources"
cargo run --release --features metal
```

### Docker Compose Quick Start
> docker Only support linux cuda x86_64 now
The easiest way to get started is using Docker Compose:

1. Create required directories:
```bash
mkdir -p models asr_data/audio asr_data/database
```

2. Download the Whisper model:
```bash
./script/download-ggml-model.sh
```

3. Start the server:
```bash
# Standard version
docker-compose up -d

# With CUDA support
ASR_FEATURES=cuda docker-compose up -d

# With Metal support (MacOS)
ASR_FEATURES=metal docker-compose up -d
```

4. Check the logs:
```bash
docker-compose logs -f
```

5. Stop the server:
```bash
docker-compose down
```

The server will be available at:
- REST API: http://localhost:7200
- gRPC: localhost:7300

### Docker Compose Configuration

The default configuration includes:
- Automatic volume mapping for models and data persistence
- GPU support (when using CUDA feature)
- Optional etcd service
- Environment variable configuration

You can customize the configuration by:
1. Modifying environment variables in docker-compose.yml
2. Adding or removing services as needed
3. Adjusting resource limits and port mappings

## Usage Examples

### gRPC Client Test
```bash
# Use local wav file
cargo run --example asr_client -- -i 2.wav

# Specify server address
cargo run --example asr_client -- -i test/2.wav -s http://127.0.0.1:7300

# Specify device id
cargo run --example asr_client -- -i input.wav -d test-device
```

### REST API Examples

#### Create Transcription Task
```bash
curl -X POST http://localhost:7200/api/v1/asr/tasks \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "https://example.com/audio.wav"}'
```

#### Check Task Status
```bash
curl http://localhost:7200/api/v1/asr/tasks/{task_id} \
  -H "Authorization: Bearer your-api-key"
```

## Configuration

### Model Selection
The server supports various Whisper model sizes. You can download different models from Hugging Face:
https://huggingface.co/ggerganov/whisper.cpp/tree/main

### Performance Tuning
- For CUDA: Adjust batch size and worker threads based on your GPU memory
- For Metal: Ensure proper resource path configuration

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- OpenAI Whisper
- whisper.cpp
- whisper-rs