# SpeakSense ASR 服务器

一个使用 Whisper 实现的高性能 ASR（自动语音识别）服务器，支持 gRPC 和 REST API。

## 概述
本项目提供了一个使用 OpenAI Whisper 模型的语音转文字服务器实现，针对不同平台和硬件加速选项进行了优化。

### 功能特性
- [x] gRPC 服务器
  - [x] 流式转录
- [x] Web API
  - [x] 任务管理
  - [x] 任务状态查询
  - [x] 通过 URL 创建任务
  - [x] 通过本地文件创建任务
  - [x] API 密钥认证管理
- [x] 任务调度
  - [x] 音频文件下载
  - [x] 语音转录
  - [x] HTTP 回调
- [x] 身份认证
- [x] 多平台支持
  - [x] MacOS (Metal)
  - [x] Linux (CUDA)
  - [x] Windows (CUDA)

## 快速开始

### 环境要求
- Rust 工具链 (1.70 或更高版本)
- CUDA 支持需要: CUDA toolkit 11.x 或更高版本
- Metal 支持需要 (MacOS): XCode 和 Metal SDK
- etcd 服务器在本地运行或可访问 (仅用于微服务 go-micro)

### 安装步骤

1. 克隆仓库:
```bash
git clone https://github.com/yourusername/whisper-asr-server.git
cd whisper-asr-server
```

2. 下载 Whisper 模型:
```bash
./script/download-ggml-model.sh
```

3. 构建项目:
```bash
# 标准构建
cargo build --release

# 启用 CUDA 支持构建
cargo build --release --features cuda

# 启用 Metal 支持构建 (MacOS)
cargo build --release --features metal
```

### 环境变量
- `ASR_SQLITE_PATH` SQLite 路径 (默认: `sqlite://./asr_data/database/storage.db?mode=rwc`)
- `ASR_AUDIO_PATH` 音频文件路径 (默认: `./asr_data/audio/`)
- `ETCD_DEFAULT_ENDPOINT` Etcd 端点 (默认: `http://localhost:2379`)
- `ASR_MODEL_PATH` Whisper 模型路径 (默认: `./models/ggml-large-v3.bin`)

### 运行服务器

#### 标准运行 (CPU)
```bash
cargo run --release
```

#### 使用 CUDA 运行
```bash
cargo run --release --features cuda
```

#### 使用 Metal 运行 (MacOS)
首先设置 Metal 资源路径:
```bash
export GGML_METAL_PATH_RESOURCES="./resources"
cargo run --release --features metal
```

### Docker Compose 快速启动
> docker 目前仅支持 Linux CUDA x86_64 平台

使用 Docker Compose 是最简单的启动方式:

1. 创建必要的目录:
```bash
mkdir -p models asr_data/audio asr_data/database
```

2. 下载 Whisper 模型:
```bash
./script/download-ggml-model.sh
```

3. 启动服务器:
```bash
# 标准版本
docker-compose up -d

# 启用 CUDA 支持
ASR_FEATURES=cuda docker-compose up -d

# 启用 Metal 支持 (MacOS)
ASR_FEATURES=metal docker-compose up -d
```

服务器将在以下地址提供服务:
- REST API: http://localhost:7200
- gRPC: localhost:7300

## 使用示例

### gRPC 客户端测试
```bash
# 使用本地 wav 文件
cargo run --example asr_client -- -i 2.wav

# 指定服务器地址
cargo run --example asr_client -- -i test/2.wav -s http://127.0.0.1:7300

# 指定设备 ID
cargo run --example asr_client -- -i input.wav -d test-device
```

### REST API 示例

#### 创建转录任务
```bash
curl -X POST http://localhost:7200/api/v1/asr/tasks \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "https://example.com/audio.wav"}'
```

#### 查询任务状态
```bash
curl http://localhost:7200/api/v1/asr/tasks/{task_id} \
  -H "Authorization: Bearer your-api-key"
```

## 配置说明

### 模型选择
服务器支持多种 Whisper 模型大小。你可以从 Hugging Face 下载不同的模型:
https://huggingface.co/ggerganov/whisper.cpp/tree/main

### 性能调优
- CUDA: 根据 GPU 内存调整批处理大小和工作线程数
- Metal: 确保正确配置资源路径

## 贡献
欢迎提交 Pull Request 来贡献代码！

## 许可证
本项目采用 Apache License 2.0 许可证。详见 [LICENSE](../LICENSE) 文件。

## 致谢
- OpenAI Whisper
- whisper.cpp
- whisper-rs 