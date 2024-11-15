#!/bin/bash

# 检测操作系统和架构
OS=$(uname -s)
ARCH=$(uname -m)

# 设置默认值
DOCKERFILE="dockerfile.cpu"
ASR_FEATURES="cpu"
PLATFORM_TAG="latest"
DOCKER_RUNTIME="none"

# 检测 NVIDIA GPU
if [ "$OS" = "Linux" ] && [ "$ARCH" = "x86_64" ] && command -v nvidia-smi >/dev/null 2>&1; then
    DOCKERFILE="dockerfile.cuda"
    ASR_FEATURES="cuda"
    PLATFORM_TAG="cuda"
    DOCKER_RUNTIME="nvidia"
else
    echo "No NVIDIA GPU detected or not on Linux x86_64, using CPU version"
fi

# 输出检测到的配置
echo "Detected platform: $OS-$ARCH"
echo "Using dockerfile: $DOCKERFILE"
echo "Features: $ASR_FEATURES"
echo "Platform tag: $PLATFORM_TAG"
echo "Docker runtime: $DOCKER_RUNTIME"

# 导出环境变量
export DOCKERFILE
export ASR_FEATURES
export PLATFORM_TAG
export DOCKER_RUNTIME

# 启动 docker-compose
docker-compose up -d "$@" 