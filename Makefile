# 基础配置
BINARY_NAME=asr-rs
VERSION=$(shell git describe --tags --always --dirty)
DOCKER_REGISTRY=bean
DOCKER_IMAGE=$(DOCKER_REGISTRY)/$(BINARY_NAME)

# 检测操作系统和架构
OS := $(shell uname -s)
ARCH := $(shell uname -m)

# 交叉编译目标
TARGET=x86_64-unknown-linux-musl

# 检测操作系统和架构，自动选择特性
ifeq ($(OS),Darwin)
    FEATURES=metal
else ifeq ($(OS),Linux)
    FEATURES=cuda
endif

.PHONY: setup build push clean run run-cuda run-metal

# 首次使用时安装必要工具
setup:
ifeq ($(OS),Darwin)
	# macOS 环境
	rustup target add $(TARGET)
	brew install FiloSottile/musl-cross/musl-cross
else ifeq ($(OS),Linux)
ifeq ($(ARCH),x86_64)
	# Linux x86_64 环境
	rustup target add $(TARGET)
	apt-get update && apt-get install -y musl-tools
else
	# Linux ARM 环境
	rustup target add $(TARGET)
	apt-get update && apt-get install -y musl-tools gcc-x86-64-linux-gnu
endif
endif

# 构建：根据不同环境选择编译方式
build:
ifeq ($(OS),Darwin)
ifeq ($(ARCH),arm64)
	# macOS ARM (M1/M2) 环境
	RUSTFLAGS="-C linker=/opt/homebrew/bin/x86_64-linux-musl-gcc" \
	CC_x86_64_unknown_linux_musl=/opt/homebrew/bin/x86_64-linux-musl-gcc \
	CXX_x86_64_unknown_linux_musl=/opt/homebrew/bin/x86_64-linux-musl-g++ \
	cargo build --target $(TARGET) --release
else
	# macOS Intel 环境
	RUSTFLAGS="-C linker=/usr/local/bin/x86_64-linux-musl-gcc" \
	CC_x86_64_unknown_linux_musl=/usr/local/bin/x86_64-linux-musl-gcc \
	CXX_x86_64_unknown_linux_musl=/usr/local/bin/x86_64-linux-musl-g++ \
	cargo build --target $(TARGET) --release
endif
else ifeq ($(OS),Linux)
ifeq ($(ARCH),x86_64)
	# Linux x86_64 环境，直接编译
	cargo build --release
else
	# Linux ARM 环境，使用交叉编译
	RUSTFLAGS="-C linker=x86_64-linux-gnu-gcc" \
	CC_x86_64_unknown_linux_musl=x86_64-linux-gnu-gcc \
	CXX_x86_64_unknown_linux_musl=x86_64-linux-gnu-g++ \
	cargo build --target $(TARGET) --release
endif
endif
	# 构建 Docker 镜像
	docker build -f dockerfile.cuda -t $(DOCKER_IMAGE):$(VERSION) -t $(DOCKER_IMAGE):latest .

# 推送镜像
push: build
	docker push $(DOCKER_IMAGE):$(VERSION)
	docker push $(DOCKER_IMAGE):latest

# 清理构建产物
clean:
	cargo clean
	rm -rf target/

# 显示当前环境信息
info:
	@echo "Operating System: $(OS)"
	@echo "Architecture: $(ARCH)"
	@echo "Target: $(TARGET)"

# 运行目标
run:
	cargo run --release --features $(FEATURES)

# 强制使用 CUDA 运行
run-cuda:
	cargo run --release --features cuda

# 强制使用 Metal 运行
run-metal:
	cargo run --release --features metal