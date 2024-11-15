# basic configuration
BINARY_NAME=asr-rs
VERSION=$(shell git describe --tags --always --dirty)
DOCKER_REGISTRY=bean
DOCKER_IMAGE=$(DOCKER_REGISTRY)/$(BINARY_NAME)

# detect operating system and architecture
OS := $(shell uname -s)
ARCH := $(shell uname -m)

# cross-compile target
TARGET=x86_64-unknown-linux-musl

# detect operating system and architecture, automatically select features
ifeq ($(OS),Darwin)
    FEATURES=metal
else ifeq ($(OS),Linux)
    FEATURES=cuda
endif

.PHONY: setup build push clean run run-cuda run-metal

# install necessary tools when first used
setup:
ifeq ($(OS),Darwin)
	# macOS environment
	rustup target add $(TARGET)
	brew install FiloSottile/musl-cross/musl-cross
else ifeq ($(OS),Linux)
ifeq ($(ARCH),x86_64)
	# Linux x86_64 environment
	rustup target add $(TARGET)
	apt-get update && apt-get install -y musl-tools
else
	# Linux ARM environment
	rustup target add $(TARGET)
	apt-get update && apt-get install -y musl-tools gcc-x86-64-linux-gnu
endif
endif

# build: according to different environments, choose different compilation methods
build:
ifeq ($(OS),Darwin)
ifeq ($(ARCH),arm64)
	# macOS ARM (M1/M2) environment
	RUSTFLAGS="-C linker=/opt/homebrew/bin/x86_64-linux-musl-gcc" \
	CC_x86_64_unknown_linux_musl=/opt/homebrew/bin/x86_64-linux-musl-gcc \
	CXX_x86_64_unknown_linux_musl=/opt/homebrew/bin/x86_64-linux-musl-g++ \
	cargo build --target $(TARGET) --release
else
	# macOS Intel environment
	RUSTFLAGS="-C linker=/usr/local/bin/x86_64-linux-musl-gcc" \
	CC_x86_64_unknown_linux_musl=/usr/local/bin/x86_64-linux-musl-gcc \
	CXX_x86_64_unknown_linux_musl=/usr/local/bin/x86_64-linux-musl-g++ \
	cargo build --target $(TARGET) --release
endif
else ifeq ($(OS),Linux)
ifeq ($(ARCH),x86_64)
	# Linux x86_64 environment, directly compile
	cargo build --release
else
	# Linux ARM environment, use cross-compile
	RUSTFLAGS="-C linker=x86_64-linux-gnu-gcc" \
	CC_x86_64_unknown_linux_musl=x86_64-linux-gnu-gcc \
	CXX_x86_64_unknown_linux_musl=x86_64-linux-gnu-g++ \
	cargo build --target $(TARGET) --release
endif
endif
# build docker image
# only support linux x86_64 now
	docker build -f dockerfile.cuda -t $(DOCKER_IMAGE):$(VERSION) -t $(DOCKER_IMAGE):latest .

# push image
push: build
	docker push $(DOCKER_IMAGE):$(VERSION)
	docker push $(DOCKER_IMAGE):latest

# clean build artifacts
clean:
	cargo clean
	rm -rf target/

# display current environment information
info:
	@echo "Operating System: $(OS)"
	@echo "Architecture: $(ARCH)"
	@echo "Target: $(TARGET)"

# run target
run:
	cargo run --release --features $(FEATURES)

# run with CUDA
run-cuda:
	cargo run --release --features cuda

# run with Metal
run-metal:
	cargo run --release --features metal