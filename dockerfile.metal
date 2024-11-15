# Build stage
FROM rust:1.75-slim-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    cmake \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Create a new empty shell project
WORKDIR /app
COPY . .

# Build with Metal support
ARG FEATURES=metal
RUN cargo build --release --features ${FEATURES}

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directories
WORKDIR /app
RUN mkdir -p /app/models /app/asr_data/audio /app/asr_data/database

# Copy the Metal resources
COPY resources /app/resources
ENV GGML_METAL_PATH_RESOURCES=/app/resources

# Copy the binary from builder
COPY --from=builder /app/target/release/asr-rs /app/asr-rs
COPY --from=builder /app/script/download-ggml-model.sh /app/script/

# Set environment variables
ENV ASR_SQLITE_PATH=sqlite:///app/asr_data/database/storage.db?mode=rwc
ENV ASR_AUDIO_PATH=/app/asr_data/audio
ENV ASR_MODEL_PATH=/app/models/ggml-large-v3.bin
ENV RUST_LOG=info

# Expose ports
EXPOSE 7200
EXPOSE 7300

# Download model if not exists and start the server
CMD if [ ! -f ${ASR_MODEL_PATH} ]; then \
        ./script/download-ggml-model.sh; \
    fi && \
    ./asr-rs 