FROM --platform=$TARGETPLATFORM alpine:3.19

RUN apk add --no-cache \
    ffmpeg \
    ca-certificates \
    tzdata

RUN if [ "$(uname -m)" = "x86_64" ]; then \
        apk add --no-cache cuda-runtime-cuda cuda-cudart; \
    fi

RUN apk update && apk add --no-cache \
    gcc-x86-64-linux-gnu \
    g++-x86-64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

RUN adduser -D -u 1000 app

RUN mkdir -p /app/data /app/config /app/asr/data \
    && chown -R app:app /app

WORKDIR /app

COPY --chown=app:app target/*/release/asr-rs /app/


ENV RUST_LOG=info

USER app


CMD ["/app/asr-rs"]
