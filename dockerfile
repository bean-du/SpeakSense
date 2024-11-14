# 使用 alpine 作为基础镜像
FROM --platform=$TARGETPLATFORM alpine:3.19

# 安装必要的系统依赖
RUN apk add --no-cache \
    ffmpeg \
    ca-certificates \
    tzdata

# 根据架构安装额外依赖
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        apk add --no-cache cuda-runtime-cuda cuda-cudart; \
    fi

# 添加交叉编译支持
RUN apt-get update && apt-get install -y \
    gcc-x86-64-linux-gnu \
    g++-x86-64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN adduser -D -u 1000 app

# 创建必要的目录
RUN mkdir -p /app/data /app/config /app/asr/data \
    && chown -R app:app /app

# 设置工作目录
WORKDIR /app

# 复制编译好的二进制文件
COPY --chown=app:app target/*/release/asr-rs /app/

# 创建默认配置文件
RUN echo '{}' > /app/config/config.json && chown app:app /app/config/config.json

# 设置环境变量
ENV RUST_LOG=info

# 切换到非 root 用户
USER app


# 设置启动命令
CMD ["/app/asr-rs"]
