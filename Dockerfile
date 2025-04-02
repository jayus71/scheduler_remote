# 构建阶段
FROM python:3.11-slim AS builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 poetry
RUN pip install poetry==1.7.1

# 复制项目文件
COPY pyproject.toml poetry.lock ./
COPY app ./app

# 配置 poetry 不创建虚拟环境
RUN poetry config virtualenvs.create false

# 安装依赖
RUN poetry install --no-dev --no-interaction --no-ansi

# 运行阶段
FROM python:3.11-slim

WORKDIR /app

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制安装好的依赖和应用代码
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

# 复制其他必要文件
COPY tests /app/tests
# 复制kubeconfig文件（如果存在）
COPY kubeconfig.yaml* /app/

# 创建README.md文件，如果不存在
RUN echo "# 星载微服务协同系统" > /app/README.md

# 创建必要的目录
RUN mkdir -p /app/config /app/logs /app/.kube

# 创建非root用户和必要的目录
RUN useradd -m appuser && \
    chown -R appuser:appuser /app

USER appuser

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV KUBECONFIG=/app/.kube/config

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# 暴露HTTP API端口、UDP服务端口
EXPOSE 8000
EXPOSE 9001/udp
EXPOSE 9002/udp

# 启动命令
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "20", "--ws-ping-timeout", "20"]