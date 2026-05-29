FROM python:3.12-slim

WORKDIR /app

RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    --default-timeout=120

COPY . .

RUN mkdir -p data/audio_files data/models

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "web_server.py"]
